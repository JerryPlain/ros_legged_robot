import numpy as np
import pinocchio as pin
import pybullet as pb

# simulator
#>>>>TODO: Fix include
from simulator.robot import Robot
from pinocchio import SE3

# whole-body controller
#>>>>TODO: Fix include
from tsid_wrapper import TSIDWrapper

# robot configs
#>>>>TODO: Fix include
import talos_conf as conf

#>>>>TODO: Fix include
from footstep_planner import Side

# ROS visualizations
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header

class Talos:
    """Talos robot
    combines wbc with pybullet, functions to read and set
    sensor values.
    """
    def __init__(self, simulator, node=None, conf=None):
        self.conf = conf if conf is not None else conf
        self.sim = simulator
        self.node = node  # Store ROS node for publishers
        
        #>>>>TODO: Like allways create the tsid wrapper for the whole body QP
        self.stack = None
        q_home = self.conf.q_home.copy()
        self.tsid = TSIDWrapper(self.conf)
        model = self.tsid.model
        
        # spawn robot in simulation
        #>>>>TODO: Create the pybullet robot in the simulatior
        self.robot = Robot(
            simulator,
            self.conf.urdf,
            model,
            q_home[:3],
            q_home[3:7],
            q=q_home,
            useFixedBase=False,
        )
        # initialize integrate variables for tsid position interface
        self.q_tsid, self.v_tsid = np.zeros_like(self.robot.q()), np.zeros_like(self.robot.v())
        
        ########################################################################
        # state
        ########################################################################
        self.support_foot = Side.RIGHT
        self.swing_foot = Side.LEFT
        
        ########################################################################
        # estimators
        ########################################################################
        self.zmp = np.zeros(3)
        self.dcm = np.zeros(3)
        
        ########################################################################
        # sensors
        ########################################################################
        # ft sensors
        #>>>>TODO: Turn on the force torque sensor in the robots feet
        pb.enableJointForceTorqueSensor(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'], True)
        pb.enableJointForceTorqueSensor(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'], True)
        
        ########################################################################
        # visualizations
        ########################################################################
        
        #>>>> TODO: joint state publisher
        # Initialize ROS2 for publishers (can be done externally)
        #>>>> joint state publisher
        self.publisher_ = self.node.create_publisher(JointState, 'joint_states', 10)
        self.tau = np.zeros(len(self.robot.actuatedJointNames()))
        
        #>>>> floating base broadcaster
        self.tf_broadcaster = TransformBroadcaster(self.node)
        
        #>>>> zmp and dcm point publisher 
        self.marker_pub = self.node.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        
        #>>>> wrench publisher for left and right foot
        self.lf_wrench_pub = self.node.create_publisher(WrenchStamped, 'left_foot_wrench', 10)
        self.rf_wrench_pub = self.node.create_publisher(WrenchStamped, 'right_foot_wrench', 10)
           
           
    def update(self):
        """update the robot
        """
        # get sim time and set t for the whole body controller
        t = self.sim.simTime()
        dt = self.sim.stepTime()

        #update the pybullet robot
        self.robot.update()
        
        # update the estimate
        self._update_zmp_estimate()
        self._update_dcm_estimate()
        
        # solve whole body control task
        self._solve(t, dt)
        
    def setSupportFoot(self, side):
        """sets the the support foot of the robot on given side
        """
        
        # The support foot is in rigid contact with the ground and should 
        # hold the weight of the robot
        self.support_foot = side
        
        #>>>> TODO: Activate the foot contact on the support foot
        #>>>> TODO: At the same time deactivate the motion task on the support foot
        if side == Side.LEFT:
            self.tsid.add_contact_LF()
            # Deactivate motion task by not updating its reference
        else:
            self.tsid.add_contact_RF()
            # Deactivate motion task by not updating its reference
    
    def setSwingFoot(self, side):
        """sets the swing foot of the robot on given side
        """
        
        # The swing foot is not in contact and can move
        self.swing_foot = side
        
        #>>>> TODO: Deactivate the foot contact on the swing foot
        #>>>> TODO: At the same time turn on the motion task on the swing foot
        if side == Side.LEFT:
            self.tsid.remove_contact_LF()
            # Motion task is activated by setting references
        else:
            self.tsid.remove_contact_RF()
            # Motion task is activated by setting references
        
    def updateSwingFootRef(self, T_swing_w, V_swing_w, A_swing_w):
        """updates the swing foot motion reference
        """
        
        #>>>> TODO: set the pose, velocity and acceleration on the swing foots
        # motion task
        if self.swing_foot == Side.LEFT:
            self.stack.set_LF_pose_ref(T_swing_w, V_swing_w, A_swing_w)
        else:
            self.stack.set_RF_pose_ref(T_swing_w, V_swing_w, A_swing_w)

    def swingFootPose(self):
        """return the pose of the current swing foot
        """
        #>>>>TODO: return correct foot pose
        if self.swing_foot == Side.LEFT:
            return self.tsid.get_placement_LF()
        else:
            return self.tsid.get_placement_RF()
    
    def supportFootPose(self):
        """return the pose of the current support foot
        """
        #>>>>TODO: return correct foot pose
        if self.support_foot == Side.LEFT:
            return self.tsid.get_placement_LF()
        else:
            return self.tsid.get_placement_RF()

    def publish(self):        
        #>>>> TODO: publish the jointstate
        joint_pos = self.robot.actuatedJointPosition()
        joint_vel = self.robot.actuatedJointVelocity()
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.robot.actuatedJointNames()
        msg.position = joint_pos.tolist()
        msg.velocity = joint_vel.tolist()
        msg.effort = self.tau.tolist()
        self.publisher_.publish(msg)
        
        #>>>> TODO: broadcast odometry
        T_b_w, _ = self.tsid.baseState()
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.node.get_clock().now().to_msg()
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link"
        trans = T_b_w.translation
        rot = pin.Quaternion(T_b_w.rotation)
        tf_msg.transform.translation.x = trans[0]
        tf_msg.transform.translation.y = trans[1]
        tf_msg.transform.translation.z = trans[2]
        tf_msg.transform.rotation.x = rot.x
        tf_msg.transform.rotation.y = rot.y
        tf_msg.transform.rotation.z = rot.z
        tf_msg.transform.rotation.w = rot.w
        self.tf_broadcaster.sendTransform(tf_msg)
        
        #>>>> TODO: publish feet wrenches
        if not hasattr(self, "wrench_left"):
            self.wrench_left = pin.Force(np.zeros(6))
            self.wrench_right = pin.Force(np.zeros(6))
        def wrench_msg(wrench, frame_id, scale_factor=0.01):
            msg = WrenchStamped()
            msg.header.frame_id = frame_id
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.wrench.force.x = float(-wrench.linear[0]) * scale_factor
            msg.wrench.force.y = float(-wrench.linear[1]) * scale_factor
            msg.wrench.force.z = float(-wrench.linear[2]) * scale_factor
            msg.wrench.torque.x = float(-wrench.angular[0]) * scale_factor
            msg.wrench.torque.y = float(-wrench.angular[1]) * scale_factor
            msg.wrench.torque.z = float(-wrench.angular[2]) * scale_factor
            return msg

        self.lf_wrench_pub.publish(wrench_msg(self.wrench_left, 'leg_left_6_link'))
        self.rf_wrench_pub.publish(wrench_msg(self.wrench_right, 'leg_right_6_link'))
        
        #>>>> TODO: publish dcm and zmp marker
        marker_array = MarkerArray()
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = "world"

        def create_marker(position, marker_id, color, ns):
            m = Marker()
            m.header = header
            m.ns = ns
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = position[0]
            m.pose.position.y = position[1]
            m.pose.position.z = position[2]
            m.pose.orientation.w = 1.0
            m.scale.x = 0.05
            m.scale.y = 0.05
            m.scale.z = 0.05
            m.color.r = color[0]
            m.color.g = color[1]
            m.color.b = color[2]
            m.color.a = 1.0
            return m

        if self.dcm is not None:
            marker_array.markers.append(create_marker(self.dcm, 0, [1.0, 0.0, 1.0], "dcm"))

        if self.zmp is not None:
            marker_array.markers.append(create_marker(self.zmp, 1, [0.0, 0.0, 1.0], "zmp"))

        self.marker_pub.publish(marker_array)

    ############################################################################
    # private funcitons
    ############################################################################

    def _solve(self, t, dt):
        # get the current state
        q = self.robot.q()
        v = self.robot.v()
        
        # solve the whole body qp
        #>>>> TODO: sovle the wbc and command the torque to pybullet robot
        torques, dv_sol = self.tsid.update(q, v, t)
        self.robot.setActuatedJointTorques(torques)
    
    def _update_zmp_estimate(self):
        """update the estimated zmp position
        """
        #>>>> TODO: compute the zmp based on force torque sensor readings
        self.data = self.robot._model.createData()
        q = self.robot.q()
        pin.framesForwardKinematics(self.robot._model, self.data, q)

        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_right_6_joint'])[2]
        wr = pin.Force(-np.array(wren))
        wren = pb.getJointState(self.robot.id(), self.robot.jointNameIndexMap()['leg_left_6_joint'])[2]
        wl = pin.Force(-np.array(wren))

        wr = self.transform_wrench_pb_to_pinocchio(wr, "leg_right_6_link", "leg_right_6_joint")
        wl = self.transform_wrench_pb_to_pinocchio(wl, "leg_left_6_link", "leg_left_6_joint")
        self.wrench_right = wr
        self.wrench_left = wl

        H_w_rsole = self.data.oMf[self.robot._model.getFrameId("right_sole_link")]
        H_w_lsole = self.data.oMf[self.robot._model.getFrameId("left_sole_link")]
        H_w_rankle = self.data.oMf[self.robot._model.getFrameId("leg_right_6_joint")]
        H_w_lankle = self.data.oMf[self.robot._model.getFrameId("leg_left_6_joint")]

        d_l = H_w_lankle.translation[2] - H_w_lsole.translation[2]
        d_r = H_w_rankle.translation[2] - H_w_rsole.translation[2]

        f_l, tau_l = wl.linear, wl.angular
        f_r, tau_r = wr.linear, wr.angular
        px_l = (-tau_l[1] - f_l[0] * d_l) / f_l[2]
        py_l = ( tau_l[0] - f_l[1] * d_l) / f_l[2]
        px_r = (-tau_r[1] - f_r[0] * d_r) / f_r[2]
        py_r = ( tau_r[0] - f_r[1] * d_r) / f_r[2]

        zmp_l_world = H_w_lsole.act(np.array([px_l, py_l, 0.0]))
        zmp_r_world = H_w_rsole.act(np.array([px_r, py_r, 0.0]))

        if self.tsid.contact_LF_active and self.tsid.contact_RF_active:
            fz_total = f_l[2] + f_r[2]
            px_total = (zmp_l_world[0] * f_l[2] + zmp_r_world[0] * f_r[2]) / fz_total
            py_total = (zmp_l_world[1] * f_l[2] + zmp_r_world[1] * f_r[2]) / fz_total
            self.zmp = np.array([px_total, py_total, 0.0])
        elif self.tsid.contact_LF_active:
            self.zmp = zmp_l_world
        elif self.tsid.contact_RF_active:
            self.zmp = zmp_r_world
        else:
            self.zmp = np.zeros(3)

        
    def _update_dcm_estimate(self):
        """update the estimated dcm position
        """
        #>>>> TODO: compute the com based on current center of mass state
        c = self.tsid.comState().value()  # CoM
        c_dot = self.tsid.comState().derivative() # CoM Vel
        h = c[2]
        omega = np.sqrt(self.conf.g / h)
        dcm = c + c_dot / omega
        self.dcm = np.array([dcm[0], dcm[1], 0.0])
    
    def transform_wrench_pb_to_pinocchio(self, wrench_pb, link_name, joint_name):
        pos, quat = pb.getLinkState(self.robot.id(), self.robot.linkNameIndexMap()[link_name], computeForwardKinematics=True)[:2]
        R_world = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3,3)
        H_pb = SE3(R_world, np.array(pos))
        wr_world = H_pb.act(wrench_pb)
        H_pin = self.data.oMf[self.robot._model.getFrameId(joint_name)].inverse()
        return H_pin.act(wr_world)