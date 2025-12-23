# Overview of Tutorials
## T1
### SE(3) & TF:
用 pinocchio.SE3 表示立方体几何结构，用 twist + exp6 在 SE(3) 上积分中心位姿，并通过 ROS2 tf2 广播完整的 TF 树
1. Node的生命周期：节点入口
```bash
class T11Node(Node): # create a NOS2 node named t11_node
    def __init__(self):
        super().__init__('t11_node')
```
没有Node就没有clock, timer, logger, TF Bradcaster

2. 角点的位置offsets: 定义“立方体几何” （cage）
```bash
L = 0.4
H = 0.5
offsets = [
  [-L, -L, -H], [ L, -L, -H],
  [-L,  L, -H], [ L,  L, -H],
  [-L, -L,  H], [ L, -L,  H],
  [-L,  L,  H], [ L,  L,  H],
]
```
在中心Oc的局部坐标系中，定义8个角点的坐标

3. 用pin.SE3(R,p)来构造8个角点frame（SE3的数学表示）
```bash
R = np.eye(3) # rotation matrix 把角点 frame 的旋转都设为identity
p = np.array(offset) 
T = pin.SE3(R, p)
self.frames.append(T)
```
把“角点坐标”包装成 SE(3) 变换（位姿）
为什么必须用 SE(3)：之后要做 twist/wrench 的坐标变换，Pinocchio 的接口都围绕 SE3 / Motion / Force

4. self.T_oc: 中心位姿 world->Oc
```bash
self.T_oc = pin.SE3.Identity() # 定义并存储中心 frame Oc 在世界系的位姿 T_oc
self.dt = 0.1
```
角点 O0~O7 都是相对 Oc 固定的；Oc 动，整个 cage 才会动
T_oc 表示的是 world→Oc（后面发布 TF 就是这么用的），别搞反。

5. TF broadcaster + timer: let system update all the time
```bash
self.br = TransformBroadcaster(self) # TransformBroadcaster：负责把 TF 发到 /tf（动态 TF）
self.timer = self.create_timer(0.1, self.broadcast_frames) # timer：每 0.1 秒调用一次 broadcast_frames()
```
ROS2 是事件循环模型；不写 timer，TF 只发一次或根本不发

6. use now = clock.now(): all the TF unified the timestamp
```bash
now = self.get_clock().now().to_msg() # 获取当前 ROS 时间，用作 TF 的 header.stamp
```
rviz 在某个时间点查询 TF tree，如果时间戳不合理，会出现 “No transform”。不同 TF stamp 相差太多，会导致 TF tree 断裂。

7. Twist：用 6D 速度描述“刚体瞬时运动”（se(3)）
```bash
twist_vec = np.array([0., 0., 0.3, 0.01, 0., 0.]) # twist_vec：定义角速度 ω 和线速度 v
twist = pin.Motion(twist_vec) # pin.Motion：把它变成 Pinocchio 的 se(3) 元素（有语义的 twist）
```
exp6() 的输入期望是 se(3) 量（Motion），这样 pinocchio 才知道如何做指数映射
- 顺序必须是 [wx, wy, wz, vx, vy, vz]（Pinocchio 的 Motion 默认这样）
- ω 和 v 的单位别混：rad/s vs m/s

8. exp6(twist*dt)：把速度积分成“这一小步 SE(3) 位姿增量”（最核心）
```bash
delta_T = pin.exp6(twist * self.dt) # 计算在 dt 时间内，从速度得到的位姿变化 ΔT（SE(3)）。
```
为什么必须 exp6：旋转不是线性空间，不能用 R += ω dt 这种简单加法; exp6 保证得到的 R 仍然是正交矩阵（合法旋转）

9. T_oc = T_oc * delta_T：位姿更新的“乘法方向”决定物理含义
```bash
self.T_oc = self.T_oc * delta_T
```
把增量 ΔT 叠加到当前世界位姿 T_oc 上，得到新位姿。
为什么是右乘：
代表“在 Oc 自身坐标系下的运动”（body twist integration）
直觉：车头朝哪就往哪走（随着自己转向改变前进方向）
如果写成左乘会怎样：
T_oc = delta_T * T_oc：相当于“在 world 坐标系下的运动”，物理含义完全不同（会出现你转了但仍沿 world x 走的感觉）

10. 从 SE(3) 提取 (p, R)：准备转换成 ROS 的 TF 消息格式
```bash
p = self.T_oc.translation
R = self.T_oc.rotation
```
为什么要拆出来：ROS TF 消息字段是 translation 和 quaternion，不能直接塞 SE3 对象。

11. R → quaternion：TF 只能发四元数
```bash
T_matrix = np.vstack((np.hstack((R, np.zeros((3,1)))), np.array([[0,0,0,1]])))
q = tf_transformations.quaternion_from_matrix(T_matrix)
```
做什么：把 3×3 旋转矩阵 R 变成 quaternion (x,y,z,w)。
为什么要拼 4×4：很多 quaternion 工具函数要求输入 homogeneous matrix。
坑：
- 这里右侧拼的是 np.zeros((3,1))，等价于平移为 0，只用于提取旋转；OK。
- q 的顺序确认是 [x,y,z,w]（tf_transformations 通常是这样）

12. 发布 TF：world → Oc（动态）
```bash
t_center.header.frame_id = 'world'
t_center.child_frame_id = 'Oc'
t_center.transform.translation = p
t_center.transform.rotation = q
self.br.sendTransform(t_center)
```
做什么：发布中心 frame 的动态位姿。
为什么关键：这是整个 tf tree 的根链接，rviz 里 Oc 的运动就靠它。
坑：
frame_id / child_frame_id 拼写错一个字符，TF tree 就断了。
rviz 的 Fixed Frame 要设成 world（或你实际 root）

13. 发布 TF：Oc → O0~O7（静态相对位姿，但你用动态 broadcaster重复发）
```bash
for i, T in enumerate(self.frames):
    tf.header.frame_id = 'Oc'
    tf.child_frame_id = f'O{i}'
    tf.transform.translation = T.translation
    tf.transform.rotation = identity
    self.br.sendTransform(tf)
```
做什么：把 8 个角点 frame 挂在 Oc 下面。

为什么这样做：角点相对 Oc 固定，Oc 动则角点整体随动。
一个值得注意的优化点：
- 这 8 个 TF 其实是静态 TF（不随时间变），严格来说更适合用 StaticTransformBroadcaster
- 但教程允许你重复发（rviz 也能正常显示），只是多占一点带宽/CPU

14. print yaw：验证旋转是否在发生（debug）
```bash
yaw = np.arctan2(R[1,0], R[0,0]) # 从旋转矩阵提取 yaw（绕 z 的角）。定义 ωz=0.3，yaw 应该稳定线性增长（每秒+0.3 rad）。
```

15. main / spin：让 timer 回调真的跑起来
```bash
rclpy.init()
node = T11Node()
rclpy.spin(node)
rclpy.shutdown()
```
初始化 ROS → 创建节点 → 进入事件循环 → Ctrl+C 后清理。