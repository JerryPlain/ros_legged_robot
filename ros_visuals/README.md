# Overview of Tutorials
## T1
### se(3), SE(3), twist, exp6, motion, pinocchio:
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
T = pin.SE3(R, p) # 这个坐标系没有旋转，方向和父坐标系完全一致，只是平移到了 p 的位置
self.frames.append(T)
```
把“角点坐标”包装成 SE(3) 变换（位姿）
为什么必须用 SE(3)：之后要做 twist/wrench 的坐标变换，Pinocchio 的接口都围绕 SE3 / Motion / Force

4. self.T_oc: 中心位姿 world -> Oc
```bash
world
 └── Oc   ← 这个就是 self.T_oc
     ├── O0
     ├── O1
     └── ...
```
如果 T_oc 初始不是 Identity，一开始就“空中生成一个立方体”。

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
twist_vec = np.array([0., 0., 0.3, 0.01, 0., 0.]) # twist_vec：定义角速度 ω 和线速度 v 它只是“数据”，不是“刚体运动”
twist = pin.Motion(twist_vec)
```
这一步等价于告诉 Pinocchio：
- “这是一个 刚体的空间速度（spatial velocity / twist），
- 属于 Lie algebra se(3)。”

Pinocchio 的约定是：
- se(3) 的元素 → 用 Motion
- SE(3) 的元素 → 用 SE3

exp6() 的输入期望是 se(3) 量（Motion），这样 pinocchio 才知道如何做指数映射
- 顺序必须是 [wx, wy, wz, vx, vy, vz]（Pinocchio 的 Motion 默认这样）
- ω 和 v 的单位别混：rad/s vs m/s

8. exp6(twist*dt)：把速度积分成“这一小步 SE(3) 位姿增量”（最核心）
```bash
twist_vec ∈ ℝ⁶
   ↓  (语义化)
ξ ∈ se(3)          ← pin.Motion
   ↓  (指数映射)
ΔT ∈ SE(3)         ← pin.exp6
   ↓  (群乘法)
T_oc ← T_oc · ΔT   ← 位姿更新
```

```bash
速度 (twist)  →  位姿变化 (delta_T) SE(3) 是 twist 的唯一合法积分结果
se(3)  --exp-->  SE(3)
Twist = 把角速度和线速度合在一起，描述刚体“此时此刻怎么动” twist = pin.Motion([ωx, ωy, ωz, vx, vy, vz])
twist 属于 se(3) se(3) 是 SE(3) 的 切空间 表示的是 位姿变化率
twist -> exp -> Delta SE3 才能更新 SE(3)。T_oc = T_oc * exp(ξ dt)
```
- ROS TF：frame = SE(3)
- Pinocchio：所有 kinematics 用 SE(3)
- SLAM：pose graph = SE(3)
- Manipulation：forward kinematics = SE(3)
- Humanoid / legged：base pose = SE(3)

```bash
关节速度 qdot
   ↓  (Jacobian)
twist ξ ∈ se(3)
   ↓  (exp)
位姿增量 ΔT ∈ SE(3)
   ↓  (群乘)
新位姿 T
```
用 SE(3) 的世界
- 位姿 = 刚体
- 运动 = 群乘法
- 速度 = Lie algebra
- 积分 = exp 映射
- TF / 机器人 / 数学 全部统一

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
T_matrix = np.vstack((np.hstack((R, np.zeros((3,1)))), np.array([[0,0,0,1]]))) # translation = 0, and last line is 0001 for homogeneous input
q = tf_transformations.quaternion_from_matrix(T_matrix) # input must be homogeneous matrix
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
t_center.transform.rotation =  # TF only publish quaternion
self.br.sendTransform(t_center)
```
做什么：发布中心 frame 的动态位姿。
为什么关键：这是整个 tf tree 的根链接，rviz 里 Oc 的运动就靠它。
坑：
frame_id / child_frame_id 拼写错一个字符，TF tree 就断了。
rviz 的 Fixed Frame 要设成 world（或你实际 root）

```bash
t_center.transform.rotation.x = q[0]
t_center.transform.rotation.y = q[1]
t_center.transform.rotation.z = q[2]
t_center.transform.rotation.w = q[3]
```

TF 不是让你“描述一个姿态”，
而是让系统“长期、反复、稳定地叠加很多姿态”。

Quaternion 是唯一能长期稳定做这件事的表示方式。
用 quaternion（TF 用的）
用 4 个数表示方向
每次转动只是：“方向 × 一个小转动

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

### adjoint transformation
```bash
def transform_twist(T: pin.SE3, V: pin.Motion) -> pin.Motion:
    return T.act(V)
```
T：从 B → A 的位姿
V：在 B 中表达的 twist
返回：在 A 中表达的 twist
你没有手写 adjoint，而是直接用 Pinocchio 的实现。

对比手写：
在 O0 定义 twist → 转到 world（重点）
```bash
self.frames = []
for i, offset in enumerate(offsets):
    R = np.eye(3)
    p = np.array(offset)
    T = pin.SE3(R, p)
    self.frames.append(T)
‵‵‵

```bash
T_c_to_O0 = self.frames[0]
T_w_to_O0 = self.T_oc * T_c_to_O0
```

### ROS2 + Pinocchio 演示「刚体运动 + 力/力矩（wrench）在不同坐标系之间正确变换」
一个在世界系中运动的刚体 Oc，带 8 个角点 O0–O7；
在某个角点施加力/力矩，用两种方式把 wrench 换到另一个角点坐标系，并验证 Pinocchio 的 actInv 和手推伴随矩阵完全一致。
```bash
self.T_oc = pin.SE3.Identity() # Oc在world中的SE3 Oc 是刚体中心坐标系
delta_T = pin.exp6(twist * dt) # Oc 在 world 里做 SE(3) 刚体运动（带角速度 + 线速度）
self.T_oc = self.T_oc * delta_T
```

每个角点：
8 个角点 O0–O7
```bash
offsets = [[±L, ±L, ±H]]
T_c_to_Oi = SE3(I, offset)
world → Oc → Oi
```
TF 在这里的真正作用是什么？
不是为了“看”，而是为了坐标一致性：
- world → Oc：在动
- Oc → Oi：静态
T_w_to_Oi = T_w_to_Oc * T_c_to_Oi


Wrench 是什么？
```bash
pin.Force(angular, linear)
angular：力矩 τ
linear ：力 f
```
这是：
- se(3)* 的元素
- 不能像向量一样直接用 R 旋转

在 O0 定义 wrench → 表达成 world
```bash
cW = pin.Force([0,0,1], [5,0,0])  # 在 O0 坐标系
wW = transform_wrensch(T_w_to_O0, cW)
```
T_w_to_O0：world → O0
actInv 的语义是：
“我有一个在 O0 表达的 wrench，把它换成 world 表达”

反过来：
```bash
wW2 = pin.Force(...)     # 在 world
T_o6_to_w = (T_w_to_O6).inverse()
c2W = transform_wrench(T_o6_to_w, wW2)
```

总结：
```bash
T_w_o = T_w_c * T_c_o
v_world = T.act(v_body)
F_new = T.actInv(F)
```
T.act(v)： 把在局部表达的速度，拉回到世界
T.actInv(F)： 把在局部表达的力，拉回到世界 （功率不变作为约束条件得到的）

**High-level summary**

This tutorial demonstrates a **fully consistent rigid-body modeling and motion framework** based on Lie groups, integrating **Pinocchio** with **ROS 2 TF**. A rigid cube is represented entirely in **SE(3)**, with its motion defined by **twists in se(3)** and integrated via the **exponential map**, guaranteeing physically correct pose updates. A complete TF tree (`world → Oc → Oi`) is continuously broadcast to maintain global frame consistency. Beyond motion, the tutorial shows how **twists and wrenches are correctly transformed between coordinate frames** using adjoint operations, ensuring invariance of physical quantities. Overall, it unifies geometry, motion, force, and visualization under one mathematically sound SE(3) pipeline—the exact abstraction used in modern robotics, SLAM, and manipulation systems.


## T2
