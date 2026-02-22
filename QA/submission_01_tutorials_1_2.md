# Submission 01 - Structured Answers (Tutorials 1-2)

## Scope

This document consolidates answers for Tutorial 1 and Tutorial 2 in a concise, technical format.

## Tutorial 1

### Q1. Are all frames in Figure 1 correct?

**Answer:** Not all of them are correct.

A valid frame must satisfy a right-handed convention:

- `z = x × y`

If a frame violates this orientation consistency, it is incorrect and must be flipped/redefined.

### Q2. During cage spinning, which component of transformed twist stays constant?

**Answer:** The angular component is invariant under a change of reference point (with the same orientation), while the linear component changes.

Reason:

- Angular velocity is shared by all points on a rigid body.
- Linear velocity depends on point offset: `v' = v + ω × r`.

So when expressing the same rigid-body motion at another point, the linear part changes and the angular part does not.

### Q3. During cage spinning, which component of transformed wrench stays constant?

**Answer:** For a pure change of reference point (no axis reorientation), the **force** component stays the same, while the **torque** changes.

Reason:

- Force is point-independent under pure translation of the frame origin.
- Torque depends on moment arm: `τ' = τ + r × f`.

So the torque part varies with reference point, but the force part is preserved.

### Q4. How to verify wrench transformation implementation?

**Answer:** Use two independent methods and compare numerically.

Method A:

- Pinocchio built-in transformation: `T.actInv(F)`

Method B:

- Manual adjoint-based transform using matrix form (`Ad_T^T` for wrench convention)

Validation criterion:

- Compute `||F_A - F_B||`
- If near machine precision (for example `< 1e-10`), implementation is consistent.

## Tutorial 2

### Q. Is `t2_temp.py` a ROS node?

**Answer:** No.

`ros_legged_robot/bullet_sims/bullet_sims/t2_temp.py` is a standalone simulation/control script:

- It does not initialize `rclpy`.
- It does not create a class derived from `rclpy.node.Node`.
- It does not use ROS topic/service communication as its core loop.

It primarily runs PyBullet + Pinocchio computations directly.
