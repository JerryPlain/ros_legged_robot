"""
TF transformations functions
R -> quaternion, Euler -> quaternion
e.g. usage:
q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
"""
import math
import numpy as np

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    Input
      :param roll: rotation around x-axis in radians (float)
      :param pitch: rotation around y-axis in radians (float)
      :param yaw: rotation around z-axis in radians (float)
    Output
      :return: quaternion [x, y, z, w]
    """
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - \
         math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + \
         math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - \
         math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + \
         math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)

    return [qx, qy, qz, qw]


def quaternion_from_matrix(matrix):
    """
    Return quaternion from rotation matrix.
    Args:
        matrix: 4x4 or 3x3 rotation matrix
    Returns:
        [x, y, z, w]
    """
    M = np.array(matrix, dtype=np.float64)[:4, :4]
    q = np.empty((4, ), dtype=np.float64)
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i = np.argmax([M[0, 0], M[1, 1], M[2, 2]])
        if i == 0:
            q[0] = M[0, 0] - (M[1, 1] + M[2, 2]) + M[3, 3]
            q[1] = M[1, 0] + M[0, 1]
            q[2] = M[0, 2] + M[2, 0]
            q[3] = M[2, 1] - M[1, 2]
        elif i == 1:
            q[1] = M[1, 1] - (M[2, 2] + M[0, 0]) + M[3, 3]
            q[2] = M[2, 1] + M[1, 2]
            q[0] = M[1, 0] + M[0, 1]
            q[3] = M[0, 2] - M[2, 0]
        else:
            q[2] = M[2, 2] - (M[0, 0] + M[1, 1]) + M[3, 3]
            q[0] = M[0, 2] + M[2, 0]
            q[1] = M[2, 1] + M[1, 2]
            q[3] = M[1, 0] - M[0, 1]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q[:4]