import matplotlib.pyplot as plt
import numpy as np

import config as cfg


def angle_between(v1, v2):
    '''
    :param v1: B*3
    :param v2: B*3
    :return: B
    '''
    v1_u = normalize(v1.copy())
    v2_u = normalize(v2.copy())

    inner_product = np.sum(v1_u * v2_u, axis=-1)
    tmp = np.clip(inner_product, -1.0, 1.0)
    tmp = np.arccos(tmp)

    return tmp


def normalize(vec_):
    '''

    :param vec:  B*3
    :return:  B*1
    '''
    vec = vec_.copy()
    len = calcu_len(vec) + 1e-8

    return vec / len


def axangle2mat(axis, angle, is_normalized=False):
    '''

    :param axis: B*3
    :param angle: B*1
    :param is_normalized:
    :return: B*3*3
    '''
    if not is_normalized:
        axis = normalize(axis)

    x = axis[:, 0];
    y = axis[:, 1];
    z = axis[:, 2]
    c = np.cos(angle);
    s = np.sin(angle);
    C = 1 - c
    xs = x * s;
    ys = y * s;
    zs = z * s
    xC = x * C;
    yC = y * C;
    zC = z * C
    xyC = x * yC;
    yzC = y * zC;
    zxC = z * xC

    Q = np.array([
        [x * xC + c, xyC - zs, zxC + ys],
        [xyC + zs, y * yC + c, yzC - xs],
        [zxC - ys, yzC + xs, z * zC + c]])
    Q = Q.transpose(2, 0, 1)

    return Q


def calcu_len(vec):
    '''
    calculate length of vector
    :param vec: B*3
    :return: B*1
    '''

    return np.linalg.norm(vec, axis=-1, keepdims=True)


def caculate_ja(joint, vis=False):
    '''

    :param joint: 21*3
    :param vis:
    :return: 15*2
    '''
    ALL_bones = np.array([
        joint[i] - joint[cfg.SNAP_PARENT[i]]
        for i in range(1, 21)
    ])
    ROOT_bones = ALL_bones[cfg.ID_ROOT_bone]  # FROM THUMB TO LITTLE FINGER
    PIP_bones = ALL_bones[cfg.ID_PIP_bone]
    DIP_bones = ALL_bones[cfg.ID_DIP_bone]
    TIP_bones = ALL_bones[cfg.ID_TIP_bone]

    ALL_Z_axis = normalize(ALL_bones)
    PIP_Z_axis = ALL_Z_axis[cfg.ID_ROOT_bone]
    DIP_Z_axis = ALL_Z_axis[cfg.ID_PIP_bone]
    TIP_Z_axis = ALL_Z_axis[cfg.ID_DIP_bone]

    normals = normalize(np.cross(ROOT_bones[1:5], ROOT_bones[0:4]))

    # ROOT bones
    PIP_X_axis = np.zeros([5, 3])  # (5,3)
    PIP_X_axis[[0, 1, 4], :] = -normals[[0, 1, 3], :]
    PIP_X_axis[2:4] = -normalize(normals[2:4] + normals[1:3])
    PIP_Y_axis = normalize(np.cross(PIP_Z_axis, PIP_X_axis))

    tmp = np.sum(PIP_bones * PIP_Y_axis, axis=-1, keepdims=True)
    PIP_bones_xz = PIP_bones - tmp * PIP_Y_axis
    PIP_theta_flexion = angle_between(PIP_bones_xz, PIP_Z_axis)  # in global coordinate
    PIP_theta_abduction = angle_between(PIP_bones_xz, PIP_bones)  # in global coordinate
    # x-component of the bone vector
    tmp = np.sum((PIP_bones * PIP_X_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    PIP_theta_flexion[tmp_index] = -PIP_theta_flexion[tmp_index]
    # y-component of the bone vector
    tmp = np.sum((PIP_bones * PIP_Y_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    PIP_theta_abduction[tmp_index] = -PIP_theta_abduction[tmp_index]

    temp_axis = normalize(np.cross(PIP_Z_axis, PIP_bones))
    temp_alpha = angle_between(PIP_Z_axis, PIP_bones)  # alpha belongs to [0, pi]
    temp_R = axangle2mat(axis=temp_axis, angle=temp_alpha, is_normalized=True)

    # DIP bones
    DIP_X_axis = np.matmul(temp_R, PIP_X_axis[:, :, np.newaxis])
    DIP_Y_axis = np.matmul(temp_R, PIP_Y_axis[:, :, np.newaxis])
    DIP_X_axis = np.squeeze(DIP_X_axis)
    DIP_Y_axis = np.squeeze(DIP_Y_axis)

    tmp = np.sum(DIP_bones * DIP_Y_axis, axis=-1, keepdims=True)
    DIP_bones_xz = DIP_bones - tmp * DIP_Y_axis
    DIP_theta_flexion = angle_between(DIP_bones_xz, DIP_Z_axis)  # in global coordinate
    DIP_theta_abduction = angle_between(DIP_bones_xz, DIP_bones)  # in global coordinate
    # x-component of the bone vector
    tmp = np.sum((DIP_bones * DIP_X_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    DIP_theta_flexion[tmp_index] = -DIP_theta_flexion[tmp_index]
    # y-component of the bone vector
    tmp = np.sum((DIP_bones * DIP_Y_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    DIP_theta_abduction[tmp_index] = -DIP_theta_abduction[tmp_index]

    temp_axis = normalize(np.cross(DIP_Z_axis, DIP_bones))
    temp_alpha = angle_between(DIP_Z_axis, DIP_bones)  # alpha belongs to [pi/2, pi]
    temp_R = axangle2mat(axis=temp_axis, angle=temp_alpha, is_normalized=True)

    # TIP bones
    TIP_X_axis = np.matmul(temp_R, DIP_X_axis[:, :, np.newaxis])
    TIP_Y_axis = np.matmul(temp_R, DIP_Y_axis[:, :, np.newaxis])
    TIP_X_axis = np.squeeze(TIP_X_axis)
    TIP_Y_axis = np.squeeze(TIP_Y_axis)

    tmp = np.sum(TIP_bones * TIP_Y_axis, axis=-1, keepdims=True)
    TIP_bones_xz = TIP_bones - tmp * TIP_Y_axis
    TIP_theta_flexion = angle_between(TIP_bones_xz, TIP_Z_axis)  # in global coordinate
    TIP_theta_abduction = angle_between(TIP_bones_xz, TIP_bones)  # in global coordinate
    # x-component of the bone vector
    tmp = np.sum((TIP_bones * TIP_X_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    TIP_theta_flexion[tmp_index] = -TIP_theta_flexion[tmp_index]
    # y-component of the bone vector
    tmp = np.sum((TIP_bones * TIP_Y_axis), axis=-1)
    tmp_index = np.where(tmp < 0)
    TIP_theta_abduction[tmp_index] = -TIP_theta_abduction[tmp_index]

    if vis:
        fig = plt.figure(figsize=[50, 50])
        ax = fig.gca(projection='3d')
        plt.plot(joint[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], 0],
                 joint[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], 1],
                 joint[[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], 2], 'yo', label='keypoint')

        plt.plot(joint[:5, 0], joint[:5, 1],
                 joint[:5, 2],
                 '--y', )
        # label='thumb')
        plt.plot(joint[[0, 5, 6, 7, 8, ], 0], joint[[0, 5, 6, 7, 8, ], 1],
                 joint[[0, 5, 6, 7, 8, ], 2],
                 '--y',
                 )
        plt.plot(joint[[0, 9, 10, 11, 12, ], 0], joint[[0, 9, 10, 11, 12], 1],
                 joint[[0, 9, 10, 11, 12], 2],
                 '--y',
                 )
        plt.plot(joint[[0, 13, 14, 15, 16], 0], joint[[0, 13, 14, 15, 16], 1],
                 joint[[0, 13, 14, 15, 16], 2],
                 '--y',
                 )
        plt.plot(joint[[0, 17, 18, 19, 20], 0], joint[[0, 17, 18, 19, 20], 1],
                 joint[[0, 17, 18, 19, 20], 2],
                 '--y',
                 )
        plt.plot(joint[4][0], joint[4][1], joint[4][2], 'rD', label='thumb')
        plt.plot(joint[8][0], joint[8][1], joint[8][2], 'r*', label='index')
        plt.plot(joint[12][0], joint[12][1], joint[12][2], 'r+', label='middle')
        plt.plot(joint[16][0], joint[16][1], joint[16][2], 'rx', label='ring')
        plt.plot(joint[20][0], joint[20][1], joint[20][2], 'ro', label='pinky')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        parent = np.array(cfg.SNAP_PARENT[1:])
        x, y, z = joint[parent, 0], joint[parent, 1], joint[parent, 2]
        u, v, w = ALL_bones[:, 0], ALL_bones[:, 1], ALL_bones[:, 2],
        ax.quiver(x, y, z, u, v, w, length=0.25, color="black", normalize=True)

        ALL_X_axis = np.stack((PIP_X_axis, DIP_X_axis, TIP_X_axis), axis=0).reshape(-1, 3)
        ALL_Y_axis = np.stack((PIP_Y_axis, DIP_Y_axis, TIP_Y_axis), axis=0).reshape(-1, 3)
        ALL_Z_axis = np.stack((PIP_Z_axis, DIP_Z_axis, TIP_Z_axis), axis=0).reshape(-1, 3)
        ALL_Bone_xz = np.stack((PIP_bones_xz, DIP_bones_xz, TIP_bones_xz), axis=0).reshape(-1, 3)

        ALL_joints_ID = np.array([cfg.ID_PIP_bone, cfg.ID_DIP_bone, cfg.ID_TIP_bone]).flatten()

        jx, jy, jz = joint[ALL_joints_ID, 0], joint[ALL_joints_ID, 1], joint[ALL_joints_ID, 2]
        ax.quiver(jx, jy, jz, ALL_X_axis[:, 0], ALL_X_axis[:, 1], ALL_X_axis[:, 2], length=0.05, color="r",
                  normalize=True)
        ax.quiver(jx, jy, jz, ALL_Y_axis[:, 0], ALL_Y_axis[:, 1], ALL_Y_axis[:, 2], length=0.10, color="g",
                  normalize=True)
        ax.quiver(jx, jy, jz, ALL_Z_axis[:, 0], ALL_Z_axis[:, 1], ALL_Z_axis[:, 2], length=0.10, color="b",
                  normalize=True)
        ax.quiver(jx, jy, jz, ALL_Bone_xz[:, 0], ALL_Bone_xz[:, 1], ALL_Bone_xz[:, 2], length=0.25, color="pink",
                  normalize=True)

        plt.legend()
        ax.view_init(-180, 90)
        plt.show()

    ALL_theta_flexion = np.stack((PIP_theta_flexion, DIP_theta_flexion, TIP_theta_flexion), axis=0).flatten()  # (15,)
    ALL_theta_abduction = np.stack((PIP_theta_abduction, DIP_theta_abduction, TIP_theta_abduction),
                                   axis=0).flatten()  # (15,)
    ALL_theta = np.stack((ALL_theta_flexion, ALL_theta_abduction), axis=1)  # (15, 2)

    return ALL_theta


if __name__ == '__main__':
    joint = np.array([[-0.6114823, 0.70412624, -0.36096475],
                      [-0.38049987, 0.303464, -0.58379203],
                      [-0.22365516, -0.05147859, -0.7509124],
                      [-0.04204354, -0.33151808, -0.92656446],
                      [0.09191624, -0.6089648, -1.0512774],
                      [0.13345791, -0.02334917, -0.24654008],
                      [0.34996158, -0.41111353, -0.16837479],
                      [0.4534959, -0.69078416, -0.12496376],
                      [0.49604133, -0.96323794, -0.10438757],
                      [0., 0., 0.],
                      [0.3559839, -0.37889394, 0.13638118],
                      [0.48572803, -0.69607633, 0.13675757],
                      [0.5390761, -0.9938516, 0.09033547],
                      [-0.14901564, 0.00647402, 0.16235268],
                      [0.19227624, -0.34850615, 0.29296255],
                      [0.37767693, -0.57762665, 0.36711285],
                      [0.27133223, -0.7816264, 0.20363101],
                      [-0.3334502, 0.0463345, 0.27828288],
                      [-0.2731263, -0.21098317, 0.49082187],
                      [-0.22576298, -0.40466458, 0.6499127],
                      [-0.16024478, -0.58365273, 0.8177859]])
    caculate_ja(joint, vis=True)
