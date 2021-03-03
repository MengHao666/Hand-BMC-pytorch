import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from rdp import rdp

from utils import hull

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky(Little)"]
BONE_NAMES = ["PIP", "DIP", "TIP"]


def count_num_in_convex_hull(points_, convex_hull_):
    '''

    :param points_: points in plane, N*2
    :param convex_hull_: list of points to construct a convex hull , M*2
    :return: numbers of points in convex hull
    '''
    points = points_.copy()
    convex_hull = convex_hull_.copy()
    convex_hull = np.append(convex_hull, [convex_hull[0]], axis=0)

    v = convex_hull[1:] - convex_hull[:-1]  # NUM*2
    v = np.tile(v, (points_.shape[0], 1, 1))

    w = -np.tile(convex_hull[:-1], (points_.shape[0], 1, 1)) + points[:, np.newaxis, :]  # N*NUM*2
    # 2d cross product (w1, w2) x (v1, v2) := w1*v2 - w2*v1
    cross_product_2d = w[:, :, 0] * v[:, :, 1] - w[:, :, 1] * v[:, :, 0]
    tmp = (cross_product_2d < 1e-6).all(axis=-1).sum()

    return tmp


def calculate_convex_hull(joint_angles, args):
    all_del_hulls = []
    for ID in range(15):
        print('*' * 40)
        ja = joint_angles[:, ID]
        ja_list = [tuple(x) for x in ja.tolist()]

        convex = hull.convex(ja_list)
        convex = np.array(convex)
        print("ori_hull.shape=", convex.shape)

        convex_list = convex.tolist()

        rdp_convex = rdp(convex_list, epsilon=args.epsilon)
        rdp_convex = np.array(rdp_convex)
        print("rdp_hull.shape=", rdp_convex.shape)

        dep_hull = rdp_convex.copy()
        for i in range(rdp_convex.shape[0]):
            tmp_index = np.argwhere(dep_hull == rdp_convex[i])[0][0]
            tmp_num = count_num_in_convex_hull(ja, np.delete(dep_hull, tmp_index, axis=0))
            tmp_ratio = tmp_num / ja.shape[0]
            print("tmp_ratio=", tmp_ratio)

            if tmp_ratio > args.ratio:
                dep_hull = np.delete(dep_hull, tmp_index, axis=0)
            elif args.ratio - args.delta <= tmp_ratio <= args.ratio:
                dep_hull = np.delete(dep_hull, tmp_index, axis=0)
                break
            else:
                break
        print("dep_hull.shape=", dep_hull.shape)
        all_del_hulls.append(dep_hull)

        if args.visualize:
            fig = plt.figure()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            ax = fig.add_subplot(111)
            ax.scatter(ja[:, 0], ja[:, 1], s=0.05, c='r')
            ax.set_xlabel("flexion")
            ax.set_ylabel("abduction")
            ax.set_title('ID:{}       Finger:{}      Bone:{}'.format(ID, FINGER_NAMES[ID % 5], BONE_NAMES[int(ID / 5)]))

            plt.plot(np.append(convex[:, 0], convex[0, 0]), np.append(convex[:, 1], convex[0, 1]), 'ro--', linewidth=3,
                     label='ori_convex_hull')
            plt.plot(np.append(rdp_convex[:, 0], rdp_convex[0, 0]), np.append(rdp_convex[:, 1], rdp_convex[0, 1]),
                     'gv--',
                     linewidth=2, label='rdp_convex_hull')
            plt.plot(np.append(dep_hull[:, 0], dep_hull[0, 0]),
                     np.append(dep_hull[:, 1], dep_hull[0, 1]),
                     '-b*', linewidth=1, label='del_convex_hull')

            ja_min = np.min(ja, axis=0)
            ja_max = np.max(ja, axis=0)

            plt.gca().add_patch(
                plt.Rectangle((ja_min[0], ja_min[1]), ja_max[0] - ja_min[0], ja_max[1] - ja_min[1], edgecolor="yellow",
                              fill=False, linewidth=2, label='rectangle'))

            plt.xticks(np.arange(-3, 4, 1))
            plt.yticks(np.arange(-2, 2, 0.5))
            plt.legend(title='Convex hull category:')

            plt.show()

        # check if hull_test is in counter-clockwise order
        # hull_test = convex
        # for END in range(
        #         hull_test.shape[0]):
        #     fig = plt.figure()
        #     figManager = plt.get_current_fig_manager()
        #     figManager.window.showMaximized()
        #     ax = fig.add_subplot(111)
        #     ax.scatter(ja[:, 0], ja[:, 1], s=0.1, c='r')
        #     ax.set_xlabel("flexion")
        #     ax.set_ylabel("abduction")
        #
        #     plt.plot(hull_test[:END, 0], hull_test[:END, 1], 'y--', linewidth=1)
        #     plt.xticks(np.arange(-3, 4, 1))
        #     plt.yticks(np.arange(-2, 2, 0.5))
        #     plt.show()

    all_del_hulls = np.array(all_del_hulls)
    np.save("BMC/CONVEX_HULLS.npy", all_del_hulls)
    print("all_del_hulls.shape=", all_del_hulls.shape)


def main(args):
    joint_angles = np.load(os.path.join(args.path, "joint_angles.npy"))
    print("joint_angles.shape=", joint_angles.shape)
    calculate_convex_hull(joint_angles, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='calculate convex hull for joint angles ')

    parser.add_argument(
        '-p',
        '--path',
        default='BMC',
        type=str,
        metavar='data_root',
        help='directory')

    parser.add_argument(
        '-vis',
        '--visualize',
        action='store_true',
        help='visualize reconstruction result',
        default=True
    )

    parser.add_argument(
        '--epsilon',
        type=float,
        default=5e-4,
        help='epsilon0.'
    )

    parser.add_argument(
        '--ratio',
        type=float,
        default=0.9995,
        help='ration0.'
    )

    parser.add_argument(
        '--delta',
        type=float,
        default=0.0005,
        help='ration_delta.'
    )

    main(parser.parse_args())
