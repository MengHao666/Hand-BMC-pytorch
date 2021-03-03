import matplotlib.pyplot as plt
import numpy as np
from rdp import rdp
from tqdm import tqdm

import utils.hull as hull

joint_angles = np.load("BMC/joint_angles.npy")
print(joint_angles.shape)
EPSILON = 0.0005

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky(Little)"]
BONE_NAMES = ["PIP", "DIP", "TIP"]  # i.e. PIP bone means the bone from MCP joint to PIP joint

del_rdp_hulls = np.load("BMC/CONVEX_HULLS.npy", allow_pickle=True)

fig = plt.figure()
plt.suptitle("CONVEX HULLS of JOINT ANGLES")

for ID in tqdm(range(15)):
    ja = joint_angles[:, ID]
    ja_list = [tuple(x) for x in ja.tolist()]

    convex = hull.convex(ja_list)
    convex = np.array(convex)
    convex_list = convex.tolist()

    rdp_convex = rdp(convex_list, epsilon=EPSILON)
    rdp_convex = np.array(rdp_convex)

    del_rdp_hull = del_rdp_hulls[ID]

    ax = fig.add_subplot(3, 5, ID + 1)
    ax.scatter(ja[:, 0], ja[:, 1], s=0.05, c='r')
    ax.set_xlabel("flexion")
    ax.set_ylabel("abduction")
    ax.set_title('ID:{}  Finger:{}  Bone:{}'.format(ID, FINGER_NAMES[ID % 5], BONE_NAMES[int(ID / 5)]))
    plt.plot(np.append(convex[:, 0], convex[0, 0]), np.append(convex[:, 1], convex[0, 1]), '--ro', linewidth=3,label='ori_convex_hull')
    plt.plot(np.append(rdp_convex[:, 0], rdp_convex[0, 0]), np.append(rdp_convex[:, 1], rdp_convex[0, 1]), '--gv',
             linewidth=2, label='rdp_convex_hull')
    plt.plot(np.append(del_rdp_hull[:, 0], del_rdp_hull[0, 0]),
             np.append(del_rdp_hull[:, 1], del_rdp_hull[0, 1]),
             '-b*', linewidth=1, label='del_convex_hull')
    plt.xticks(np.arange(-3, 4, 1))
    plt.yticks(np.arange(-2, 2, 0.5))

    ja_min = np.min(ja, axis=0)
    ja_max = np.max(ja, axis=0)

    plt.gca().add_patch(
        plt.Rectangle((ja_min[0], ja_min[1]), ja_max[0] - ja_min[0], ja_max[1] - ja_min[1], edgecolor="yellow",
                      fill=False, linewidth=2, label='rectangle'))

plt.legend(title='Convex hull category:')

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.subplots_adjust(left=0.125, bottom=0.05, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
plt.show()
