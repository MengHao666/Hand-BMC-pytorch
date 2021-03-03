import numpy as np
SNAP_PARENT = [
    0,  # 0's parent
    0,  # 1's parent
    1,
    2,
    3,
    0,  # 5's parent
    5,
    6,
    7,
    0,  # 9's parent
    9,
    10,
    11,
    0,  # 13's parent
    13,
    14,
    15,
    0,  # 17's parent
    17,
    18,
    19,
]

JOINT_ROOT_IDX = 9

REF_BONE_LINK = (0, 9)  # mid mcp

# bone indexes in 20 bones setting
ID_ROOT_bone = np.array([0, 4, 8, 12, 16])  # ROOT_bone from wrist to MCP
ID_PIP_bone = np.array([1, 5, 9, 13, 17])  # PIP_bone from MCP to PIP
ID_DIP_bone = np.array([2, 6, 10, 14, 18])  # DIP_bone from  PIP to DIP
ID_TIP_bone = np.array([3, 7, 11, 15, 19])  # TIP_bone from DIP to TIP
