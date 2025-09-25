#!/usr/bin/env python3
import os
import joblib
import numpy as np
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

# --- 1) Load KDE ---
MODEL_PATH = "/media/local/2_robot-behaviors/proxemic_modeling/ba_modelling-human-robot-proxemics/results/kde_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find {MODEL_PATH}.")
kde = joblib.load(MODEL_PATH)

# --- 2) Poses (x, y, yaw) ---
poses = [
    #single
    #(1.139707, -2.781449, -2.019),
    #(1.514146, -1.949175, 0.46241),
    #L shape
    #(0.9, -0.0, -1.5827),
    #(1.514146, 0.5, 2.8194)
    #side by side
    #(1.514, -0.5,  -1.5827),
    #(1.514146, 0.5, 4.3894)
    
    #face to face
    #(1.514, -0.5, 3.429),
    #(1.514146, 0.5, 0.28965) 
    
    #3 persons
    (2.078499, -3.296296,  3.106),
    (1.139707, -2.781449, -2.019),
    (1.514146, -1.949175, 0.4624)#3.604)
]

# --- 3) Helpers ---
def R_world_from_local(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    # Apply 115° anticlockwise rotation
    theta = np.deg2rad(145)  
    R115 = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return R @ R115   # extra clockwise rotation
    
def R_local_from_world(theta):
    return R_world_from_local(theta).T

def world_to_local(Xw, pose):
    x0, y0, yaw = pose
    d = Xw - np.array([x0, y0])
    return d @ R_local_from_world(yaw)

# --- 4) Grid ---
xs = [p[0] for p in poses]; ys = [p[1] for p in poses]
margin = 2.0
xmin, xmax = min(xs)-margin, max(xs)+margin
ymin, ymax = min(ys)-margin, max(ys)+margin
nx, ny = 300, 300
xv = np.linspace(xmin, xmax, nx); yv = np.linspace(ymin, ymax, ny)
XX, YY = np.meshgrid(xv, yv)
grid_world = np.column_stack([XX.ravel(), YY.ravel()])

# --- 5) Evaluate SUM ---
SUM = np.zeros(grid_world.shape[0])
for pose in poses:
    loc = world_to_local(grid_world, pose)
    SUM += np.exp(kde.score_samples(loc))
SUM = SUM.reshape(XX.shape)

# --- 6) Normalize to [0,1] with robust clipping (提升视觉对比度) ---
# --- 对 SUM 做分位裁剪和归一化 ---
# --- 1) 归一化 + 可选的分位裁剪和γ ---
p_lo, p_hi = np.percentile(SUM, [5, 99.5])
eps_floor = 0.05  # 颜色下限，防止接近0时出黑色，可调 0.02~0.1
gamma = 0.8       # γ校正，0.6~0.9 常用

S = np.clip(SUM, p_lo, p_hi)
S = (S - p_lo) / max(1e-12, (p_hi - p_lo))
S = np.power(S, 0.8)  # γ 校正，0.6~0.9 可调

# --- 2) 取一个对比度强的 colormap（如 inferno/viridis）---
import matplotlib.cm as cm
cmap = cm.get_cmap('inferno')   # 'viridis', 'plasma', 'magma', 'turbo' 也可

# --- 3) 为每个格子取颜色（RGB，不用透明度）---
# colors.append(ColorRGBA(r, g, b, 1.0))
val = S  # 0~1




# --- 7) Publish as a single CUBE_LIST marker ---
def publish_to_rviz():
    rospy.init_node("kde_sum_to_rviz", anonymous=True)
    pub = rospy.Publisher("kde_sum_cubelist", Marker, queue_size=1)
    rate = rospy.Rate(2)  # 2 Hz

    # 方块大小取网格步长
    dx = (xmax - xmin) / max(1, nx - 1)
    dy = (ymax - ymin) / max(1, ny - 1)
    cell = min(dx, dy)

    # 下采样（越大越轻，越小越细）
    stride = 2  # 可改 1/2/4/5...

    # 组装 CUBE_LIST
    m = Marker()
    m.header.frame_id = "world"         # 改成你的坐标系
    m.ns = "kde_sum"
    m.id = 0
    m.type = Marker.CUBE_LIST
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = cell
    m.scale.y = cell
    m.scale.z = 0.01                  # 厚度（Z），可根据需要调整
    m.lifetime = rospy.Duration(0)    # 0 = 不自动过期

    # 把所有点塞进 points / colors
    points = []
    colors = []
    z_plane = 0.0  # 把热力图铺在 z=0 平面；可改成地面高度

    for iy in range(0, ny, stride):
        for ix in range(0, nx, stride):
            v = SUM[iy, ix]
            if v <= 0.1:   # 跳过数值为0或负数的点
                continue
                
            # 2) 归一化（按分位裁剪）
            v_clip = v
            if v_clip < p_lo: 
                # 比 lo 还小的，若你希望可见就别强制设0；我们给个极小正值
                # 也可以直接 continue（完全不画），按你需求选其一
                norm_v = eps_floor
            else:
                v_clip = min(v_clip, p_hi)
                norm_v = (v_clip - p_lo) / max(1e-12, (p_hi - p_lo))
                # γ校正
                norm_v = np.power(norm_v, gamma)
                # 3) 设下限，避免 0 → 黑
                norm_v = max(norm_v, eps_floor)

            r, g, b, _ = cmap(norm_v)
            x = XX[iy, ix]; y = YY[iy, ix]

            points.append(Point(x=x, y=y, z=z_plane))
            colors.append(ColorRGBA(float(r), float(g), float(b), 1.0))

    m.points = points
    m.colors = colors

    while not rospy.is_shutdown():
        m.header.stamp = rospy.Time.now()
        pub.publish(m)
        rate.sleep()

if __name__ == "__main__":
    publish_to_rviz()

