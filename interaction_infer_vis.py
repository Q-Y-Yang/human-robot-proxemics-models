# kde_sum_plot.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the saved KDE model (scikit-learn KernelDensity)
# Make sure this path matches how you saved it: joblib.dump(kde, "kde_model.pkl")
MODEL_PATH = "results/kde_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find {MODEL_PATH}. "
        "Save your model with joblib.dump(kde, 'kde_model.pkl') first."
    )
kde = joblib.load(MODEL_PATH)

# Poses: (x, y, yaw in radians)
poses = [
    # (2.078499, -3.296296,  3.106),
    (1.139707, -2.781449, -2.019),
    # (1.514146, -1.949175,  3.604),
]

# Rotation helpers (ROS REP-103: x forward, y left, yaw CCW about +z)
def R_world_from_local(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def R_local_from_world(theta):
    return R_world_from_local(theta).T  # inverse/transpose

def world_to_local(Xw, pose):
    x0, y0, yaw = pose
    d = Xw - np.array([x0, y0])
    return d @ R_local_from_world(yaw)  # rotate by -yaw

# Grid to evaluate on
xs = [p[0] for p in poses]; ys = [p[1] for p in poses]
margin = 2.0
xmin, xmax = min(xs)-margin, max(xs)+margin
ymin, ymax = min(ys)-margin, max(ys)+margin
nx, ny = 300, 300
xv = np.linspace(xmin, xmax, nx); yv = np.linspace(ymin, ymax, ny)
XX, YY = np.meshgrid(xv, yv)
grid_world = np.column_stack([XX.ravel(), YY.ravel()])

# Sum of three KDE evaluations (model is learned in local frame)
SUM = np.zeros(grid_world.shape[0])
for pose in poses:
    loc = world_to_local(grid_world, pose)
    SUM += np.exp(kde.score_samples(loc))
SUM = SUM.reshape(XX.shape)

#  Plot: heatmap of summed density with pose markers, contours, and yaw arrows 
plt.figure(figsize=(7, 6))

# Heatmap
plt.imshow(
    SUM,
    origin="lower",
    extent=[xmin, xmax, ymin, ymax],
    aspect="equal",
)


plt.title("Summed KDE over Three Poses with Yaw")
plt.xlabel("x")
plt.ylabel("y")
# plt.colorbar(label="Summed density")
plt.tight_layout()
plt.show()

