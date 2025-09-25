#!/usr/bin/env python3
import rospy
import numpy as np
from math import atan2
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from people_msgs.msg import People
from tf.transformations import quaternion_from_euler
from scipy.spatial.transform import Rotation as R
from mvem.stats import multivariate_skewnorm as mvsn

# ---- base model (person local frame) ----
BASE_MU = np.array([ 0.0170915, -0.3079768])
BASE_SHAPE = np.array([[ 0.0791303, -0.01143162],
                       [-0.01143162, 0.29992035]])
BASE_LMBDA = np.array([-0.01057365, 2.15069084])

def transform_params(position_xy, quat_xyzw, mu, shape, lmbda):
    # quat order for scipy is [x,y,z,w]
    R3 = R.from_quat(quat_xyzw).as_matrix()
    R2 = R3[np.ix_([0,1],[0,1])]               # project to XY plane
    mu_t      = R2 @ mu + position_xy
    shape_t   = R2 @ shape @ R2.T
    lmbda_t   = R2 @ lmbda
    return mu_t, shape_t, lmbda_t
    

class SocialCostPublisher:
    def __init__(self):
        self.frame_id   = rospy.get_param('~frame_id', 'world')
        self.resolution = float(rospy.get_param('~resolution', 0.05))       #in meters/cell
        self.size_x     = int(rospy.get_param('~size_x', 72))      #number of cells
        self.size_y     = int(rospy.get_param('~size_y', 72))
        origin_x_to_world = 1.5
        origin_y_to_world = -2.5#-2.5
        self.origin_x = 1.5
        self.origin_y = -2.5#-2.5

        self.scale      = float(rospy.get_param('~scale', 1000))  # PDF→0..100

        self.pub = rospy.Publisher('social_costmap', OccupancyGrid, queue_size=1, latch=True)
        self.sub = rospy.Subscriber('people', People, self.people_cb, queue_size=1)

        x_offsets = self.resolution * (np.arange(self.size_x) - (self.size_x/2 - 0.5))
        y_offsets = self.resolution * (np.arange(self.size_y) - (self.size_y/2 - 0.5))

        xs = self.origin_x + x_offsets
        ys = self.origin_y + y_offsets
        # xs = self.origin_x + self.resolution*(np.arange(self.size_x) + 0.5)
        # ys = self.origin_y + self.resolution*(np.arange(self.size_y) + 0.5)
        self.X, self.Y = np.meshgrid(xs, ys)              # (Ny, Nx)
        self.grid_pts  = np.dstack((self.X, self.Y))      # (Ny, Nx, 2)

        rospy.loginfo("social_costmap node (mvem) ready.")

    def people_cb(self, msg: People):
        params = []
        for p in msg.people:
            vx, vy = p.velocity.x, p.velocity.y
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                yaw = 0.0
            else:
                yaw = atan2(vy, vx)
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
            mu, shape, lmbda = transform_params(
                np.array([p.position.x, p.position.y]),
                np.array([qx, qy, qz, qw]),
                BASE_MU, BASE_SHAPE, BASE_LMBDA
            )
            params.append((mu, shape, lmbda))

        if params:
            # mvem.pdf supports broadcasting over a grid: (Ny, Nx, 2)
            Z = np.zeros((self.size_y, self.size_x), dtype=np.float64)
            for mu, shape, lmbda in params:
                Z += mvsn.pdf(self.grid_pts, mu, shape, lmbda)
        else:
            Z = np.zeros((self.size_y, self.size_x), dtype=np.float64)

        grid = np.clip(Z * self.scale, 0, 100).astype(np.uint8)
        # print((Z * self.scale).min(), (Z * self.scale).max())
        # Normalize to [0, 100]
        # Z_min, Z_max = Z.min(), Z.max()
        
        #if Z_max > Z_min:
        #    Z_norm = 100 * (Z - Z_min) / (Z_max - Z_min)
        #else:
        #    Z_norm = np.zeros_like(Z)  # all values same → map to 0
        
        #grid = Z_norm.astype(np.uint8)

        out = OccupancyGrid()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.frame_id

        info = MapMetaData()
        info.resolution = self.resolution
        info.width  = self.size_x
        info.height = self.size_y
        info.origin = Pose()
        info.origin.position.x = self.origin_x - 0.5 * self.size_x * self.resolution
        info.origin.position.y = self.origin_y - 0.5 * self.size_y * self.resolution
        info.origin.orientation.w = 1.0
        out.info = info

        out.data = grid.flatten(order='C').tolist()
        self.pub.publish(out)

if __name__ == '__main__':
    rospy.init_node('social_costmap_node')
    SocialCostPublisher()
    rospy.spin()
