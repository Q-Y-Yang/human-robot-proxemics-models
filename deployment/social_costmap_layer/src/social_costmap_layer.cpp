#include "social_costmap_layer.h"
#include <pluginlib/class_list_macros.hpp>
#include <tf2/utils.h>
#include <algorithm>
#include <cstdint>

namespace social_layer {

SocialCostmapLayer::SocialCostmapLayer()
: grid_received_(false),
  tf_listener_(tf_buffer_) {}

void SocialCostmapLayer::onInitialize() {
  ROS_INFO(" onInitialize()");
  ros::NodeHandle nh("~/" + name_);
  std::string topic = nh.param<std::string>("topic", "social_costmap");
  sub_ = nh.subscribe(topic, 1, &SocialCostmapLayer::gridCb, this);
  global_frame_ = layered_costmap_->getGlobalFrameID();   // e.g., "odom"
  current_ = true;
  enabled_ = true;
}

void SocialCostmapLayer::matchSize() {
  // No-op
}

static void normalizeOccupancyTo0_254(const std::vector<int8_t>& in,
                                             std::vector<uint8_t>& out) {
  out.resize(in.size());

  // Find min/max among known cells (>=0)
  int8_t min_val = 0;
  int8_t max_val = 100;
  bool have_known = false;
  for (auto v : in) {
    if (v >= 0) {
      have_known = true;
      if (v < min_val) min_val = v;
      if (v > max_val) max_val = v;
    }
  }

  if (!have_known) {
    // All unknown → keep 255 everywhere
    std::fill(out.begin(), out.end(), static_cast<uint8_t>(255));
    return;
  }

  if (max_val > min_val) {
    // const double denom = static_cast<double>(max_val - min_val);
    for (size_t i = 0; i < in.size(); ++i) {
      int v = static_cast<int>(in[i]);
      if (v < 0) {
        out[i] = 255;  // unknown
      } else {
        double val = static_cast<double>(v - min_val) / 100;
        uint8_t scaled = static_cast<uint8_t>(val * 254.0 + 0.5); // round to 0..252
        out[i] = scaled;
      }
    }
  } else {
    // All known cells equal → map to 0, unknowns remain 255
    for (size_t i = 0; i < in.size(); ++i) {
      out[i] = (in[i] < 0) ? 255 : 0;
    }
  }
}

void SocialCostmapLayer::gridCb(const nav_msgs::OccupancyGridConstPtr& msg) {
  std::lock_guard<std::mutex> lock(mutex_);
  info_ = msg->info;
  grid_origin_x_ = info_.origin.position.x;
  grid_origin_y_ = info_.origin.position.y;
  data_ = msg->data;  // copy
  grid_frame_ = msg->header.frame_id;
  normalizeOccupancyTo0_254(data_, data_scaled_);
  grid_received_ = true;

  // ROS_INFO("social_layer: received grid in frame [%s], size = %u x %u",
  //         grid_frame_.c_str(), info_.width, info_.height);
           
  // Try to fetch transforms now
  have_tf_ = getTransforms();
}

bool SocialCostmapLayer::getTransforms() {
  if (grid_frame_.empty() || global_frame_.empty()) return false;
  try {
    // transform from grid -> global (odom)
    tf_grid_to_global_ = tf_buffer_.lookupTransform(
        global_frame_, grid_frame_, ros::Time(0), ros::Duration(0.1));
    // and global -> grid
    tf_global_to_grid_ = tf_buffer_.lookupTransform(
        grid_frame_, global_frame_, ros::Time(0), ros::Duration(0.1));
        
    // ROS_INFO("social_layer: lookupTransform success [%s -> %s] and [%s -> %s]",
    //         grid_frame_.c_str(), global_frame_.c_str(),
    //         global_frame_.c_str(), grid_frame_.c_str());

    return true;
  } catch (const tf2::TransformException& ex) {
    ROS_WARN_THROTTLE(2.0, "social_layer: TF lookup failed (%s -> %s): %s",
                      grid_frame_.c_str(), global_frame_.c_str(), ex.what());
    return false;
  }
}

// helper: transform a point (x,y) from global_frame into grid_frame
inline void transformGlobalToGrid(const geometry_msgs::TransformStamped& T,
                                  double xg, double yg, double& xq, double& yq) {
  const auto& t = T.transform.translation;
  const auto& q = T.transform.rotation;
  tf2::Quaternion qq(q.x, q.y, q.z, q.w);
  const double yaw = tf2::getYaw(qq);
  const double c = std::cos(yaw), s = std::sin(yaw);

  // Correct forward application: p_grid = R * p_global + t
  xq =  c * xg - s * yg + t.x;
  yq =  s * xg + c * yg + t.y;
}

void SocialCostmapLayer::updateBounds(double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
                                      double* min_x, double* min_y, double* max_x, double* max_y) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!grid_received_) return;
  if (!have_tf_) have_tf_ = getTransforms();
  if (!have_tf_) return;

  // Grid corners in grid_frame
  const double w = info_.width  * info_.resolution;
  const double h = info_.height * info_.resolution;

  // Corner points relative to grid origin, then rotate+translate to global
  auto toGlobal = [&](double xg_local, double yg_local, double& X, double& Y){
    // First rotate by origin yaw in grid frame (info_.origin.orientation)
    tf2::Quaternion q0;
    tf2::fromMsg(info_.origin.orientation, q0);
    double yaw0 = tf2::getYaw(q0);
    double c0 = cos(yaw0), s0 = sin(yaw0);
    double xr = grid_origin_x_ + c0*xg_local - s0*yg_local;
    double yr = grid_origin_y_ + s0*xg_local + c0*yg_local;

    // Then apply grid->global transform
    const auto& T = tf_grid_to_global_.transform;
    double cg = cos(tf2::getYaw(tf2::Quaternion(
                  T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w)));
    double sg = sin(tf2::getYaw(tf2::Quaternion(
                  T.rotation.x, T.rotation.y, T.rotation.z, T.rotation.w)));
    double Xr = cg*xr - sg*yr + T.translation.x;
    double Yr = sg*xr + cg*yr + T.translation.y;
    X = Xr; Y = Yr;
  };

  double Xs[4], Ys[4];
  toGlobal(0, 0, Xs[0], Ys[0]);
  toGlobal(w, 0, Xs[1], Ys[1]);
  toGlobal(0, h, Xs[2], Ys[2]);
  toGlobal(w, h, Xs[3], Ys[3]);

  for (int k=0; k<4; ++k) {
    *min_x = std::min(*min_x, Xs[k]);
    *min_y = std::min(*min_y, Ys[k]);
    *max_x = std::max(*max_x, Xs[k]);
    *max_y = std::max(*max_y, Ys[k]);
  }
}

void SocialCostmapLayer::updateCosts(costmap_2d::Costmap2D& master_grid,
                                     int min_i, int min_j, int max_i, int max_j) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!grid_received_) return;
  if (!have_tf_) have_tf_ = getTransforms();
  if (!have_tf_) return;

  for (int j = min_j; j < max_j; ++j) {
    for (int i = min_i; i < max_i; ++i) {
      double wx, wy;                      // in global_frame (odom)
      master_grid.mapToWorld(i, j, wx, wy);

      // transform this world point into the grid frame
      double xg, yg;                      // in grid_frame (world)
      // We looked up global->grid directly above; use that:
      transformGlobalToGrid(tf_global_to_grid_, wx, wy, xg, yg);

      // account for grid origin pose (translation + yaw) inside grid_frame
      tf2::Quaternion q0;
      tf2::fromMsg(info_.origin.orientation, q0);
      double yaw0 = tf2::getYaw(q0);
      double c0 = cos(-yaw0), s0 = sin(-yaw0); // inverse rotate by origin yaw
      double xr =  c0*(xg - grid_origin_x_) - s0*(yg - grid_origin_y_);
      double yr =  s0*(xg - grid_origin_x_) + c0*(yg - grid_origin_y_);

      int gi = static_cast<int>(floor(xr / info_.resolution));
      int gj = static_cast<int>(floor(yr / info_.resolution));
      if (gi < 0 || gj < 0 || gi >= static_cast<int>(info_.width) || gj >= static_cast<int>(info_.height))
        continue;

      int idx = gj * info_.width + gi;
      // int8_t occ = data_[idx];
      // if (occ < 0) continue;  // unknown
      
        // choose where "no-go" starts in the 0..100 domain
      // constexpr int lethal_threshold = 95;   // e.g., 95..100 become lethal

      // if (occ >= lethal_threshold) {
      //  master_grid.setCost(i, j, costmap_2d::LETHAL_OBSTACLE);  // 254
      // } else {
      // unsigned char cost = static_cast<unsigned char>(occ);  // 0..100 → 0..252
      // unsigned char old  = master_grid.getCost(i, j);
      // if (cost > old) {master_grid.setCost(i, j, cost);
               // ROS_INFO("Applied social cost %d at cell (%d,%d), old=%d new=%d",
               //     cost, i, j, old,
               //     master_grid.getCost(i, j));
               //     }
      uint8_t occ_scaled = data_scaled_[idx];   // 0..252 or 255
      if (occ_scaled == 255) continue;          // unknown → skip
      // unsigned char old = master_grid.getCost(i, j);
      // if (occ_scaled > old) {
      master_grid.setCost(i, j, occ_scaled);
      // }
    }
  }
  ROS_INFO_THROTTLE(5.0, "Social layer successfully merged into local costmap.");
}

void SocialCostmapLayer::reset() {
  grid_received_ = false;
  have_tf_ = false;
  data_scaled_.clear();
}

} // namespace social_layer

PLUGINLIB_EXPORT_CLASS(social_layer::SocialCostmapLayer, costmap_2d::Layer)

