#pragma once
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>
#include <mutex>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace social_layer {

class SocialCostmapLayer : public costmap_2d::Layer {
public:
  SocialCostmapLayer();

  void onInitialize() override;
  void matchSize() override;
  void updateBounds(double robot_x, double robot_y, double robot_yaw,
                    double* min_x, double* min_y, double* max_x, double* max_y) override;
  void updateCosts(costmap_2d::Costmap2D& master_grid,
                   int min_i, int min_j, int max_i, int max_j) override;
  bool isDiscretized() { return true; }
  void deactivate() override {}
  void activate() override {}
  void reset() override;

private:
  void gridCb(const nav_msgs::OccupancyGridConstPtr& msg);
  bool getTransforms(); // refresh TFs

  ros::Subscriber sub_;
  nav_msgs::MapMetaData info_;
  std::vector<int8_t> data_;
  std::string grid_frame_;          // frame of incoming grid (msg->header.frame_id)
  std::string global_frame_;        // layered_costmap_->getGlobalFrameID()
  double grid_origin_x_{0.0}, grid_origin_y_{0.0};
  bool grid_received_;
  std::mutex mutex_;
  
  std::vector<uint8_t> data_scaled_;  // normalized copy: 0..252, 255 for unknown


  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  geometry_msgs::TransformStamped tf_grid_to_global_; // grid_frame -> global_frame (odom)
  geometry_msgs::TransformStamped tf_global_to_grid_; // global_frame -> grid_frame
  bool have_tf_{false};
};

} // namespace social_layer

