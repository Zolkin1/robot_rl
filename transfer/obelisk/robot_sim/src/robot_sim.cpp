#include "rclcpp/rclcpp.hpp"

#include "obelisk_mujoco_sim_robot.h"
#include "obelisk_ros_utils.h"

int main(int argc, char* argv[]) {
    obelisk::utils::SpinObelisk<obelisk::ObeliskMujocoRobot<obelisk_control_msgs::msg::PDFeedForward>,
                                rclcpp::executors::MultiThreadedExecutor>(argc, argv, "mujoco_sim");
}