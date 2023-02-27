#include "rclcpp/rclcpp.hpp"
#include <gz/transport/Node.hh>
#include <gz/msgs/twist.pb.h>
#include <gz/msgs/pointcloud_packed.pb.h>
#include "std_msgs/msg/string.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>



rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_publisher;


sensor_msgs::msg::PointCloud2 convertPointCloudPackedToPointCloud2(const gz::msgs::PointCloudPacked& msg)
{
    sensor_msgs::msg::PointCloud2 pc2;
    rclcpp::Time now = rclcpp::Clock().now();
    pc2.header.stamp = now;
    pc2.header.frame_id = "lidar_frame";

    pc2.height = 1;
    pc2.width = msg.data().size() / msg.point_step();
    pc2.fields.resize(3);

    // Populate the fields
    pc2.fields[0].name = "x";
    pc2.fields[0].offset = 0;
    pc2.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    pc2.fields[0].count = 1;
    pc2.fields[1].name = "y";
    pc2.fields[1].offset = 4;
    pc2.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    pc2.fields[1].count = 1;
    pc2.fields[2].name = "z";
    pc2.fields[2].offset = 8;
    pc2.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    pc2.fields[2].count = 1;

    pc2.is_bigendian = false;
    pc2.point_step = msg.point_step();
    pc2.row_step = pc2.point_step * pc2.width;

    pc2.data.resize(msg.data().size());
    std::memcpy(pc2.data.data(), msg.data().data(), msg.data().size());

    pc2.is_dense = true;

    return pc2;
}

void lidar_cb(const gz::msgs::PointCloudPacked &msg)
{
    sensor_msgs::msg::PointCloud2 pcl2_msg = convertPointCloudPackedToPointCloud2(msg);
    lidar_publisher->publish(pcl2_msg);
}



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("publisher_node");
    lidar_publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>("lidar", 10);

    gz::transport::Node nodegz;
    std::string topic_sub = "/lidar1/points";   // subscribe to this topic
    if (!nodegz.Subscribe(topic_sub, lidar_cb))
    {
      std::cerr << "Error subscribing to topic [" << topic_sub << "]" << std::endl;
      return -1;
    }

    rclcpp::spin(node);
    return 0;
}
