#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float32.h>
#include <fis_vo/VOFISData.h> 

class DataConstructNode
{
public:
    DataConstructNode()
    {
        // 初始化订阅者
        keypoint_sub_ = nh_.subscribe("/orb_keypoint_num_sync", 1, &DataConstructNode::keypointCallback, this);
        repro_error_sub_ = nh_.subscribe("/orb_repro_error_sync", 1, &DataConstructNode::reproErrorCallback, this);

        // 初始化发布者
        vofis_data_pub_ = nh_.advertise<fis_vo::VOFISData>("/VOFIS_msg", 1); // 替换为你的发布话题
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber keypoint_sub_;
    ros::Subscriber repro_error_sub_;
    ros::Publisher vofis_data_pub_;

    int32_t keypoint_num_;
    float repro_error_;
    bool keypoint_received_ = false; // 新增标记是否接收到数据
    bool repro_error_received_ = false; // 新增标记是否接收到数据

    void keypointCallback(const std_msgs::Int32::ConstPtr &msg)
    {
        keypoint_num_ = msg->data;
        keypoint_received_ = true; // 标记已接收到
        if (repro_error_received_) // 如果两个数据都已接收，则发布消息
        {
            publishVOFISData();
        }
    }

    void reproErrorCallback(const std_msgs::Float32::ConstPtr &msg)
    {
        repro_error_ = msg->data;
        repro_error_received_ = true; // 标记已接收到
        if (keypoint_received_) // 如果两个数据都已接收，则发布消息
        {
            publishVOFISData();
        }
    }

    void publishVOFISData()
    {
        fis_vo::VOFISData vofis_data_msg;
        vofis_data_msg.keypoint_num = keypoint_num_;
        vofis_data_msg.repro_error = repro_error_;

        vofis_data_pub_.publish(vofis_data_msg);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "orbdata_construct_node");
    DataConstructNode node;
    ros::spin();
    return 0;
}
