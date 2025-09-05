#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import yaml
import os
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import Header, ColorRGBA
import tf2_ros
import tf2_geometry_msgs


class ObjectMarkerPublisher:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('object_marker_publisher', anonymous=True)
        
        # 创建发布器
        self.marker_pub = rospy.Publisher('object_markers', MarkerArray, queue_size=10)
        
        # 获取参数
        self.config_file = rospy.get_param('~config_file', 'objects.yaml')
        self.frame_id = rospy.get_param('~frame_id', 'map')
        self.marker_scale = rospy.get_param('~marker_scale', 1.0)
        self.publish_rate = rospy.get_param('~publish_rate', 1.0)
        
        # 物品数据
        self.objects = []
        
        # 加载物品配置
        self.load_objects_config()
        
        # 创建定时器
        self.timer = rospy.Timer(rospy.Duration(1.0/self.publish_rate), self.publish_markers)
        
        # 添加语义标记订阅器
        self.semantic_sub = rospy.Subscriber('semantic_markers', MarkerArray, self.semantic_callback)
        self.semantic_markers = []
        
        rospy.loginfo("Object Marker Publisher initialized")
        rospy.loginfo(f"Loaded {len(self.objects)} objects")

    def load_objects_config(self):
        """加载物品配置文件"""
        try:
            # 查找配置文件
            package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(package_path, 'config', self.config_file)
            
            if not os.path.exists(config_path):
                rospy.logwarn(f"Config file not found: {config_path}")
                rospy.logwarn("Using default objects")
                self.create_default_objects()
                return
            
            with open(config_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
                if 'objects' in data:
                    self.objects = data['objects']
                    rospy.loginfo(f"Loaded objects from {config_path}")
                else:
                    rospy.logwarn("No 'objects' key found in config file")
                    self.create_default_objects()
                    
        except Exception as e:
            rospy.logerr(f"Error loading config file: {str(e)}")
            self.create_default_objects()

    def create_default_objects(self):
        """创建默认的物品示例"""
        self.objects = [
            {
                'name': '桌子1',
                'type': 'table',
                'position': {'x': 2.0, 'y': 1.0, 'z': 0.5},
                'size': {'x': 1.2, 'y': 0.8, 'z': 0.05},
                'color': {'r': 0.8, 'g': 0.4, 'b': 0.2, 'a': 0.8},
                'description': '会议室桌子'
            },
            {
                'name': '椅子1',
                'type': 'chair',
                'position': {'x': 1.5, 'y': 0.5, 'z': 0.4},
                'size': {'x': 0.5, 'y': 0.5, 'z': 0.8},
                'color': {'r': 0.2, 'g': 0.2, 'b': 0.8, 'a': 0.8},
                'description': '办公椅'
            },
            {
                'name': '书架1',
                'type': 'shelf',
                'position': {'x': -1.0, 'y': 2.0, 'z': 1.0},
                'size': {'x': 0.3, 'y': 1.5, 'z': 2.0},
                'color': {'r': 0.6, 'g': 0.3, 'b': 0.1, 'a': 0.8},
                'description': '图书架'
            }
        ]

    def create_object_marker(self, obj, marker_id):
        """创建物品的3D标记"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        
        marker.ns = "objects"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # 设置位置
        marker.pose.position.x = obj['position']['x']
        marker.pose.position.y = obj['position']['y']
        marker.pose.position.z = obj['position']['z']
        marker.pose.orientation.w = 1.0
        
        # 设置大小
        marker.scale.x = obj['size']['x'] * self.marker_scale
        marker.scale.y = obj['size']['y'] * self.marker_scale
        marker.scale.z = obj['size']['z'] * self.marker_scale
        
        # 设置颜色
        marker.color.r = obj['color']['r']
        marker.color.g = obj['color']['g']
        marker.color.b = obj['color']['b']
        marker.color.a = obj['color']['a']
        
        marker.lifetime = rospy.Duration(0)  # 永久显示
        
        return marker

    def create_text_marker(self, obj, marker_id):
        """创建物品的文本标记"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        
        marker.ns = "object_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # 设置位置（在物品上方）
        marker.pose.position.x = obj['position']['x']
        marker.pose.position.y = obj['position']['y']
        marker.pose.position.z = obj['position']['z'] + obj['size']['z'] + 0.3
        marker.pose.orientation.w = 1.0
        
        # 设置文本内容
        marker.text = f"{obj['name']}\n{obj.get('description', '')}"
        
        # 设置大小
        marker.scale.z = 0.2 * self.marker_scale
        
        # 设置颜色
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        marker.lifetime = rospy.Duration(0)  # 永久显示
        
        return marker

    def create_arrow_marker(self, obj, marker_id):
        """创建指向物品的箭头标记"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        
        marker.ns = "object_arrows"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 设置箭头起点和终点
        start_point = Point()
        start_point.x = obj['position']['x']
        start_point.y = obj['position']['y']
        start_point.z = obj['position']['z'] + obj['size']['z'] + 0.5
        
        end_point = Point()
        end_point.x = obj['position']['x']
        end_point.y = obj['position']['y']
        end_point.z = obj['position']['z'] + obj['size']['z'] + 0.1
        
        marker.points = [start_point, end_point]
        
        # 设置大小
        marker.scale.x = 0.05 * self.marker_scale  # 轴径
        marker.scale.y = 0.1 * self.marker_scale   # 箭头直径
        marker.scale.z = 0.1 * self.marker_scale   # 箭头长度
        
        # 设置颜色（黄色）
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.lifetime = rospy.Duration(0)  # 永久显示
        
        return marker

    def semantic_callback(self, marker_array_msg):
        """接收语义检测结果"""
        self.semantic_markers = marker_array_msg.markers
        rospy.logdebug(f"Received {len(self.semantic_markers)} semantic markers")

    def publish_markers(self, event):
        """发布所有标记（包括静态物品和动态语义检测结果）"""
        marker_array = MarkerArray()
        
        # 添加静态物品标记
        for i, obj in enumerate(self.objects):
            # 添加物品3D标记
            object_marker = self.create_object_marker(obj, i)
            marker_array.markers.append(object_marker)
            
            # 添加文本标记
            text_marker = self.create_text_marker(obj, i + 1000)
            marker_array.markers.append(text_marker)
            
            # 添加箭头标记
            arrow_marker = self.create_arrow_marker(obj, i + 2000)
            marker_array.markers.append(arrow_marker)
        
        # 添加语义检测标记
        for semantic_marker in self.semantic_markers:
            # 调整命名空间以避免冲突
            semantic_marker.ns = f"semantic_{semantic_marker.ns}"
            semantic_marker.id += 5000  # 偏移ID避免冲突
            marker_array.markers.append(semantic_marker)
        
        # 发布标记数组
        self.marker_pub.publish(marker_array)

    def run(self):
        """运行节点"""
        rospy.loginfo("Object Marker Publisher running...")
        rospy.spin()


if __name__ == '__main__':
    try:
        publisher = ObjectMarkerPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Object Marker Publisher shutting down")
