#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import json
import requests
import threading
import os
import base64
from queue import Queue
from collections import deque

import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import message_filters

# 导入火山引擎API客户端
try:
    from volcenginesdkarkruntime import Ark
except ImportError:
    rospy.logwarn("volcenginesdkarkruntime not found, falling back to requests")


class SemanticDetector:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('semantic_detector', anonymous=True)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 参数配置
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/rgb/image_raw')
        self.target_frame = rospy.get_param('~target_frame', 'map')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_rgb_optical_frame')
        self.detection_interval = rospy.get_param('~detection_interval', 2.0)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.5)
        
        # 火山引擎API配置
        self.api_key = os.getenv('ARK_API_KEY')
        self.model_id = os.getenv('ARK_MODEL_ID', rospy.get_param('~model_id', 'doubao-seed-1-6-flash-250615'))
        self.api_url = rospy.get_param('~api_url', 'https://ark.cn-beijing.volces.com/api/v3')
        
        # 初始化火山引擎客户端
        try:
            self.client = Ark(
                api_key=self.api_key,
                base_url=self.api_url
            )
            rospy.loginfo("火山引擎API客户端初始化成功")
        except Exception as e:
            rospy.logwarn(f"火山引擎API客户端初始化失败: {e}, 使用备用方案")
            self.client = None
        
        # 发布器
        self.marker_pub = rospy.Publisher('semantic_markers', MarkerArray, queue_size=10)
        self.debug_pub = rospy.Publisher('detection_debug', Image, queue_size=1)
        
        # 订阅器 - 使用message_filters同步
        self.image_sub = message_filters.Subscriber(self.camera_topic, Image)
        
        # 时间同步器
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub], 1, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.image_callback)
        
        # 检测结果存储
        self.semantic_objects = {}  # {object_id: {position, category, confidence, timestamp}}
        self.object_id_counter = 0
        
        # 异步处理队列
        self.detection_queue = Queue(maxsize=5)
        self.processing_thread = threading.Thread(target=self.detection_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 最后检测时间
        self.last_detection_time = 0
        
        # 发布定时器
        self.marker_timer = rospy.Timer(rospy.Duration(0.5), self.publish_semantic_markers)
        
        rospy.loginfo("Semantic Detector initialized")
        rospy.loginfo(f"Subscribing to: {self.camera_topic}")
        rospy.loginfo(f"Detection interval: {self.detection_interval}s")

    def image_callback(self, img_msg):
        """图像回调函数"""
        current_time = rospy.Time.now().to_sec()
        
        # 控制检测频率
        if current_time - self.last_detection_time < self.detection_interval:
            return
            
        try:
            # 转换图像格式
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            
            # 获取相机位姿
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame, 
                    self.camera_frame,
                    img_msg.header.stamp,
                    rospy.Duration(0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed: {e}")
                return
            
            # 添加到检测队列
            if not self.detection_queue.full():
                detection_data = {
                    'image': cv_image.copy(),
                    'timestamp': current_time,
                    'transform': transform,
                    'header': img_msg.header
                }
                self.detection_queue.put(detection_data)
                self.last_detection_time = current_time
                
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def detection_worker(self):
        """异步检测工作线程"""
        while not rospy.is_shutdown():
            try:
                # 从队列获取检测任务
                if not self.detection_queue.empty():
                    data = self.detection_queue.get(timeout=1.0)
                    self.process_detection(data)
            except Exception as e:
                rospy.logerr(f"Detection worker error: {e}")

    def process_detection(self, data):
        """处理单次检测"""
        try:
            image = data['image']
            transform = data['transform']
            timestamp = data['timestamp']
            
            # 调用大模型API进行检测
            detections = self.call_detection_api(image)
            
            if detections:
                # 处理检测结果
                self.process_detections(detections, transform, timestamp, image)
                rospy.loginfo(f"Detected {len(detections)} objects")
            
        except Exception as e:
            rospy.logerr(f"Detection processing error: {e}")

    def call_detection_api(self, image):
        """调用火山引擎多模态大模型进行物体检测"""
        try:
            # 编码图像为base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构建物体检测的提示词
            detection_prompt = """请仔细观察这张图片，检测并识别图片中的所有物体。请按以下JSON格式返回结果：

{
  "detections": [
    {
      "class": "物体类别名称",
      "confidence": 0.95,
      "bbox": {
        "x": 100,
        "y": 50,
        "width": 200,
        "height": 150
      },
      "description": "物体的详细描述"
    }
  ]
}

请检测常见的室内物体，如：person（人）、chair（椅子）、table（桌子）、bottle（瓶子）、cup（杯子）、book（书）、laptop（笔记本电脑）、door（门）、window（窗户）等。

注意：
1. bbox坐标系统：x,y为左上角坐标，width和height为宽度和高度
2. confidence为检测置信度（0-1之间）
3. 只返回置信度大于0.3的检测结果
4. 确保返回有效的JSON格式"""

            if self.client:
                # 使用火山引擎SDK
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": detection_prompt
                                }
                            ]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
                print(response)
                if response.choices and len(response.choices) > 0:
                    result_text = response.choices[0].message.content
                    return self.parse_detection_result(result_text)
                else:
                    rospy.logwarn("火山引擎API返回空结果")
                    return []
            else:
                # 备用方案：使用requests直接调用API
                return self.call_detection_api_fallback(image_base64, detection_prompt)
                
        except Exception as e:
            rospy.logerr(f"火山引擎API调用错误: {e}")
            return []

    def call_detection_api_fallback(self, image_base64, prompt):
        """备用API调用方案"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.model_id,
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/jpeg;base64,{image_base64}'
                                }
                            },
                            {
                                'type': 'text',
                                'text': prompt
                            }
                        ]
                    }
                ],
                'temperature': 0.1,
                'max_tokens': 2048
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=15.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    result_text = result['choices'][0]['message']['content']
                    return self.parse_detection_result(result_text)
            else:
                rospy.logwarn(f"备用API请求失败: {response.status_code}")
                
            return []
            
        except requests.exceptions.Timeout:
            rospy.logwarn("备用API请求超时")
            return []
        except Exception as e:
            rospy.logerr(f"备用API调用错误: {e}")
            return []

    def parse_detection_result(self, result_text):
        """解析大模型返回的检测结果"""
        try:
            # 尝试从文本中提取JSON部分
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                result_data = json.loads(json_str)
                
                detections = result_data.get('detections', [])
                
                # 过滤置信度低的检测结果
                filtered_detections = []
                for detection in detections:
                    confidence = detection.get('confidence', 0.0)
                    if confidence >= self.confidence_threshold:
                        filtered_detections.append(detection)
                
                rospy.loginfo(f"解析到 {len(filtered_detections)} 个有效检测结果")
                return filtered_detections
            else:
                rospy.logwarn("无法从响应中提取JSON格式")
                return []
                
        except json.JSONDecodeError as e:
            rospy.logwarn(f"JSON解析失败: {e}")
            # 尝试解析非标准格式的响应
            return self.parse_fallback_format(result_text)
        except Exception as e:
            rospy.logerr(f"结果解析错误: {e}")
            return []

    def parse_fallback_format(self, result_text):
        """解析非标准格式的检测结果"""
        try:
            detections = []
            lines = result_text.split('\n')
            
            current_detection = {}
            for line in lines:
                line = line.strip()
                if '类别' in line or 'class' in line.lower():
                    if current_detection:
                        detections.append(current_detection)
                        current_detection = {}
                    # 提取类别名称
                    class_name = line.split(':')[-1].strip()
                    current_detection['class'] = class_name
                    current_detection['confidence'] = 0.8  # 默认置信度
                    current_detection['bbox'] = {
                        'x': 100, 'y': 100, 'width': 200, 'height': 200
                    }
            
            if current_detection:
                detections.append(current_detection)
            
            return detections
            
        except Exception as e:
            rospy.logerr(f"备用格式解析失败: {e}")
            return []

    def process_detections(self, detections, transform, timestamp, image):
        """处理检测结果，转换为世界坐标"""
        h, w = image.shape[:2]
        
        for detection in detections:
            try:
                # 解析检测结果
                category = detection.get('class', 'unknown')
                confidence = detection.get('confidence', 0.0)
                bbox = detection.get('bbox', {})  # {x, y, width, height}
                
                if confidence < self.confidence_threshold:
                    continue
                
                # 计算边界框中心点（像素坐标）
                center_x = bbox.get('x', 0) + bbox.get('width', 0) / 2
                center_y = bbox.get('y', 0) + bbox.get('height', 0) / 2
                
                # 估算物体在相机坐标系中的3D位置
                # 这里使用简化的深度估计，实际应用中可以结合深度相机或双目视觉
                estimated_depth = self.estimate_object_depth(category, bbox)
                
                # 相机内参（需要根据实际相机标定结果调整）
                fx = fy = 500.0  # 焦距
                cx, cy = w/2, h/2  # 主点
                
                # 像素坐标转相机坐标
                camera_x = (center_x - cx) * estimated_depth / fx
                camera_y = (center_y - cy) * estimated_depth / fy
                camera_z = estimated_depth
                
                # 转换到世界坐标系
                world_pos = self.transform_to_world(
                    camera_x, camera_y, camera_z, transform)
                
                if world_pos:
                    # 存储语义对象
                    object_id = self.object_id_counter
                    self.object_id_counter += 1
                    
                    self.semantic_objects[object_id] = {
                        'category': category,
                        'confidence': confidence,
                        'position': world_pos,
                        'timestamp': timestamp,
                        'bbox': bbox
                    }
                    
                    # 绘制调试信息
                    self.draw_detection_debug(image, detection, (center_x, center_y))
                
            except Exception as e:
                rospy.logerr(f"Detection processing error: {e}")
        
        # 发布调试图像
        self.publish_debug_image(image)

    def estimate_object_depth(self, category, bbox):
        """根据物体类别和大小估算深度"""
        # 简化的深度估计，基于物体类别的先验知识
        typical_sizes = {
            'person': 1.7,      # 人的典型高度
            'chair': 1.0,       # 椅子典型高度  
            'table': 0.8,       # 桌子典型高度
            'bottle': 0.25,     # 瓶子典型高度
            'cup': 0.1,         # 杯子典型高度
            'book': 0.03,       # 书的典型厚度
            'laptop': 0.02,     # 笔记本典型厚度
        }
        
        typical_size = typical_sizes.get(category, 0.5)  # 默认0.5米
        
        # 根据边界框大小估算距离
        bbox_height = bbox.get('height', 100)
        
        # 简化公式：距离 = (真实高度 * 焦距) / 像素高度
        focal_length = 500.0  # 像素
        estimated_depth = (typical_size * focal_length) / max(bbox_height, 1)
        
        # 限制深度范围
        return max(0.5, min(estimated_depth, 10.0))

    def transform_to_world(self, cam_x, cam_y, cam_z, transform):
        """将相机坐标转换为世界坐标"""
        try:
            # 创建相机坐标系中的点
            point_camera = PointStamped()
            point_camera.header.frame_id = self.camera_frame
            point_camera.header.stamp = rospy.Time.now()
            point_camera.point.x = cam_x
            point_camera.point.y = cam_y
            point_camera.point.z = cam_z
            
            # 转换到世界坐标系
            point_world = tf2_geometry_msgs.do_transform_point(point_camera, transform)
            
            return {
                'x': point_world.point.x,
                'y': point_world.point.y,
                'z': point_world.point.z
            }
            
        except Exception as e:
            rospy.logerr(f"Coordinate transformation error: {e}")
            return None

    def draw_detection_debug(self, image, detection, center):
        """在图像上绘制检测结果"""
        bbox = detection.get('bbox', {})
        category = detection.get('class', 'unknown')
        confidence = detection.get('confidence', 0.0)
        
        x = int(bbox.get('x', 0))
        y = int(bbox.get('y', 0))
        w = int(bbox.get('width', 0))
        h = int(bbox.get('height', 0))
        
        # 绘制边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"{category}: {confidence:.2f}"
        cv2.putText(image, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制中心点
        cv2.circle(image, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)

    def publish_debug_image(self, image):
        """发布调试图像"""
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            debug_msg.header.stamp = rospy.Time.now()
            self.debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Debug image publish error: {e}")

    def publish_semantic_markers(self, event):
        """发布语义标记"""
        if not self.semantic_objects:
            return
            
        marker_array = MarkerArray()
        current_time = rospy.Time.now()
        
        # 清理过期的对象
        self.cleanup_old_objects(current_time.to_sec())
        
        for obj_id, obj_data in self.semantic_objects.items():
            # 创建物体标记
            marker = self.create_semantic_marker(obj_id, obj_data, current_time)
            marker_array.markers.append(marker)
            
            # 创建文本标记
            text_marker = self.create_text_marker(obj_id, obj_data, current_time)
            marker_array.markers.append(text_marker)
        
        self.marker_pub.publish(marker_array)

    def create_semantic_marker(self, obj_id, obj_data, timestamp):
        """创建语义物体标记"""
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = timestamp
        
        marker.ns = "semantic_objects"
        marker.id = obj_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 设置位置
        pos = obj_data['position']
        marker.pose.position.x = pos['x']
        marker.pose.position.y = pos['y']
        marker.pose.position.z = pos['z']
        marker.pose.orientation.w = 1.0
        
        # 设置大小（根据置信度调整）
        confidence = obj_data['confidence']
        scale = 0.1 + confidence * 0.2
        marker.scale.x = marker.scale.y = marker.scale.z = scale
        
        # 设置颜色（根据类别）
        color = self.get_category_color(obj_data['category'])
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8
        
        marker.lifetime = rospy.Duration(30.0)  # 30秒后消失
        
        return marker

    def create_text_marker(self, obj_id, obj_data, timestamp):
        """创建文本标记"""
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = timestamp
        
        marker.ns = "semantic_labels"
        marker.id = obj_id + 10000
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # 设置位置（在物体上方）
        pos = obj_data['position']
        marker.pose.position.x = pos['x']
        marker.pose.position.y = pos['y']
        marker.pose.position.z = pos['z'] + 0.3
        marker.pose.orientation.w = 1.0
        
        # 设置文本
        marker.text = f"{obj_data['category']}\n{obj_data['confidence']:.2f}"
        
        # 设置大小
        marker.scale.z = 0.15
        
        # 设置颜色
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        marker.lifetime = rospy.Duration(30.0)
        
        return marker

    def get_category_color(self, category):
        """根据类别返回颜色"""
        colors = {
            'person': (1.0, 0.0, 0.0),      # 红色
            'chair': (0.0, 1.0, 0.0),       # 绿色
            'table': (0.0, 0.0, 1.0),       # 蓝色
            'bottle': (1.0, 1.0, 0.0),      # 黄色
            'cup': (1.0, 0.0, 1.0),         # 品红
            'book': (0.0, 1.0, 1.0),        # 青色
            'laptop': (1.0, 0.5, 0.0),      # 橙色
            'door': (0.5, 0.0, 1.0),        # 紫色
            'window': (1.0, 0.7, 0.3),      # 橙黄色
        }
        return colors.get(category, (0.5, 0.5, 0.5))  # 默认灰色

    def cleanup_old_objects(self, current_time):
        """清理过期的语义对象"""
        timeout = 60.0  # 60秒超时
        expired_ids = []
        
        for obj_id, obj_data in self.semantic_objects.items():
            if current_time - obj_data['timestamp'] > timeout:
                expired_ids.append(obj_id)
        
        for obj_id in expired_ids:
            del self.semantic_objects[obj_id]

    def run(self):
        """运行节点"""
        rospy.loginfo("Semantic Detector running...")
        rospy.spin()


if __name__ == '__main__':
    try:
        detector = SemanticDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Semantic Detector shutting down")
