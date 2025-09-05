#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: /home/shrenqi/semantic_slam_ws/src/object_marker_publisher/scripts/semantic_detector.py

import rospy
import cv2
import numpy as np
import json
import threading
import os
import base64
from queue import Queue

import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import message_filters

# 导入火山引擎API客户端
try:
    from volcenginesdkarkruntime import Ark
    ARK_AVAILABLE = True
except ImportError:
    rospy.logwarn("volcenginesdkarkruntime not found")
    ARK_AVAILABLE = False


class SemanticDetector:
    def __init__(self):
        rospy.init_node('semantic_detector', anonymous=True)
        
        # 初始化组件
        self._init_components()
        self._init_parameters()
        self._init_api_client()
        self._init_ros_interface()
        
        # 启动处理线程
        self._start_processing_thread()
        
        rospy.loginfo("Semantic Detector initialized")

    def _init_components(self):
        """初始化基础组件"""
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 数据存储
        self.semantic_objects = {}
        self.object_id_counter = 0
        self.detection_queue = Queue(maxsize=3)
        self.last_detection_time = 0

    def _init_parameters(self):
        """初始化参数配置"""
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/rgb/image_raw')
        self.target_frame = rospy.get_param('~target_frame', 'map')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_rgb_optical_frame')
        self.detection_interval = rospy.get_param('~detection_interval', 3.0)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.6)
        
        # API配置
        self.api_key = os.getenv('ARK_API_KEY')
        self.model_id = os.getenv('ARK_MODEL_ID', 'doubao-vision-pro-32k')
        self.api_url = rospy.get_param('~api_url', 'https://ark.cn-beijing.volces.com/api/v3')

    def _init_api_client(self):
        """初始化API客户端"""
        if not ARK_AVAILABLE or not self.api_key:
            rospy.logfatal("API client not available or API key not set")
            return
            
        try:
            self.client = Ark(api_key=self.api_key, base_url=self.api_url)
            rospy.loginfo("API client initialized successfully")
        except Exception as e:
            rospy.logfatal(f"Failed to initialize API client: {e}")
            self.client = None

    def _init_ros_interface(self):
        """初始化ROS接口"""
        # 发布器
        self.marker_pub = rospy.Publisher('semantic_markers', MarkerArray, queue_size=5)
        self.debug_pub = rospy.Publisher('detection_debug', Image, queue_size=1)
        
        # 订阅器
        self.image_sub = message_filters.Subscriber(self.camera_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub], 1, 0.1)
        self.ts.registerCallback(self._image_callback)
        
        # 定时器
        self.marker_timer = rospy.Timer(rospy.Duration(1.0), self._publish_markers_timer)

    def _start_processing_thread(self):
        """启动异步处理线程"""
        self.processing_thread = threading.Thread(target=self._detection_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _image_callback(self, img_msg):
        """图像回调函数"""
        current_time = rospy.Time.now().to_sec()
        
        # 控制检测频率
        if current_time - self.last_detection_time < self.detection_interval:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            transform = self._get_camera_transform(img_msg.header.stamp)
            
            if transform and not self.detection_queue.full():
                detection_data = {
                    'image': cv_image.copy(),
                    'timestamp': current_time,
                    'transform': transform
                }
                self.detection_queue.put(detection_data)
                self.last_detection_time = current_time
                
        except (CvBridgeError, Exception) as e:
            rospy.logwarn(f"Image processing error: {e}")

    def _get_camera_transform(self, timestamp):
        """获取相机变换"""
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame, 
                self.camera_frame,
                timestamp,
                rospy.Duration(0.1)
            )
        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return None

    def _detection_worker(self):
        """异步检测工作线程"""
        while not rospy.is_shutdown():
            try:
                if not self.detection_queue.empty():
                    data = self.detection_queue.get(timeout=1.0)
                    self._process_detection(data)
            except Exception as e:
                rospy.logwarn(f"Detection worker error: {e}")

    def _process_detection(self, data):
        """处理单次检测"""
        try:
            detections = self._call_vision_api(data['image'])
            if detections:
                self._update_semantic_objects(detections, data)
                self._publish_debug_image(data['image'], detections)
                rospy.loginfo(f"Detected {len(detections)} objects")
        except Exception as e:
            rospy.logerr(f"Detection processing error: {e}")

    def _call_vision_api(self, image):
        """调用视觉API进行物体检测"""
        if not self.client:
            return []
            
        try:
            # 编码图像
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构建提示词
            prompt = self._build_detection_prompt()
            
            # API调用
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.1,
                max_tokens=1024
            )
            
            if response.choices:
                return self._parse_detection_result(response.choices[0].message.content)
            return []
            
        except Exception as e:
            rospy.logerr(f"API call error: {e}")
            return []

    def _build_detection_prompt(self):
        """构建检测提示词"""
        return """请检测图片中的物体并返回JSON格式结果：
{
  "detections": [
    {
      "class": "物体类别",
      "confidence": 0.95,
      "bbox": {"x": 100, "y": 50, "width": 200, "height": 150}
    }
  ]
}

检测类别包括：person, chair, table, bottle, cup, book, laptop, door, window等。
只返回置信度>0.5的结果，确保JSON格式正确。"""

    def _parse_detection_result(self, result_text):
        """解析检测结果"""
        try:
            # 提取JSON部分
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                result_data = json.loads(json_str)
                detections = result_data.get('detections', [])
                
                # 过滤低置信度结果
                return [d for d in detections if d.get('confidence', 0) >= self.confidence_threshold]
            return []
            
        except json.JSONDecodeError:
            rospy.logwarn("Failed to parse detection result")
            return []

    def _update_semantic_objects(self, detections, data):
        """更新语义对象 - 添加去重逻辑"""
        h, w = data['image'].shape[:2]
        
        for detection in detections:
            try:
                world_pos = self._calculate_world_position(detection, data, (h, w))
                if world_pos:
                    # 检查是否与现有物体重复
                    existing_id = self._find_duplicate_object(world_pos, detection['class'])
                    
                    if existing_id is not None:
                        # 更新现有物体
                        self.semantic_objects[existing_id].update({
                            'confidence': max(self.semantic_objects[existing_id]['confidence'], 
                                            detection.get('confidence', 0.0)),
                            'timestamp': data['timestamp'],
                            'detection_count': self.semantic_objects[existing_id].get('detection_count', 1) + 1
                        })
                    else:
                        # 添加新物体
                        self.semantic_objects[self.object_id_counter] = {
                            'category': detection.get('class', 'unknown'),
                            'confidence': detection.get('confidence', 0.0),
                            'position': world_pos,
                            'timestamp': data['timestamp'],
                            'detection_count': 1,
                            'bbox_size': detection.get('bbox', {}),
                            'estimated_depth': self._estimate_depth(detection.get('class', 'unknown'), 
                                                                  detection.get('bbox', {}))
                        }
                        self.object_id_counter += 1
            except Exception as e:
                rospy.logwarn(f"Object update error: {e}")

    def _find_duplicate_object(self, new_position, category):
        """查找重复物体"""
        # 根据物体类别设置不同的距离阈值
        distance_thresholds = {
            'person': 1.0,
            'chair': 0.8,
            'table': 1.2,
            'bottle': 0.3,
            'cup': 0.3,
            'book': 0.4,
            'laptop': 0.5,
            'door': 1.5,
            'window': 1.0
        }
        
        threshold = distance_thresholds.get(category, 0.6)
        
        for obj_id, obj_data in self.semantic_objects.items():
            if obj_data['category'] == category:
                existing_pos = obj_data['position']
                distance = np.sqrt(
                    (new_position['x'] - existing_pos['x'])**2 +
                    (new_position['y'] - existing_pos['y'])**2 +
                    (new_position['z'] - existing_pos['z'])**2
                )
                
                if distance < threshold:
                    return obj_id
        
        return None

    def _calculate_world_position(self, detection, data, image_size):
        """计算世界坐标位置"""
        h, w = image_size
        bbox = detection.get('bbox', {})
        
        # 计算中心点
        center_x = bbox.get('x', 0) + bbox.get('width', 0) / 2
        center_y = bbox.get('y', 0) + bbox.get('height', 0) / 2
        
        # 估算深度
        depth = self._estimate_depth(detection['class'], bbox)
        
        # 相机内参（简化）
        fx = fy = 500.0
        cx, cy = w/2, h/2
        
        # 像素到相机坐标
        cam_x = (center_x - cx) * depth / fx
        cam_y = (center_y - cy) * depth / fy
        cam_z = depth
        
        # 转换到世界坐标
        return self._transform_to_world_coordinates(cam_x, cam_y, cam_z, data['transform'])

    def _estimate_depth(self, category, bbox):
        """估算物体深度 - 改进版本"""
        # 物体典型尺寸（米）- 更精确的数据
        typical_sizes = {
            'person': {'height': 1.7, 'width': 0.5},
            'chair': {'height': 0.9, 'width': 0.6}, 
            'table': {'height': 0.75, 'width': 1.2},
            'bottle': {'height': 0.25, 'width': 0.08},
            'cup': {'height': 0.1, 'width': 0.08},
            'book': {'height': 0.25, 'width': 0.2},
            'laptop': {'height': 0.02, 'width': 0.35},
            'door': {'height': 2.0, 'width': 0.8},
            'window': {'height': 1.2, 'width': 1.0}
        }
        
        # 默认大小
        default_size = {'height': 0.5, 'width': 0.3}
        size_info = typical_sizes.get(category, default_size)
        
        # 获取边界框尺寸
        bbox_height = max(bbox.get('height', 100), 1)
        bbox_width = max(bbox.get('width', 100), 1)
        
        # 假设相机焦距 (像素)
        focal_length = 500.0
        
        # 基于高度和宽度计算深度，取平均值
        depth_from_height = (size_info['height'] * focal_length) / bbox_height
        depth_from_width = (size_info['width'] * focal_length) / bbox_width
        
        # 使用加权平均，高度权重更高
        depth = (depth_from_height * 0.7 + depth_from_width * 0.3)
        
        # 根据物体类别调整深度范围
        if category in ['person', 'door']:
            depth = np.clip(depth, 1.0, 10.0)
        elif category in ['table', 'chair']:
            depth = np.clip(depth, 0.8, 5.0)
        elif category in ['bottle', 'cup', 'book']:
            depth = np.clip(depth, 0.3, 3.0)
        else:
            depth = np.clip(depth, 0.5, 8.0)
            
        return depth

    def _transform_to_world_coordinates(self, cam_x, cam_y, cam_z, transform):
        """转换到世界坐标系"""
        try:
            point_camera = PointStamped()
            point_camera.header.frame_id = self.camera_frame
            point_camera.header.stamp = rospy.Time.now()
            point_camera.point.x = cam_x
            point_camera.point.y = cam_y
            point_camera.point.z = cam_z
            
            point_world = tf2_geometry_msgs.do_transform_point(point_camera, transform)
            
            return {
                'x': point_world.point.x,
                'y': point_world.point.y,
                'z': point_world.point.z
            }
        except Exception as e:
            rospy.logwarn(f"Coordinate transformation error: {e}")
            return None

    def _publish_debug_image(self, image, detections):
        """发布调试图像"""
        try:
            debug_image = image.copy()
            for detection in detections:
                self._draw_detection(debug_image, detection)
            
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_msg.header.stamp = rospy.Time.now()
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            rospy.logwarn(f"Debug image publish error: {e}")

    def _draw_detection(self, image, detection):
        """在图像上绘制检测结果"""
        bbox = detection.get('bbox', {})
        x, y = int(bbox.get('x', 0)), int(bbox.get('y', 0))
        w, h = int(bbox.get('width', 0)), int(bbox.get('height', 0))
        
        # 绘制边界框和标签
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{detection.get('class', 'unknown')}: {detection.get('confidence', 0):.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def _publish_markers_timer(self, event):
        """发布语义标记定时器"""
        current_time = rospy.Time.now()
        self._cleanup_old_objects(current_time.to_sec())
        
        if self.semantic_objects:
            marker_array = self._create_marker_array(current_time)
            self.marker_pub.publish(marker_array)

    def _create_marker_array(self, timestamp):
        """创建标记数组"""
        marker_array = MarkerArray()
        
        for obj_id, obj_data in self.semantic_objects.items():
            # 物体标记
            marker = self._create_object_marker(obj_id, obj_data, timestamp)
            marker_array.markers.append(marker)
            
            # 文本标记
            text_marker = self._create_text_marker(obj_id, obj_data, timestamp)
            marker_array.markers.append(text_marker)
        
        return marker_array

    def _create_object_marker(self, obj_id, obj_data, timestamp):
        """创建物体标记 - 动态大小"""
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = timestamp
        marker.ns = "semantic_objects"
        marker.id = obj_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 位置
        pos = obj_data['position']
        marker.pose.position.x = pos['x']
        marker.pose.position.y = pos['y']
        marker.pose.position.z = pos['z']
        marker.pose.orientation.w = 1.0
        
        # 根据物体类别和深度动态调整大小
        category = obj_data['category']
        confidence = obj_data['confidence']
        depth = obj_data.get('estimated_depth', 2.0)
        
        # 基础大小映射
        base_sizes = {
            'person': 0.3,
            'chair': 0.25,
            'table': 0.35,
            'bottle': 0.1,
            'cup': 0.08,
            'book': 0.12,
            'laptop': 0.15,
            'door': 0.4,
            'window': 0.3
        }
        
        base_size = base_sizes.get(category, 0.2)
        
        # 考虑置信度和检测次数的大小调整
        detection_count = obj_data.get('detection_count', 1)
        size_factor = min(1.0 + (detection_count - 1) * 0.1, 1.5)  # 最大1.5倍
        confidence_factor = 0.5 + confidence * 0.5  # 0.5-1.0范围
        
        final_size = base_size * size_factor * confidence_factor
        
        marker.scale.x = marker.scale.y = marker.scale.z = final_size
        
        # 颜色 - 根据置信度调整透明度
        color = self._get_category_color(category)
        alpha = 0.6 + confidence * 0.4  # 置信度越高越不透明
        marker.color = ColorRGBA(*color, alpha)
        
        # 生存时间根据检测次数调整
        lifetime = min(30.0 + detection_count * 5.0, 60.0)
        marker.lifetime = rospy.Duration(lifetime)
        
        return marker

    def _create_text_marker(self, obj_id, obj_data, timestamp):
        """创建文本标记"""
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = timestamp
        marker.ns = "semantic_labels"
        marker.id = obj_id + 10000
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # 位置（物体上方）
        pos = obj_data['position']
        marker.pose.position.x = pos['x']
        marker.pose.position.y = pos['y']
        marker.pose.position.z = pos['z'] + 0.2
        marker.pose.orientation.w = 1.0
        
        # 文本内容
        marker.text = f"{obj_data['category']}\n{obj_data['confidence']:.2f}"
        marker.scale.z = 0.12
        marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        marker.lifetime = rospy.Duration(30.0)
        
        return marker

    def _get_category_color(self, category):
        """获取类别颜色"""
        colors = {
            'person': (1.0, 0.0, 0.0), 'chair': (0.0, 1.0, 0.0),
            'table': (0.0, 0.0, 1.0), 'bottle': (1.0, 1.0, 0.0),
            'cup': (1.0, 0.0, 1.0), 'book': (0.0, 1.0, 1.0),
            'laptop': (1.0, 0.5, 0.0), 'door': (0.5, 0.0, 1.0)
        }
        return colors.get(category, (0.5, 0.5, 0.5))

    def _cleanup_old_objects(self, current_time):
        """清理过期对象 - 改进版本"""
        # 根据检测次数设置不同的超时时间
        base_timeout = 30.0
        
        expired_ids = []
        for obj_id, obj_data in self.semantic_objects.items():
            detection_count = obj_data.get('detection_count', 1)
            # 检测次数越多，保留时间越长
            timeout = base_timeout + min(detection_count * 5.0, 30.0)
            
            if current_time - obj_data['timestamp'] > timeout:
                expired_ids.append(obj_id)
        
        for obj_id in expired_ids:
            del self.semantic_objects[obj_id]
            
        if expired_ids:
            rospy.loginfo(f"Cleaned up {len(expired_ids)} expired objects")

    def run(self):
        """运行节点"""
        rospy.loginfo("Semantic Detector running...")
        rospy.spin()


if __name__ == '__main__':
    try:
        detector = SemanticDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Semantic Detector node terminated.")