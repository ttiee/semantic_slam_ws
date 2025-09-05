#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import message_filters
from image_geometry import PinholeCameraModel

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
        
        # 相机模型
        self.camera_model = PinholeCameraModel()
        self.camera_info_received = False
        
        # 数据存储
        self.semantic_objects = {}
        self.object_id_counter = 0
        self.detection_queue = Queue(maxsize=3)
        self.last_detection_time = 0

    def _init_parameters(self):
        """初始化参数配置"""
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/rgb/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/rgb/camera_info')
        self.target_frame = rospy.get_param('~target_frame', 'map')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_rgb_optical_frame')
        self.detection_interval = rospy.get_param('~detection_interval', 3.0)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.6)
        
        # 平面高度配置
        self.ground_plane_height = rospy.get_param('~ground_plane_height', 0.0)
        self.table_plane_height = rospy.get_param('~table_plane_height', 0.75)
        self.shelf_plane_height = rospy.get_param('~shelf_plane_height', 1.2)
        
        # 物体类别到平面的映射
        self.category_plane_mapping = {
            'person': 'ground',
            'chair': 'ground', 
            'table': 'ground',
            'door': 'ground',
            'window': 'wall',
            'bottle': 'table',
            'cup': 'table',
            'book': 'table',
            'laptop': 'table'
        }
        
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
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_callback)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub], 1, 0.1)
        self.ts.registerCallback(self._image_callback)
        
        # 定时器
        self.marker_timer = rospy.Timer(rospy.Duration(1.0), self._publish_markers_timer)

    def _camera_info_callback(self, camera_info_msg):
        """相机信息回调函数"""
        if not self.camera_info_received:
            self.camera_model.fromCameraInfo(camera_info_msg)
            self.camera_info_received = True
            rospy.loginfo("Camera calibration received and initialized")

    def _start_processing_thread(self):
        """启动异步处理线程"""
        self.processing_thread = threading.Thread(target=self._detection_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _image_callback(self, img_msg):
        """图像回调函数"""
        if not self.camera_info_received:
            rospy.logwarn_throttle(5.0, "Camera info not received yet, skipping detection")
            return
            
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
                    'transform': transform,
                    'camera_frame': img_msg.header.frame_id
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
        """使用射线与平面交点计算世界坐标位置"""
        bbox = detection.get('bbox', {})
        category = detection.get('class', 'unknown')
        
        # 计算中心点像素坐标
        center_x = bbox.get('x', 0) + bbox.get('width', 0) / 2
        center_y = bbox.get('y', 0) + bbox.get('height', 0) / 2
        
        # 计算像素射线
        ray_direction = self._get_pixel_ray(center_x, center_y)
        if ray_direction is None:
            return None
            
        # 获取相机在世界坐标系中的位置
        camera_world_pos = self._get_camera_world_position(data['transform'])
        if camera_world_pos is None:
            return None
            
        # 根据物体类别选择合适的平面进行交点计算
        intersection_point = self._calculate_ray_plane_intersection(
            camera_world_pos, ray_direction, category, data['transform']
        )
        
        return intersection_point

    def _get_pixel_ray(self, u, v):
        """计算像素点对应的单位方向向量（相机坐标系）"""
        try:
            # 使用相机模型将像素坐标转换为归一化坐标
            ray = self.camera_model.projectPixelTo3dRay((u, v))
            # 归一化射线方向
            ray_norm = np.linalg.norm(ray)
            if ray_norm > 0:
                return np.array(ray) / ray_norm
            return None
        except Exception as e:
            rospy.logwarn(f"Pixel ray calculation error: {e}")
            return None

    def _get_camera_world_position(self, transform):
        """获取相机在世界坐标系中的位置"""
        try:
            translation = transform.transform.translation
            return np.array([translation.x, translation.y, translation.z])
        except Exception as e:
            rospy.logwarn(f"Camera world position error: {e}")
            return None

    def _calculate_ray_plane_intersection(self, camera_pos, ray_direction_cam, category, transform):
        """计算射线与平面的交点"""
        try:
            # 将相机坐标系的射线方向转换到世界坐标系
            ray_direction_world = self._transform_vector_to_world(ray_direction_cam, transform)
            if ray_direction_world is None:
                return None
                
            # 根据物体类别确定目标平面
            plane_type = self.category_plane_mapping.get(category, 'ground')
            plane_height = self._get_plane_height(plane_type)
            
            # 计算射线与水平面的交点
            # 平面方程: z = plane_height
            # 射线方程: P = camera_pos + t * ray_direction_world
            # 求解: camera_pos[2] + t * ray_direction_world[2] = plane_height
            
            if abs(ray_direction_world[2]) < 1e-6:
                # 射线与平面平行
                rospy.logwarn("Ray parallel to plane, using fallback depth estimation")
                return self._fallback_depth_estimation(camera_pos, ray_direction_world, category)
                
            t = (plane_height - camera_pos[2]) / ray_direction_world[2]
            
            # 检查交点是否在相机前方
            if t <= 0:
                rospy.logwarn("Intersection behind camera, using fallback")
                return self._fallback_depth_estimation(camera_pos, ray_direction_world, category)
                
            # 计算交点坐标
            intersection = camera_pos + t * ray_direction_world
            
            # 距离合理性检查
            distance = np.linalg.norm(intersection - camera_pos)
            max_distance = self._get_max_detection_distance(category)
            
            if distance > max_distance:
                rospy.logwarn(f"Intersection too far ({distance:.2f}m), using clamped distance")
                # 使用最大距离限制
                clamped_distance = min(distance, max_distance)
                intersection = camera_pos + (clamped_distance / distance) * (intersection - camera_pos)
            
            return {
                'x': intersection[0],
                'y': intersection[1], 
                'z': intersection[2]
            }
            
        except Exception as e:
            rospy.logwarn(f"Ray-plane intersection error: {e}")
            return None

    def _transform_vector_to_world(self, vector_cam, transform):
        """将相机坐标系的向量转换到世界坐标系"""
        try:
            # 提取旋转四元数
            rotation = transform.transform.rotation
            
            # 创建旋转矩阵
            import tf.transformations as tf_trans
            rotation_matrix = tf_trans.quaternion_matrix([
                rotation.x, rotation.y, rotation.z, rotation.w
            ])[:3, :3]
            
            # 变换向量
            vector_world = rotation_matrix.dot(vector_cam)
            return vector_world
            
        except Exception as e:
            rospy.logwarn(f"Vector transformation error: {e}")
            return None

    def _get_plane_height(self, plane_type):
        """获取平面高度"""
        height_mapping = {
            'ground': self.ground_plane_height,
            'table': self.table_plane_height,
            'shelf': self.shelf_plane_height,
            'wall': self.shelf_plane_height  # 窗户等
        }
        return height_mapping.get(plane_type, self.ground_plane_height)

    def _get_max_detection_distance(self, category):
        """获取不同类别物体的最大检测距离"""
        max_distances = {
            'person': 15.0,
            'chair': 10.0,
            'table': 10.0,
            'door': 20.0,
            'window': 20.0,
            'bottle': 5.0,
            'cup': 5.0,
            'book': 5.0,
            'laptop': 8.0
        }
        return max_distances.get(category, 12.0)

    def _fallback_depth_estimation(self, camera_pos, ray_direction_world, category):
        """回退深度估算方法"""
        # 使用改进的典型深度
        typical_depths = {
            'person': 3.0,
            'chair': 2.5,
            'table': 3.0,
            'door': 4.0,
            'window': 5.0,
            'bottle': 1.5,
            'cup': 1.5,
            'book': 2.0,
            'laptop': 2.0
        }
        
        depth = typical_depths.get(category, 2.5)
        intersection = camera_pos + depth * ray_direction_world
        
        return {
            'x': intersection[0],
            'y': intersection[1],
            'z': intersection[2]
        }

    def _estimate_depth(self, category, bbox):
        """保留原有估算深度方法作为备选"""
        # 这个方法现在主要用于标记大小计算等辅助功能
        typical_depths = {
            'person': 3.0,
            'chair': 2.5, 
            'table': 3.0,
            'door': 4.0,
            'window': 5.0,
            'bottle': 1.5,
            'cup': 1.5,
            'book': 2.0,
            'laptop': 2.0
        }
        return typical_depths.get(category, 2.5)

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
        """创建物体标记 - 根据射线交点结果调整"""
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
        
        # 根据物体类别和置信度调整大小
        category = obj_data['category']
        confidence = obj_data['confidence']
        detection_count = obj_data.get('detection_count', 1)
        
        # 基础大小映射（基于射线交点方法的精度更高，可以适当减小标记）
        base_sizes = {
            'person': 0.25,
            'chair': 0.2,
            'table': 0.3,
            'bottle': 0.08,
            'cup': 0.06,
            'book': 0.1,
            'laptop': 0.12,
            'door': 0.35,
            'window': 0.25
        }
        
        base_size = base_sizes.get(category, 0.15)
        
        # 考虑置信度和检测次数的大小调整
        size_factor = min(1.0 + (detection_count - 1) * 0.1, 1.5)
        confidence_factor = 0.6 + confidence * 0.4
        
        final_size = base_size * size_factor * confidence_factor
        marker.scale.x = marker.scale.y = marker.scale.z = final_size
        
        # 颜色和透明度
        color = self._get_category_color(category)
        alpha = 0.7 + confidence * 0.3
        marker.color = ColorRGBA(*color, alpha)
        
        # 生存时间
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