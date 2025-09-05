# Object Marker Publisher

这是一个ROS包，用于在RViz中显示地图中的物品信息，如桌子、椅子、书架等，使用3D标记和文本标签进行可视化。

## 功能特点

- 在RViz中显示3D物品标记
- 为每个物品添加文本标签和描述
- 可配置的物品位置、大小、颜色
- 支持多种物品类型（桌子、椅子、书架等）
- 基于YAML配置文件，易于修改

## 安装

1. 将此包放入您的catkin工作空间的src目录
2. 编译工作空间：
```bash
cd /home/shrenqi/semantic_slam_ws
catkin_make
source devel/setup.bash
```

‵‵‵bash
pip install 'volcengine-python-sdk[ark]'
pip install opencv-python numpy requests
‵‵‵

## 使用方法

### 基本启动

启动物品标记发布器：
```bash
roslaunch object_marker_publisher object_markers.launch
```

### 带RViz启动

同时启动物品标记发布器和RViz：
```bash
roslaunch object_marker_publisher object_markers.launch launch_rviz:=true
```

### 手动启动RViz

如果您想使用自定义的RViz配置：
```bash
# 启动物品标记发布器
roslaunch object_marker_publisher object_markers.launch

# 在另一个终端启动RViz
rosrun rviz rviz -d $(rospack find object_marker_publisher)/config/objects.rviz
```

## 配置

### 物品配置文件

编辑 `config/objects.yaml` 文件来添加、修改或删除物品：

```yaml
objects:
  - name: "桌子1"
    type: "table"
    position:
      x: 2.0
      y: 1.0
      z: 0.4
    size:
      x: 2.0
      y: 1.0
      z: 0.05
    color:
      r: 0.8
      g: 0.4
      b: 0.2
      a: 0.8
    description: "会议桌"
```

### 参数说明

- `name`: 物品名称
- `type`: 物品类型（用于分类）
- `position`: 物品在地图中的位置 (x, y, z)
- `size`: 物品的尺寸 (长, 宽, 高)
- `color`: 物品的颜色 (r, g, b, a) - 值范围0-1
- `description`: 物品描述

### Launch文件参数

您可以通过修改launch文件或命令行参数来自定义行为：

- `config_file`: 配置文件名 (默认: objects.yaml)
- `frame_id`: 坐标系 (默认: map)
- `marker_scale`: 标记缩放比例 (默认: 1.0)
- `publish_rate`: 发布频率 (默认: 1.0 Hz)

示例：
```bash
roslaunch object_marker_publisher object_markers.launch marker_scale:=1.5 publish_rate:=2.0
```

## RViz显示

启动后，在RViz中您会看到：

1. **3D物品标记**: 彩色的立方体表示各种物品
2. **文本标签**: 显示物品名称和描述
3. **箭头指示**: 黄色箭头指向物品位置

### RViz设置

确保在RViz中添加以下显示项：

1. MarkerArray - 订阅话题: `/object_markers`
2. Map - 订阅话题: `/map` (如果有地图的话)

## 话题

- `/object_markers` (visualization_msgs/MarkerArray): 发布所有物品标记

## 自定义扩展

### 添加新的物品类型

1. 编辑 `config/objects.yaml` 添加新物品
2. 可以在Python脚本中添加特殊的标记类型处理逻辑

### 动态更新

当前版本从配置文件读取静态物品信息。您可以扩展代码以支持：
- 动态添加/删除物品
- 从数据库读取物品信息
- 通过服务调用更新物品状态

## 故障排除

### 标记不显示
1. 检查话题是否正在发布：`rostopic echo /object_markers`
2. 确保RViz中的MarkerArray显示项已启用
3. 检查坐标系是否正确

### 配置文件错误
1. 验证YAML文件格式是否正确
2. 检查文件路径是否存在
3. 查看节点日志：`rosnode info object_marker_publisher`

## 依赖

- roscpp
- rospy
- std_msgs
- geometry_msgs
- visualization_msgs
- tf
- PyYAML (Python包)

## 许可证

MIT License
