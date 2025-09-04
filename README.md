# Semantic SLAM Workspace

基于TurtleBot3的语义SLAM工作空间，支持地图构建、导航和语义理解功能。

## 目录
- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [安装说明](#安装说明)
- [项目结构](#项目结构)
- [使用说明](#使用说明)
  - [SLAM建图](#slam建图)
  - [地图保存](#地图保存)
  - [导航模式](#导航模式)
  - [遥控操作](#遥控操作)
- [Launch文件说明](#launch文件说明)
- [故障排除](#故障排除)
- [维护者](#维护者)

## 项目概述

本项目是一个基于ROS的语义SLAM系统，主要功能包括：
- 使用TurtleBot3进行SLAM建图
- 基于构建地图的自主导航
- 遥控操作支持
- 仿真环境集成

## 环境要求

- **操作系统**: Ubuntu 18.04/20.04
- **ROS版本**: ROS Melodic/Noetic
- **Python版本**: 2.7/3.6+
- **依赖包**:
  - turtlebot3
  - turtlebot3_simulations
  - turtlebot3_navigation
  - turtlebot3_slam
  - gmapping
  - map_server
  - move_base

## 安装说明

### 1. 克隆项目
```bash
cd ~
git clone https://github.com/ttiee/semantic_slam_ws semantic_slam_ws
cd semantic_slam_ws
```

### 2. 安装依赖
```bash
# 安装TurtleBot3相关包
sudo apt update
sudo apt install ros-$ROS_DISTRO-turtlebot3-*
sudo apt install ros-$ROS_DISTRO-gmapping
sudo apt install ros-$ROS_DISTRO-navigation
sudo apt install ros-$ROS_DISTRO-map-server
```

### 3. 编译工作空间
```bash
cd ~/semantic_slam_ws
catkin_make
source devel/setup.bash
```

### 4. 设置环境变量
```bash
# 添加到 ~/.bashrc
echo "export TURTLEBOT3_MODEL=waffle_pi" >> ~/.bashrc
echo "source ~/semantic_slam_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 项目结构

```
semantic_slam_ws/
├── src/
│   ├── tt_turtle/                  # 主功能包
│   │   ├── launch/                 # 启动文件
│   │   │   ├── gazebo_setup.launch # Gazebo仿真设置
│   │   │   ├── slam_setup.launch   # SLAM配置
│   │   │   ├── nav_setup.launch    # 导航配置
│   │   │   └── tt_main.launch      # 主启动文件
│   │   ├── map/                    # 地图文件
│   │   │   ├── map.yaml           # 地图配置
│   │   │   ├── map.pgm            # 地图数据
│   │   │   ├── map_house.yaml     # 房屋地图配置
│   │   │   └── map_house.pgm      # 房屋地图数据
│   │   ├── package.xml            # 包配置文件
│   │   └── CMakeLists.txt         # CMake配置
│   └── CMakeLists.txt
├── build/                         # 编译文件
├── devel/                         # 开发环境
└── README.md                      # 项目说明
```

## 使用说明

### SLAM建图

#### 1. 启动Gazebo仿真环境
```bash
# 终端1：启动仿真环境
roslaunch tt_turtle gazebo_setup.launch
```

#### 2. 启动SLAM节点
```bash
# 终端2：启动SLAM
roslaunch tt_turtle slam_setup.launch slam_methods:=gmapping
```

**可选SLAM方法**:
- `gmapping` (默认)
- `cartographer`
- `hector`
- `karto`
- `frontier_exploration`

#### 3. 启动遥控节点
```bash
# 终端3：启动键盘遥控
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

**遥控按键说明**:
- `w` - 前进
- `x` - 后退  
- `a` - 左转
- `d` - 右转
- `s` - 停止
- `q`/`z` - 增加/减少线速度
- `e`/`c` - 增加/减少角速度

#### 4. 可视化建图过程
```bash
# 终端4：启动RViz可视化
rviz -d `rospack find turtlebot3_slam`/rviz/turtlebot3_slam.rviz
```

### 地图保存

建图完成后，保存地图到指定位置：

```bash
# 保存到home目录
rosrun map_server map_saver -f ~/map

# 保存到项目map目录
rosrun map_server map_saver -f ~/semantic_slam_ws/src/tt_turtle/map/my_map

# 保存到指定路径
rosrun map_server map_saver -f /path/to/your/map
```

**保存的文件**:
- `map.yaml` - 地图元数据配置文件
- `map.pgm` - 地图图像数据文件

### 导航模式

#### 1. 启动导航系统
```bash
# 方式1：使用完整启动文件（包含Gazebo+导航）
roslaunch tt_turtle tt_main.launch

# 方式2：分别启动
# 终端1：Gazebo环境
roslaunch tt_turtle gazebo_setup.launch
# 终端2：导航节点
roslaunch tt_turtle nav_setup.launch
```

#### 2. 启动RViz进行导航
```bash
# 启动导航可视化
rviz -d `rospack find turtlebot3_navigation`/rviz/turtlebot3_navigation.rviz
```

#### 3. 设置导航目标
在RViz中：
1. 点击 **"2D Pose Estimate"** 设置机器人初始位置
2. 点击 **"2D Nav Goal"** 设置目标点
3. 机器人将自动规划路径并导航到目标位置

### 遥控操作

在任何时候都可以通过键盘进行手动控制：

```bash
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

## Launch文件说明

### gazebo_setup.launch
启动TurtleBot3在Gazebo房屋环境中的仿真
```xml
<include file="$(find turtlebot3_gazebo)/launch/turtlebot3_house.launch" />
```

### slam_setup.launch  
配置SLAM算法，默认使用gmapping
```xml
<include file="$(find turtlebot3_slam)/launch/turtlebot3_slam.launch">
    <arg name="slam_methods" value="$(arg slam_methods)" />
</include>
```

### nav_setup.launch
启动导航系统，使用预构建的地图
```xml
<include file="$(find turtlebot3_navigation)/launch/turtlebot3_navigation.launch">
    <arg name="map_file" value="$(find tt_turtle)/map/map.yaml"/>
</include>
```

### tt_main.launch
主启动文件，同时启动Gazebo和导航系统
```xml
<include file="$(find tt_turtle)/launch/gazebo_setup.launch" />
<include file="$(find tt_turtle)/launch/nav_setup.launch" />
```

## 故障排除

### 1. 找不到TurtleBot3包
```bash
# 安装缺失的包
sudo apt install ros-$ROS_DISTRO-turtlebot3-*
```

### 2. 环境变量未设置
```bash
# 检查环境变量
echo $TURTLEBOT3_MODEL
export TURTLEBOT3_MODEL=waffle_pi
```

### 3. 工作空间未source
```bash
cd ~/semantic_slam_ws
source devel/setup.bash
```

### 4. Gazebo启动缓慢
```bash
# 第一次启动会下载模型，请耐心等待
# 或手动下载模型文件到 ~/.gazebo/models/
```

### 5. 地图文件路径错误
确保地图文件路径正确：
```bash
ls ~/semantic_slam_ws/src/tt_turtle/map/map.yaml
```

### 6. 导航无法启动
```bash
# 检查地图服务器是否正常运行
rosnode list | grep map_server

# 检查tf变换是否正常
rosrun tf tf_echo map base_link
```

## 常用命令总结

```bash
# 完整SLAM流程
roslaunch tt_turtle gazebo_setup.launch                    # 终端1
roslaunch tt_turtle slam_setup.launch                      # 终端2  
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch   # 终端3
rosrun map_server map_saver -f ~/map                       # 保存地图

# 完整导航流程
roslaunch tt_turtle tt_main.launch                         # 一键启动
# 或分步启动：
roslaunch tt_turtle gazebo_setup.launch                    # 终端1
roslaunch tt_turtle nav_setup.launch                       # 终端2

# 可视化
rviz -d `rospack find turtlebot3_slam`/rviz/turtlebot3_slam.rviz         # SLAM可视化
rviz -d `rospack find turtlebot3_navigation`/rviz/turtlebot3_navigation.rviz # 导航可视化
```

## 维护者

- **姓名**: shrenqi
- **邮箱**: shrenqi@hotmail.com
- **项目**: tt_turtle语义SLAM系统

## 许可证

TODO - 请根据项目需求添加合适的开源许可证

---

**注意**: 首次使用时请确保所有依赖包已正确安装，并且Gazebo模型已下载完成。如遇到问题，请参考故障排除部分或查看ROS官方文档。
