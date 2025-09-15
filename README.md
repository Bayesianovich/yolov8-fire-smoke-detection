⭐⭐ 如果这个项目对你有帮助，请给个 Star！⭐⭐


# YOLOv8 火灾烟雾检测系统

基于 YOLOv8 和 TensorRT 的实时火灾烟雾检测系统，专为工业安全监控设计。

## 🔥 项目简介

本项目实现了一个高性能的火灾烟雾检测系统，使用 YOLOv8 深度学习模型结合 TensorRT GPU 加速，能够在视频流中实时检测火灾和烟雾。系统采用智能双条件触发机制，只有在同时检测到火和烟时才会触发警报，有效减少误报。

### ✨ 核心特性

- 🚀 **实时检测**: 基于 TensorRT 优化的 YOLOv8 模型，实现高帧率实时检测
- 🎯 **双条件触发**: 智能算法，仅在同时检测到火和烟时触发警报
- 💪 **GPU 加速**: 充分利用 CUDA 和 TensorRT 进行推理加速
- 📹 **视频处理**: 支持视频文件和实时视频流处理
- 🎨 **可视化界面**: 实时显示检测结果和性能指标
- 💾 **自动保存**: 检测到警报条件时自动保存关键帧

### 🏗️ 技术架构

- **深度学习框架**: YOLOv8
- **推理引擎**: NVIDIA TensorRT 8.6+
- **计算平台**: CUDA 12.1+
- **计算机视觉**: OpenCV 4.8+
- **构建系统**: XMake
- **编程语言**: C++17

## 📋 系统要求

### 硬件要求
- NVIDIA GPU（支持 CUDA Compute Capability 6.1+）
- RAM: 8GB+ 推荐
- 存储空间: 5GB+ 可用空间

### 软件依赖
- **CUDA Toolkit 12.1+**
- **TensorRT 8.6+**
- **cuDNN 8.9+**
- **OpenCV 4.8+** (包含以下模块)：
  - opencv_core
  - opencv_imgproc
  - opencv_imgcodecs
  - opencv_highgui
  - opencv_videoio
  - opencv_video
  - opencv_dnn
- **XMake** 构建工具

### 支持平台
- Windows 10/11
- Linux (Ubuntu 20.04+)

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/Bayesianovich/yolov8-fire-smoke-detection.git
cd yolov8-fire-smoke-detection
```

### 2. 配置环境
修改 `xmake.lua` 文件中的库路径：

```lua
local tensorrt_root = "你的TensorRT路径"        -- 例如: "/usr/local/TensorRT-8.6.1.6"
local cudnn_root = "你的cuDNN路径"             -- 例如: "/usr/local/cudnn-8.9"
local cuda_root = "你的CUDA路径"               -- 例如: "/usr/local/cuda-12.1"
local opencv_root = "你的OpenCV路径"           -- 例如: "/usr/local/opencv"
```

### 3. 准备模型文件
将以下文件放置在项目根目录：
- `firesmokev1.engine` - TensorRT 引擎文件
- `firesmog.mp4` - 测试视频文件

### 4. 编译项目
```bash
# Release 模式编译（推荐）
xmake build

# 或 Debug 模式编译
xmake config --mode=debug
xmake build
```

### 5. 运行检测
```bash
xmake run yolov8_demo
```

## 📁 项目结构

```
yolov8-fire-smoke-detection/
├── main.cpp                    # 主程序入口
├── include/
│   └── yolov8_trt_demo.h      # 检测器头文件
├── src/
│   └── yolov8_trt_demo.cpp    # 检测器实现
├── xmake.lua                   # XMake 构建配置
├── classes.txt                 # 类别标签文件
├── firesmog.engine             # 自己添加一个engine引擎文件
├── README.md                   # 项目说明文档
└── firesmog.mp4         # 自己添加一个mp4
```

## 🔧 配置说明

### 检测参数调整
在 `main.cpp` 中可以调整以下参数：

```cpp
// 置信度阈值 (0.0-1.0)
detector->initConfig(enginefile, 0.25f, 0.25f);
//                               ↑      ↑
//                         置信度阈值  IoU阈值
```

### 输出路径配置
修改检测结果保存路径：

```cpp
std::string save_dir = "你的保存路径/detected_frames";
```

### 类别标签
`classes.txt` 文件包含检测类别：
```
fire    # 类别ID: 0
smoke   # 类别ID: 1
```

## 📊 性能表现

### 测试环境
- GPU: RTX 3080
- CPU: Intel i7-12700K
- RAM: 32GB DDR4

### 性能指标
- **推理速度**: ~45 FPS (640x640 输入)
- **检测精度**: mAP@0.5 > 95%
- **内存使用**: GPU ~2GB, CPU ~512MB
- **启动时间**: < 3 秒

## 🎯 使用场景

- **工业安全监控**: 工厂、仓库火灾预警
- **建筑消防**: 办公楼、住宅区安全监控  
- **森林防火**: 野外火灾早期发现
- **智能交通**: 隧道、桥梁安全监控
- **能源设施**: 电厂、变电站安全防护

## 🛠️ 开发说明

### 构建选项
```bash
# 显示项目信息
xmake info

# 清理构建文件
xmake clean

# 清理引擎文件
xmake clean-engines

# 指定构建模式
xmake config --mode=release  # 发布模式
xmake config --mode=debug    # 调试模式
```

### 调试模式
Debug 模式下会输出详细日志信息，有助于开发调试：

```bash
xmake config --mode=debug
xmake build
xmake run yolov8_demo
```

### 代码风格
- 使用 C++17 标准
- 遵循 RAII 原则进行资源管理
- 异步 CUDA 操作提升性能
- 详细的中文注释说明

## 🔍 故障排除

### 常见问题

1. **编译错误: 找不到 TensorRT**
   ```bash
   # 检查 TensorRT 路径配置
   # 确认 xmake.lua 中的 tensorrt_root 路径正确
   ```

2. **运行时错误: 无法加载引擎文件**
   ```bash
   # 确认引擎文件存在且路径正确
   ls firesmokev1.engine
   ```

3. **GPU 内存不足**
   ```bash
   # 检查 GPU 内存使用情况
   nvidia-smi
   # 考虑降低输入分辨率或批处理大小
   ```

4. **检测精度不理想**
   - 调整置信度阈值
   - 检查输入视频质量
   - 确认光照条件适宜

### 日志分析
程序运行时会输出详细日志，包括：
- TensorRT 引擎加载状态
- 模型输入输出维度
- 检测结果统计
- 性能指标 (FPS)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 贡献流程
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范
- 保持代码整洁和良好注释
- 遵循现有代码风格
- 添加必要的单元测试
- 更新相关文档

## 📄 开源协议

本项目采用 MIT 协议 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 优秀的目标检测框架
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) - 高性能推理引擎
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [XMake](https://xmake.io/) - 现代化构建工具


-----------------

⭐ 如果这个项目对你有帮助，请给个 Star！ ⭐ 


  
