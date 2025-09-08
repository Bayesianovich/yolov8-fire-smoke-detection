#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <algorithm> 


#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "NvInfer.h"


using namespace nvinfer1;
using namespace cv;

/**
 * @brief 目标检测结果结构体
 * 
 * 这个结构体封装了单个目标检测的完整信息，包括：
 * - 目标类别：识别出的对象类型（如人、车、狗等）
 * - 置信度：模型对检测结果的确信程度（0-1之间）
 * - 边界框：目标在图像中的精确位置和大小
 */
struct DetectResult {
    int class_id;
    float conf;
    cv::Rect box;
};

/**
 * @brief YOLOv8 TensorRT 目标检测器类
 * 
 * 这个类实现了完整的 YOLOv8 目标检测流水线：
 * 1. 模型初始化:加载YOLOv8模型引擎文件
 * 2. 图像预处理:调整尺寸、归一化、转换为浮点数
 * 3. GPU推理执行:执行TensorRT推理
 * 4. 后处理检测结果:解析输出，非极大值抑制(NMS)过滤，坐标转换
 * 5. 结果可视化：绘制边界框和标签
 *
 * 技术特点：
 * 1. 零拷贝内存管理:使用CUDA流进行异步数据传输
 * 2. 动态批处理:根据输入大小自动调整批处理大小
 * 3. 内存池复用:避免频繁的内存分配和释放
 * 4.多线程安全:支持并发推理(需要多个context)

*/


 
class YOLOv8TRTDetector {
    public:
     // ========================================================================
    // 公共接口方法
    // ========================================================================
    
    /**
     * @brief 初始化检测器配置 
     *
     * 这个方法完成检测器的完整初始化过程：：
     * 1. 加载TensorRT引擎文件
     * 2. 创建推理运行时和执行上下文
     * 3. 分配GPU内存缓存区
     * 4. 获取模型输入输出维度信息
     * 5.创建CUDA流用于异步操作
     * 
     *
     */
        void initConfig(const std::string& engine_file, const float conf_threshold, const float iou_threshold);

        /**
        * 1. 图像预处理:尺寸调整、填充、归一化
        * 2. 数据传输:CPU到GPU的异步内存拷贝
        * 3. 模型推理:TensorRT 引擎执行前向传播
        * 4. 结果解析：从原始输出中提取边界框和类别
        * 5. NMS处理: 非极大值抑制过滤重叠框
        * 6. 坐标转换：将结果转换为原始图像坐标系
        * 7. 可视化绘制：在图像上绘制检测结果 
             
     */
        void detect(cv::Mat& frame, std::vector<DetectResult>& results);

        ~YOLOv8TRTDetector();



    private:
     // ========================================================================
    // 私有成员变量
    // ========================================================================
    
    float conf_threshold = 0.25f;
    float iou_threshold = 0.25f; 

    int inputH = 640;
    int inputW = 640;

    int output_feat;
    int output_detbox;

    nvinfer1::IRuntime* runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};

    // 内存缓冲区
    void* buffers[2] = {nullptr, nullptr}; // 输入和输出缓冲区
    std::vector<float> prob; // 输出数据临时存储：CPU 端缓冲区,用于存储从 GPU 拷贝回来的推理结果

    cudaStream_t stream; // 用于异步内存拷贝和核函数执行
};
        