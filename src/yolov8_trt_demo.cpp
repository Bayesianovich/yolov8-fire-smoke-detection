#include "yolov8_trt_demo.h"
// TensorRT 日志记录器类
//=================================================================

/**
 * @brief TensorRT 自定义日志记录器
 * 
 * TensorRT 要求用户提供一个日志记录器来处理运行时信息
 * 这个类继承自 ILogger 接口，用于控制日志输出级别 
 *
 * 
 * 日志级别说明：
 * - kINTERNAL_ERROR: 内部错误，严重问题
 * - kERROR: 一般错误，需要关注
 * - kWARNING: 警告信息，可能的问题
 * - kINFO: 信息性消息，正常运行状态
 * - kVERBOSE: 详细调试信息
*/

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity != Severity::kINFO)
            std::cout << "[TENSORRT] " << msg << std::endl;
    }
} gLogger; // 全局日志记录器实例


// 析构函数：资源清理和内存释放
// =================================================================
/**
 * @brief YOLOv8TRTDetector 析构函数
 * 
 * 负责清理所有在初始化过程中分配的资源，确保没有内存泄漏
 * 清理顺序很重要：先同步CUDA操作，再释放TensorRT资源，最后释放GPU内存
 * 
 * 清理步骤：
 * 1. 同步CUDA流：等待所有异步操作完成
 * 2. 销毁CUDA流：释放流资源
 * 3. 销毁TensorRT执行上下文：释放推理状态
 * 4. 销毁TensorRT引擎：释放模型资源
 * 5. 销毁TensorRT运行时：释放运行时资源
 * 6. 释放GPU内存缓冲区：释放输入输出缓冲区
 */
YOLOv8TRTDetector::~YOLOv8TRTDetector()
{
    // 步骤1：同步cuda流，确保所有GPU操作完成
    // gpu操作是异步的，需要等待完成才能安全释放资源
    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    // 步骤2：释放TensorRT执行上下文
    if (context) {
        context->destroy();
        context = nullptr;
    }
    // 步骤3：释放TensorRT引擎
    if (engine) {
        engine->destroy();
        engine = nullptr;
    }
    // 步骤4：释放TensorRT运行时
    if (runtime) {
        runtime->destroy();
        runtime = nullptr;
    }
    // 步骤5：释放GPU内存缓冲区
    if (buffers[0]) {
        cudaFree(buffers[0]); // 释放输入缓冲区  
        buffers[0] = nullptr;
    }
    if (buffers[1]) {
        cudaFree(buffers[1]); // 释放输出缓冲区
        buffers[1] = nullptr;
    }
    std::cout << "YOLOv8TRTDetector resources released." << std::endl;

}

// 检测器初始化方法：加载模型和配置参数
// =================================================================
/**
    * @brief 初始化YOLOv8TRTDetector检测器
    * 
    * 1. 加载TensorRT引擎文件
    * 2. 创建TensorRT运行时和执行上下文
    * 3. 获取模型输入输出维度信息
    * 4. 分配GPU内存缓冲区
    * 5. 创建CUDA流用于异步数据传输和推理
    */
void YOLOv8TRTDetector::initConfig(const std::string& engine_file, float conf_threshold, float iou_threshold){
    // 步骤1. 读取TensorRT引擎文件
    std::ifstream file(engine_file,std::ios::binary);
    if (!file.good()){
        std::cerr << "Error opening engine file." << std::endl;
        exit(-1);
    }   

    // 获取文件大小
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    // 分配内存并读取文件内容
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    std::cout << "Engine file read successfully, size: " << size << " bytes." << std::endl;

    // 步骤2. 创建TensorRT运行时，引擎 和执行上下文

    // createInferRuntime 负责引擎的反序列化和管理
    runtime = nvinfer1::createInferRuntime(gLogger);
    assert(runtime != nullptr);
    // deserializeCudaEngine 负责将序列化的引擎数据转换为可执行的引擎对象
    engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    // createExecutionContext 负责创建执行上下文，用于管理推理状态
    context = engine->createExecutionContext();
    assert(context != nullptr);

    delete[] trtModelStream; // 释放模型文件内存

    std::cout << "TensorRT engine and context created successfully." << std::endl;
    
    // 步骤3. 获取模型输入输出维度信息
    // =================================================================
    
    int nbBindings = engine->getNbBindings();
    std::cout << "Model has " << nbBindings << " bindings" << std::endl;
    
    int inputIndex = -1;
    int outputIndex = -1;
    
    // 遍历所有绑定，找到输入和输出
    for (int i = 0; i < nbBindings; ++i) {
        const char* bindingName = engine->getBindingName(i);
        bool isInput = engine->bindingIsInput(i);
        std::cout << "Binding " << i << ": " << bindingName << " (input: " << isInput << ")" << std::endl;
        
        if (isInput) {
            inputIndex = i;
        } else {
            outputIndex = i;
        }
    }

    if (inputIndex == -1 || outputIndex == -1) {
        std::cerr << "Error: Unable to find input/output binding indices." << std::endl;
        exit(-1);
    }

    // 步骤4：获取输入维度信息 (NCHW格式)

    auto  inputDims = engine->getBindingDimensions(inputIndex);
    inputH = inputDims.d[2]; // 宽度
    inputW = inputDims.d[3]; // 高度

    std::cout << "Model Input Size: " << inputDims.d[0] << "x" << inputDims.d[1] << "x" << inputH << "x" << inputW << std::endl;

    // 步骤5：获取输出维度信息
    // YOLOv8 输出格式：1x6x8400
    auto outputDims = engine->getBindingDimensions(outputIndex);
    output_feat = outputDims.d[1]; // 特征维度
    output_detbox = outputDims.d[2]; // 目标检测框数量

    std::cout << "Model Output Size: " << outputDims.d[0] << "x" << output_feat << "x" << output_detbox << std::endl;


    // 步骤6：分配GPU内存缓冲区
    std::cout << "input/output bindings: " << engine->getNbBindings() << std::endl;

    // 分配输入缓冲区：存储预处理的图像数据
    size_t inputSize = inputH * inputW * 3 * sizeof(float); // 输入大小 (NCHW)
    cudaMalloc(&buffers[inputIndex], inputSize);

    // 分配输出缓冲区：存储模型推理结果
    size_t outputSize = output_feat * output_detbox * sizeof(float); // 输出大小
    cudaMalloc(&buffers[outputIndex], outputSize);

    std::cout << "Allocated GPU memory for input and output buffers." <<"input bytes: "<< inputSize << " output bytes: " << outputSize << std::endl;

    // 步骤7：初始化CPU端输出缓冲区和CUDA流
    prob.resize(output_feat * output_detbox); // CPU端输出缓冲区
    cudaStreamCreate(&stream); // 创建CUDA流

    //保存配置参数
    this->conf_threshold = conf_threshold;
    this->iou_threshold = iou_threshold;

    std::cout << "YOLOv8TRTDetector initialized successfully." << std::endl;
}

// 核心检测方法：执行完整的目标检测流水线
    // ============================================================================
    /**
    * @brief 执行 YOLOv8 目标检测
    * 
    * 这是检测器的核心方法，实现完整的检测流水线：
    * 1. 图像预处理：尺寸调整、填充、归一化、格式转换
    * 2. 数据传输：CPU到GPU的异步内存拷贝
    * 3. 模型推理：TensorRT引擎执行前向传播
    * 4. 结果解析：从原始输出中提取边界框和类别
    * 5. NMS处理：去除重叠的检测框
    * 6. 坐标转换：将结果映射回原始图像坐标系
    * 7. 可视化绘制：在图像上绘制检测结果和性能信息
    */
void YOLOv8TRTDetector::detect(cv::Mat& frame, std::vector<DetectResult>& results){
    //开始计时，计算FPS
    int64 start = cv::getTickCount();

    // 步骤1：图像预处理 - 尺寸调整和格式转换
    int original_h = frame.rows; // 原始图像高度
    int original_w = frame.cols; // 原始图像宽度

    // yolov8预处理: 等比例缩放 + 填充
    // 1.创建正方形画布，尺寸为原图长边
    // 2.将原图放在画布左上方，其余区域填充为黑色
    // 3.缩放后的图像尺寸不超过640x640
    int max_side = std::max(original_h, original_w); // 取长边为正方形画布边长
    cv::Mat canvas = cv::Mat::zeros(max_side, max_side, CV_8UC3); // 创建正方形黑色画布

    // 将原图复制到画布左上角
    cv::Rect roi(0, 0, original_w, original_h); // 定义ROI区域
    frame.copyTo(canvas(roi)); // 复制整张原图到画布左上角

    // 计算缩放比例,用于后续坐标转换
    float x_scale = canvas.cols / static_cast<float>(inputW);
    float y_scale = canvas.rows / static_cast<float>(inputH);

    // 使用OpenCV DNN模块进行图像预处理
    // blobFromImage 会执行以下操作：
    // 1. 调整图像大小到模型输入尺寸 (640x640)
    // 2. 归一化像素值到 [0,1]
    // 3. 转换图像格式从 HWC 到 CHW
    // 4. 减去均值（这里为0,所以不变）
    // 5. BGR 转 RGB

    cv::Mat tensor = cv::dnn::blobFromImage(canvas, 1.0/255.0, cv::Size(inputW, inputH), cv::Scalar(0,0,0), true, false);
    std::cout << "Preprocessed image" << "original size: " << original_w << "x" << original_h 
              << " canvas size: " << canvas.cols << "x" << canvas.rows 
              << " tensor size: " << tensor.size[0] << "x" << tensor.size[1] << "x" << tensor.size[2] << "x" << tensor.size[3] << std::endl;

    // 步骤2：数据传输 - 将预处理图像数据从CPU异步拷贝到GPU
    cudaMemcpyAsync(buffers[0], // trarget GPU输入缓冲区
                    tensor.ptr<float>(),  // source CPU输入数据
                    inputH * inputW * 3 * sizeof(float), // 拷贝大小
                    cudaMemcpyHostToDevice, // 传输方向cpu->gpu
                    stream); // 使用的CUDA流

    // 步骤3：模型推理 - 执行TensorRT引擎的前向传播
    context->enqueueV2(buffers, stream, nullptr);

    // 步骤4：结果解析 - 将推理结果从GPU异步拷贝回CPU
    cudaMemcpyAsync(prob.data(), // target CPU输出缓冲区
                    buffers[1], // source GPU输出数据
                    output_feat * output_detbox * sizeof(float), // 拷贝大小
                    cudaMemcpyDeviceToHost, // 传输方向gpu->cpu
                    stream); // 使用的CUDA流

    // 同步CUDA流，等待所有CUDA操作完成
    cudaStreamSynchronize(stream);
    std::cout << "Inference completed." << std::endl;

    // 步骤5：解析检测结果
    // prob数据格式：1x6x8400
    // 6维度包含4个边界框坐标 + 1个目标置信度 + 1个类别ID
    //8400个检测框：来自3个不同尺度的特征图

    std::vector<int> class_ids; // 存储类别ID
    std::vector<float> confidences; // 存储置信度
    std::vector<cv::Rect> boxes; // 存储边界框

    // 将输出数据重新组织为OpenCV Mat格式，便于处理
    cv::Mat detMat(output_feat, output_detbox, CV_32F, (float*)prob.data()); // 6x8400
    cv::Mat detMat_t = detMat.t(); // 转置为 8400x6

    std::cout << "start parsing results..." << std::endl;

    // 遍历每个检测框
    for (int i = 0; i < detMat_t.rows; ++i) {
        // 提取类别概率部分 - 火灾烟雾模型只有2个类别
        cv::Mat scores = detMat_t.row(i).colRange(4, output_feat); // 从第4列开始提取类别概率

        // 获取最大类别概率及其索引
        cv::Point classIdPoint;
        double max_class_socre;
        cv::minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint); // 返回最大值和索引

        // 置信度筛选：只保留高于阈值的检测框
        if (max_class_socre > conf_threshold) {
            // 提取边界框坐标
            float cx = detMat_t.at<float>(i, 0); // 中心x
            float cy = detMat_t.at<float>(i, 1); // 中心y
            float w = detMat_t.at<float>(i, 2);  // 宽度
            float h = detMat_t.at<float>(i, 3);  // 高度

            // 转换为左上角坐标和宽高格式
            int left = static_cast<int>(cx - w / 2);
            int top = static_cast<int>(cy - h / 2);
            int width = static_cast<int>(w);
            int height = static_cast<int>(h);

            // 映射回原始图像坐标系，不能超出图像边界
            left = std::max(0, static_cast<int>(left * x_scale));
            top = std::max(0, static_cast<int>(top * y_scale));
            width = std::min(static_cast<int>(width * x_scale), original_w - left);
            height = std::min(static_cast<int>(height * y_scale), original_h - top);

            // 创建边界框矩形
            cv::Rect box(left, top, width, height);

            boxes.push_back(box); // 保存边界框
            class_ids.push_back(classIdPoint.x); // 保存类别ID
            confidences.push_back(static_cast<float>(max_class_socre)); // 保存置信度
        }

    }
    std::cout << "Total boxes before NMS: " << boxes.size() << std::endl;
    // 步骤6：NMS处理 - 使用非极大值抑制去除重叠的检测框
    // NMS算法用于同一目标被多次检测的问题
    // 工作原理:
    // 1. 按置信度降序排列所以检测框
    // 2. 选择置信度最高的框，计算其与其他框的IoU
    // 3. 删除所有与该框IoU超过阈值的框
    // 4. 重复直到所有框处理完毕
    std::vector<int> nms_indices; // NMS后保留的框索引
    cv::dnn::NMSBoxes(
        boxes, // 输入边界框
        confidences,// 输入置信度
        conf_threshold,// 置信度阈值:低于此值的框会被丢弃
        iou_threshold,// IoU阈值：超过此值的框会被抑制
        nms_indices // 输出: 保留框的索引
    );

    std::cout << "Total boxes after NMS: " << nms_indices.size() << std::endl;

    // 步骤7: 结果整理 - 将NMS后的结果保存到输出结构体  可视化留给任务文件处理 
    // 遍历NMS保留的框，整理结果
    for (size_t i = 0; i < nms_indices.size(); ++i) {
        int idx = nms_indices[i]; // 获取保留框的原始索引
        
        // 创建检测结果结构体
        DetectResult result;

        result.class_id = class_ids[idx]; // 类别ID
        result.conf = confidences[idx]; // 置信度
        result.box = boxes[idx]; // 边界框

        // 将检测结果添加到输出向量(检测结果绘制矩形框留给任务文件处理)
        results.push_back(result); // 保存结果
       
    }
    // 计算并打印FPS
    int64 end = cv::getTickCount();
    float fps = cv::getTickFrequency() / (end - start);
    std::cout << "Detection completed. FPS: " << fps << std::endl;
    // 在图像上绘制FPS信息
    cv::putText(frame, cv::format("FPS: %.2f", fps), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
}