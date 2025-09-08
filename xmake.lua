-- ============================================================================
-- XMake 配置文件：YOLOv8 TensorRT 目标检测项目
-- ============================================================================
-- 这个配置文件用于构建基于 TensorRT 的 YOLOv8 目标检测项目
-- YOLOv8 是 Ultralytics 开发的最新版本 YOLO 目标检测算法
-- 支持实时目标检测、分类和分割任务，具有更高的精度和更快的推理速度
-- XMake 是一个基于 Lua 的现代构建工具，语法比 CMake 更简洁直观

-- 项目基本信息
-- set_project(): 设置项目名称，用于生成的可执行文件和项目标识
set_project("yolov8_demo")
-- set_version(): 设置项目版本，用于版本管理和发布
set_version("1.0.0")

-- C++ 标准设置
-- 为什么选择 C++17：
-- 1. TensorRT 8.6 推荐使用 C++17 标准
-- 2. C++17 提供了更多现代特性：结构化绑定、if constexpr、std::optional 等
-- 3. 更好的类型推导和模板支持
-- 4. YOLOv8 项目可以利用现代 C++ 特性提高代码质量
set_languages("c++17")

-- 构建模式配置
-- add_rules(): 添加预定义的构建规则
-- "mode.debug": 启用调试模式 - 包含调试符号，禁用优化，便于调试
-- "mode.release": 启用发布模式 - 启用优化，去除调试符号，提高性能
-- 为什么需要两种模式：开发调试时用 debug，部署发布时用 release
add_rules("mode.debug", "mode.release")

-- 路径变量定义
-- 为什么使用变量：
-- 1. 集中管理路径，修改时只需改一处
-- 2. 提高可读性和可维护性
-- 3. 便于在不同开发环境间移植
-- 注意：请根据你的实际安装路径修改这些变量
local tensorrt_root = "F:/TensorRT-8.6.1.6"        -- TensorRT 推理引擎库
local cudnn_root = "F:/cudnn_64-8.9.0.131"         -- NVIDIA 深度学习加速库
local cuda_root = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1"  -- CUDA 工具包
local opencv_root = "F:/opencv_cpu_install"        -- OpenCV 计算机视觉库

-- ============================================================================
-- 主目标定义：YOLOv8 目标检测可执行程序
-- ============================================================================
target("yolov8_demo")
    -- 目标类型设置
    -- set_kind("binary"): 生成可执行文件（.exe）
    -- 其他选项：static（静态库）、shared（动态库）
    set_kind("binary")
    
    -- ========================================================================
    -- 源文件配置
    -- ========================================================================
    -- add_files(): 指定要编译的源文件
    -- 支持通配符，如 "src/*.cpp" 或 "src/**.cpp"（递归）
    add_files(
        "main.cpp",                 -- 主程序入口文件
        "src/yolov8_trt_demo.cpp"   -- YOLOv8 TensorRT 检测器实现
    )
    
    -- ========================================================================
    -- 头文件包含目录配置
    -- ========================================================================
    -- add_includedirs(): 告诉编译器在哪里查找头文件
    -- 相当于 gcc 的 -I 参数或 Visual Studio 的"附加包含目录"
    add_includedirs(
        "include",                      -- 项目头文件目录
        tensorrt_root .. "/include",    -- TensorRT 头文件：NvInfer.h 等
        cudnn_root .. "/include",       -- cuDNN 头文件：cudnn.h 等
        cuda_root .. "/include",        -- CUDA 头文件：cuda_runtime.h 等
        opencv_root .. "/include",      -- OpenCV 头文件：opencv2/opencv.hpp 等
        {public = true}                 -- 设置为公共包含目录
    )
    
    -- ========================================================================
    -- 库文件搜索目录配置
    -- ========================================================================
    -- add_linkdirs(): 告诉链接器在哪里查找库文件
    -- 相当于 gcc 的 -L 参数或 Visual Studio 的"附加库目录"
    add_linkdirs(
        tensorrt_root .. "/lib",        -- TensorRT 库文件目录
        cudnn_root .. "/lib/x64",       -- cuDNN 64位库文件目录
        cuda_root .. "/lib/x64",        -- CUDA 64位库文件目录
        opencv_root .. "/lib"           -- OpenCV 库文件目录
    )
    
    -- ========================================================================
    -- 链接库配置
    -- ========================================================================
    -- add_links(): 指定要链接的库文件
    -- 相当于 gcc 的 -l 参数或 Visual Studio 的"附加依赖项"
    add_links(
        -- TensorRT 核心库
        "nvinfer",          -- 核心推理引擎：创建网络、执行推理
        "nvinfer_plugin",   -- 插件支持：自定义层和操作
        "nvonnxparser",     -- ONNX 模型解析器：加载 ONNX 格式模型
        "nvparsers",        -- 其他格式解析器：Caffe、UFF 等
        
        -- CUDA 运行时库
        "cudart",           -- CUDA 运行时：内存管理、流操作等
        "cublas",           -- CUDA 线性代数库：矩阵运算加速
        "curand",           -- CUDA 随机数生成库：随机数生成
        
        -- cuDNN 深度学习库
        "cudnn"             -- 深度学习原语：卷积、池化、激活函数等
    )
    
    -- ========================================================================
    -- OpenCV 库配置（版本相关）
    -- ========================================================================
    -- OpenCV 库名包含版本号，需要根据实际安装的版本调整
    -- 例如：opencv_core480 表示 OpenCV 4.8.0 版本的 core 模块
    -- YOLOv8 项目需要的 OpenCV 模块：
    add_links(
        "opencv_core480",       -- 核心模块：基本数据结构（Mat、Point等）
        "opencv_imgproc480",    -- 图像处理：滤波、变换、形态学操作、轮廓检测
        "opencv_imgcodecs480",  -- 图像编解码：读写 JPEG、PNG 等格式
        "opencv_highgui480",    -- 高级GUI：窗口显示、鼠标键盘事件
        "opencv_videoio480",    -- 视频I/O：读写视频文件、摄像头捕获
        "opencv_video480",      -- 视频分析：光流、背景减除、目标跟踪
        "opencv_dnn480"         -- 深度神经网络：blobFromImage、NMSBoxes等函数
    )
    
    -- OpenCV CUDA 模块（可选，仅在OpenCV编译时包含CUDA支持时启用）
    -- 如果你的OpenCV包含CUDA支持，可以取消注释以下行：
    -- add_links(
    --     "opencv_cudaimgproc480", -- CUDA图像处理：GPU加速的图像处理操作
    --     "opencv_cudawarping480", -- CUDA几何变换：GPU加速的图像变换
    --     "opencv_cudafilters480", -- CUDA滤波器：GPU加速的图像滤波
    --     "opencv_cudaarithm480"   -- CUDA算术运算：GPU加速的数学运算
    -- )
    
    -- ========================================================================
    -- 平台特定配置
    -- ========================================================================
    -- Windows 特定配置
    if is_plat("windows") then
        -- 运行时库路径配置
        -- add_rpathdirs(): 设置运行时动态库搜索路径
        add_rpathdirs(
            tensorrt_root .. "/lib",
            cudnn_root .. "/lib/x64",
            cuda_root .. "/bin",
            opencv_root .. "/bin"
        )
        
        -- MSVC 编译器特定选项
        if is_mode("release") then
            add_cxflags("/O2")      -- 启用优化
            add_cxflags("/DNDEBUG") -- 定义 NDEBUG 宏
        else
            add_cxflags("/Od")      -- 禁用优化
            add_cxflags("/DDEBUG")  -- 定义 DEBUG 宏
        end
        
        -- 设置运行时库
        add_cxflags("/MD")          -- 使用多线程 DLL 运行时库
        
        -- 构建后自动复制必要的 DLL 文件到输出目录
        after_build(function (target)
            local targetdir = target:targetdir()
            print("正在复制必要的 DLL 文件到输出目录: " .. targetdir)
            
            -- 复制 OpenCV DLL（使用通配符复制所有相关DLL）
            print("复制 OpenCV DLL...")
            os.trycp(opencv_root .. "/bin/opencv_*.dll", targetdir)
            
            -- 复制 TensorRT DLL（使用通配符复制所有DLL）
            print("复制 TensorRT DLL...")
            os.trycp(tensorrt_root .. "/lib/*.dll", targetdir)
            
            -- 复制 CUDA DLL（包含所有必要的CUDA运行时库）
            print("复制 CUDA DLL...")
            os.trycp(cuda_root .. "/bin/cudart64*.dll", targetdir)
            os.trycp(cuda_root .. "/bin/cublas64*.dll", targetdir)
            os.trycp(cuda_root .. "/bin/cublasLt64*.dll", targetdir)
            os.trycp(cuda_root .. "/bin/curand64*.dll", targetdir)
            -- 添加更多可能需要的CUDA DLL
            os.trycp(cuda_root .. "/bin/cusparse64*.dll", targetdir)
            os.trycp(cuda_root .. "/bin/cusolver64*.dll", targetdir)
            os.trycp(cuda_root .. "/bin/cufft64*.dll", targetdir)
            
            -- 复制 cuDNN DLL（从正确的路径复制）
            print("复制 cuDNN DLL...")
            os.trycp(cudnn_root .. "/bin/*.dll", targetdir)
            os.trycp(cudnn_root .. "/lib/x64/*.dll", targetdir)
            
            -- 复制项目资源文件（如果存在）
            print("复制项目资源文件...")
            os.trycp("*.engine", targetdir)        -- TensorRT 引擎文件
            os.trycp("*.trt", targetdir)           -- TensorRT 引擎文件
            os.trycp("classes.txt", targetdir)     -- 类别标签文件
            os.trycp("*.jpg", targetdir)           -- 测试图像
            os.trycp("*.png", targetdir)           -- 测试图像
            os.trycp("*.mp4", targetdir)           -- 测试视频
            
            print("✓ 所有必要文件已复制完成")
        end)
        
    -- Linux 特定配置
    elseif is_plat("linux") then
        -- GCC/Clang 编译选项
        if is_mode("release") then
            add_cxflags("-O3")          -- 最高级别优化
            add_cxflags("-DNDEBUG")     -- 定义 NDEBUG 宏
            add_cxflags("-march=native") -- 针对当前CPU优化
        else
            add_cxflags("-O0")          -- 禁用优化
            add_cxflags("-g")           -- 包含调试信息
            add_cxflags("-DDEBUG")      -- 定义 DEBUG 宏
        end
        
        -- 添加常用的编译警告
        add_cxflags("-Wall", "-Wextra", "-Wpedantic")
        
        -- 设置运行时库路径
        add_rpathdirs("$ORIGIN")
        add_rpathdirs("/usr/local/cuda/lib64")
        add_rpathdirs("/usr/local/lib")
    end
    
    -- ========================================================================
    -- 预处理器定义
    -- ========================================================================
    -- add_defines(): 定义预处理器宏
    add_defines(
        "OPENCV_VERSION_4",     -- 标识使用 OpenCV 4.x 版本
        "TENSORRT_VERSION_8",   -- 标识使用 TensorRT 8.x 版本
        "YOLOV8_VERSION"        -- 标识 YOLOv8 版本
    )
    
    -- 根据构建模式添加不同的宏定义
    if is_mode("debug") then
        add_defines("YOLOV8_DEBUG_MODE")
    end
    
    -- 在构建前执行配置验证
    before_build(function (target)
        print("正在验证项目配置...")
        
        -- 检查关键源文件
        if not os.isfile("main.cpp") then
            raise("错误：找不到 main.cpp 文件")
        end
        
        if not os.isfile("src/yolov8_trt_demo.cpp") then
            raise("错误：找不到 src/yolov8_trt_demo.cpp 文件")
        end
        
        if not os.isfile("include/yolov8_trt_demo.h") then
            raise("错误：找不到 include/yolov8_trt_demo.h 文件")
        end
        
        -- 检查 TensorRT 安装
        if not os.isdir(tensorrt_root) then
            print("警告：TensorRT 路径不存在: " .. tensorrt_root)
            print("请修改 xmake.lua 中的 tensorrt_root 变量")
        end
        
        -- 检查 OpenCV 安装
        if not os.isdir(opencv_root) then
            print("警告：OpenCV 路径不存在: " .. opencv_root)
            print("请修改 xmake.lua 中的 opencv_root 变量")
        end
        
        print("✓ 配置验证完成")
    end)

-- ============================================================================
-- 可选目标：性能测试程序（暂时禁用，需要创建 benchmark.cpp）
-- ============================================================================
-- target("yolov8_benchmark")
--     set_kind("binary")
--     set_default(false)  -- 默认不构建，需要手动指定
--     
--     add_files("benchmark.cpp", "src/yolov8_trt_demo.cpp")
--     add_includedirs("include")
--     
--     -- 继承主目标的配置
--     add_deps("yolov8_demo")
--     
--     -- 性能测试特定的编译选项
--     if is_mode("release") then
--         add_cxflags("-DBENCHMARK_MODE")
--     end

-- ============================================================================
-- 可选目标：单元测试程序（暂时禁用，需要创建 test 目录和测试文件）
-- ============================================================================
-- target("yolov8_test")
--     set_kind("binary")
--     set_default(false)  -- 默认不构建，需要手动指定
--     
--     add_files("test/*.cpp", "src/yolov8_trt_demo.cpp")
--     add_includedirs("include", "test")
--     
--     -- 继承主目标的配置
--     add_deps("yolov8_demo")
--     
--     -- 测试特定的编译选项
--     add_defines("UNIT_TEST_MODE")

-- ============================================================================
-- 自定义任务：清理生成的引擎文件
-- ============================================================================
task("clean-engines")
    on_run(function ()
        print("清理 TensorRT 引擎文件...")
        os.rm("*.engine")
        os.rm("*.trt")
        print("✓ 引擎文件清理完成")
    end)
    set_menu {
        usage = "xmake clean-engines",
        description = "清理生成的 TensorRT 引擎文件",
    }

-- ============================================================================
-- 自定义任务：显示项目信息
-- ============================================================================
task("info")
    on_run(function ()
        print("=== YOLOv8 TensorRT 目标检测项目信息 ===")
        print("项目名称: yolov8_demo")
        print("项目版本: 1.0.0")
        print("C++ 标准: C++17")
        print("构建模式: " .. (get_config("mode") or "release"))
        print("目标平台: " .. (get_config("plat") or "windows"))
        print("目标架构: " .. (get_config("arch") or "x64"))
        print("")
        print("主要功能:")
        print("  - YOLOv8 目标检测推理")
        print("  - TensorRT GPU 加速")
        print("  - 实时视频处理")
        print("  - 多类别目标识别")
        print("")
        print("支持的应用场景:")
        print("  - 实时目标检测")
        print("  - 视频监控分析")
        print("  - 自动驾驶感知")
        print("  - 工业质检")
        print("=====================================")
    end)
    set_menu {
        usage = "xmake info",
        description = "显示项目详细信息",
    }