#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "include/yolov8_trt_demo.h"

int main() {
    // 初始化
    const std::string videofile = "firesmog.mp4";
    const std::string enginefile = "firesmokev1.engine";
    
    cv::VideoCapture cap(videofile);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件：" << videofile << std::endl;
        return -1;
    }
    
    auto detector = std::make_shared<YOLOv8TRTDetector>();
    detector->initConfig(enginefile, 0.25f, 0.25f);
    
    cv::Mat frame;
    std::vector<DetectResult> results;
    int frame_count = 0;
    
    std::cout << "开始检测，按ESC键退出..." << std::endl;
    
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "视频处理完成" << std::endl;
            break;
        }
        
        frame_count++;
        detector->detect(frame, results);
        
        // 检查是否同时存在火和烟
        bool has_fire = false, has_smoke = false;
        for (const auto& result : results) {
            if (result.class_id == 0) has_fire = true;   // fire
            if (result.class_id == 1) has_smoke = true;  // smoke
        }
        
        // 只有火和烟同时存在时才绘制和保存
        std::string save_dir = "E:/smog_detv8/detected_frames";
        if (!std::filesystem::exists(save_dir)) {
            std::filesystem::create_directory(save_dir);
        }
        
        if (has_fire && has_smoke) {
            // 绘制烟雾检测框
            for (const auto& result : results) {
                if (result.class_id == 1) {  // 只绘制烟雾
                    const cv::Rect& box = result.box;
                    const cv::Scalar box_color(0, 165, 255);  // 橙色框
                    
                    std::string label = "SMOKE " + cv::format("%.2f", result.conf) + " [FIRE+SMOKE ALERT!]";
                    
                    // 绘制检测框
                    cv::rectangle(frame, box, box_color, 3);
                    
                    // 绘制标签背景和文字
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                    cv::Point text_origin(box.tl().x, std::max(15, box.tl().y - 10));
                    cv::Rect text_bg(text_origin.x, text_origin.y - text_size.height - 5,
                                    text_size.width + 10, text_size.height + 10);
                    
                    cv::rectangle(frame, text_bg, cv::Scalar(0, 0, 0), -1);
                    cv::putText(frame, label, cv::Point(text_bg.x + 5, text_bg.y + text_size.height + 2),
                              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                }
            }
            
            // 绘制警报状态
            std::string status = "ALERT: SMOKE DETECTED with FIRE!";
            cv::Size status_size = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
            cv::Rect status_bg(10, 10, status_size.width + 20, status_size.height + 20);
            cv::rectangle(frame, status_bg, cv::Scalar(0, 0, 0), -1);
            cv::putText(frame, status, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            
            // 保存检测结果帧
            std::string save_name = save_dir + "/frame_" + cv::format("%04d", frame_count) + ".jpg";
            cv::imwrite(save_name, frame);
            std::cout << "检测到火与烟，已保存: " << save_name << std::endl;
        } else {
            // 绘制正常状态
            std::string status = "Normal - No Fire+Smoke Condition";
            cv::Size status_size = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
            cv::Rect status_bg(10, 10, status_size.width + 20, status_size.height + 20);
            cv::rectangle(frame, status_bg, cv::Scalar(0, 0, 0), -1);
            cv::putText(frame, status, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }
        
        // 显示图像
        cv::imshow("Fire & Smoke Detection", frame);
        
        // 处理按键
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) break;  // ESC退出
        
        results.clear();
    }
    
    // 清理资源
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "程序执行完成！总共处理了 " << frame_count << " 帧图像" << std::endl;
    
    return 0;
}
