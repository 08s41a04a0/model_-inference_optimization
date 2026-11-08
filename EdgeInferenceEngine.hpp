#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <functional>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct DetectionResult {
    cv::Rect box;
    float confidence;
    int class_id;
};

class ThreadSafeFrameQueue {
private:
    std::queue<cv::Mat> queue;
    std::mutex mtx;
    std::condition_variable cv;
    size_t max_size;
public:
    ThreadSafeFrameQueue(size_t limit = 3) : max_size(limit) {}
    void push(const cv::Mat& frame);
    bool pop(cv::Mat& frame, std::atomic<bool>& running_flag);
    bool empty();
};

class ProductionTelemetryLogger {
private:
    std::ofstream csv_file;
    std::mutex mtx;
public:
    ProductionTelemetryLogger(const std::string& path);
    ~ProductionTelemetryLogger();
    void log_metrics(int id, double prep, double trans, double infer, double post, double total, double fps);
};

class EdgeInferenceEngine {
private:
    std::string model_path;
    std::atomic<bool> is_running{false};
    int num_threads;
    ThreadSafeFrameQueue frame_queue;
    std::thread processing_worker_thread;
    std::unique_ptr<ProductionTelemetryLogger> telemetry;
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions session_options;
    std::vector<int64_t> input_shape{1, 3, 640, 640};
    std::vector<float> input_tensor_mem;
    std::vector<const char*> input_names{"images"};
    std::vector<const char*> output_names{"output0"};
    std::function<void(const cv::Mat&, const std::vector<DetectionResult>&)> result_callback;
    void pipeline_consumer_loop();
    void execute_post_processing(float* raw_output, const std::vector<int64_t>& shape, cv::Mat& frame, std::vector<DetectionResult>& out_results);
public:
    EdgeInferenceEngine(const std::string& model, int execution_threads = std::thread::hardware_concurrency());
    ~EdgeInferenceEngine();
    bool initialize(const std::string& csv_log_path = "");
    void start_async_processing();
    void stop_processing();
    void push_frame(const cv::Mat& frame);
    void register_callback(std::function<void(const cv::Mat&, const std::vector<DetectionResult>&)> callback) {
        result_callback = callback;
    }
};
'@ | Out-File -FilePath EdgeInferenceEngine.hpp -Encoding utf8

# 4. Recreate the Class Implementation
@'
#include "EdgeInferenceEngine.hpp"
#include <chrono>

void ThreadSafeFrameQueue::push(const cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(mtx);
    if (queue.size() >= max_size) queue.pop();
    queue.push(frame.clone());
    lock.unlock();
    cv.notify_one();
}
bool ThreadSafeFrameQueue::pop(cv::Mat& frame, std::atomic<bool>& running_flag) {
    std::unique_lock<std::mutex> lock(mtx);
    while (queue.empty() && running_flag) {
        if (cv.wait_for(lock, std::chrono::milliseconds(50)) == std::cv_status::timeout && !running_flag) return false;
    }
    if (!running_flag && queue.empty()) return false;
    frame = queue.front();
    queue.pop();
    return true;
}
bool ThreadSafeFrameQueue::empty() { std::lock_guard<std::mutex> lock(mtx); return queue.empty(); }

ProductionTelemetryLogger::ProductionTelemetryLogger(const std::string& path) {
    csv_file.open(path, std::ios::out | std::ios::trunc);
    if (csv_file.is_open()) {
        csv_file << "frame_id,prep_ms,transpose_ms,inference_ms,post_ms,total_ms,fps\n";
        csv_file.flush();
    }
}
ProductionTelemetryLogger::~ProductionTelemetryLogger() { if (csv_file.is_open()) csv_file.close(); }
void ProductionTelemetryLogger::log_metrics(int id, double prep, double trans, double infer, double post, double total, double fps) {
    std::lock_guard<std::mutex> lock(mtx);
    if (csv_file.is_open()) {
        csv_file << id << "," << prep << "," << trans << "," << infer << "," << post << "," << total << "," << fps << "\n";
        if (id % 15 == 0) csv_file.flush();
    }
}

EdgeInferenceEngine::EdgeInferenceEngine(const std::string& model, int execution_threads) 
    : model_path(model), num_threads(execution_threads) {
    input_tensor_mem.resize(1 * 3 * 640 * 640);
}
EdgeInferenceEngine::~EdgeInferenceEngine() { stop_processing(); }

bool EdgeInferenceEngine::initialize(const std::string& csv_log_path) {
    try {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ProductionEngineInstance");
        session_options.SetIntraOpNumThreads(num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.AppendExecutionProvider_CPU(0);
        session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
        if (!csv_log_path.empty()) telemetry = std::make_unique<ProductionTelemetryLogger>(csv_log_path);
        return true;
    } catch (...) { return false; }
}
void EdgeInferenceEngine::start_async_processing() {
    if (is_running) return;
    is_running = true;
    processing_worker_thread = std::thread(&EdgeInferenceEngine::pipeline_consumer_loop, this);
}
void EdgeInferenceEngine::stop_processing() {
    if (!is_running) return;
    is_running = false;
    if (processing_worker_thread.joinable()) processing_worker_thread.join();
}
void EdgeInferenceEngine::push_frame(const cv::Mat& frame) { frame_queue.push(frame); }

void EdgeInferenceEngine::pipeline_consumer_loop() {
    cv::Mat frame, resized_image;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    int local_frame_id = 0;
    while (frame_queue.pop(frame, is_running)) {
        local_frame_id++;
        auto start_total = std::chrono::high_resolution_clock::now();
        double m_prep=0, m_trans=0, m_infer=0, m_post=0;

        auto t_start = std::chrono::high_resolution_clock::now();
        cv::resize(frame, resized_image, cv::Size(640, 640));
        cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
        resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);
        m_prep = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();

        t_start = std::chrono::high_resolution_clock::now();
        float* input_ptr = input_tensor_mem.data();
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 640; ++h) {
                for (int w = 0; w < 640; ++w) {
                    input_ptr[c * 640 * 640 + h * 640 + w] = resized_image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        m_trans = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_mem.data(), input_tensor_mem.size(), input_shape.data(), input_shape.size()
        );

        t_start = std::chrono::high_resolution_clock::now();
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        m_infer = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();

        t_start = std::chrono::high_resolution_clock::now();
        std::vector<DetectionResult> results;
        float* raw_output = output_tensors.GetTensorMutableData<float>();
        auto out_shape = output_tensors.GetTensorTypeAndShapeInfo().GetShape();
        execute_post_processing(raw_output, out_shape, frame, results);
        m_post = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();

        double total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_total).count();
        if (telemetry) telemetry->log_metrics(local_frame_id, m_prep, m_trans, m_infer, m_post, total_ms, 1000.0 / total_ms);
        if (result_callback) result_callback(frame, results);
    }
}

void EdgeInferenceEngine::execute_post_processing(float* raw_output, const std::vector<int64_t>& shape, cv::Mat& frame, std::vector<DetectionResult>& out_results) {
    int dimensions = shape, anchors = shape;
    const float CONF_THRESH = 0.25f, NMS_THRESH = 0.45f;
    std::vector<cv::Rect> boxes; std::vector<float> confidences;
    float scale_x = (float)frame.cols / 640.0f, scale_y = (float)frame.rows / 640.0f;
    for (int i = 0; i < anchors; ++i) {
        float max_score = 0.0f; int target_class_id = -1;
        for (int c = 4; c < dimensions; ++c) {
            float score = raw_output[c * anchors + i];
            if (score > max_score) { max_score = score; target_class_id = c - 4; }
        }
        if (max_score >= CONF_THRESH) {
            float xc = raw_output[0 * anchors + i], yc = raw_output[1 * anchors + i];
            float w  = raw_output[2 * anchors + i], h  = raw_output[3 * anchors + i];
            int left = (int)((xc - 0.5f * w) * scale_x), top = (int)((yc - 0.5f * h) * scale_y);
            boxes.emplace_back(cv::Rect(left, top, (int)(w * scale_x), (int)(h * scale_y)));
            confidences.emplace_back(max_score);
        }
    }
    std::vector<int> nms_indices;