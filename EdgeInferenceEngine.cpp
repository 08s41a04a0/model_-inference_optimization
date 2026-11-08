#include "EdgeInferenceEngine.hpp"
void on_frame_processed_callback(const cv::Mat& visual_frame, const std::vector& objects_detected) {
cv::Mat canvas = visual_frame.clone();
for (const auto& obj : objects_detected) {
cv::rectangle(canvas, obj.box, cv::Scalar(0, 0, 255), 2);
}
cv::imshow("Consumer Presentation View", canvas);
cv::waitKey(1);
}
int main() {
EdgeInferenceEngine engine("yolov8n_int8_cpu.onnx");
if (!engine.initialize("production_telemetry.csv")) return -1;
engine.register_callback(on_frame_processed_callback);
engine.start_async_processing();
cv::VideoCapture video("test_stream.mp4");
cv::Mat incoming_raw_frame;
while (video.read(incoming_raw_frame)) { engine.push_frame(incoming_raw_frame); }
engine.stop_processing();
return 0;
}