#include <onnxruntime_cxx_api.h>
#include <vector>
#include <array>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "wbflow");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);

    std::vector<int64_t> input_shape{1, 3, 256, 256};
    std::vector<float> image_data(input_shape[1] * input_shape[2] * input_shape[3], 0.5f);

    std::vector<int64_t> cam_shape{1, 15, 1, 1};
    std::vector<float> cam_data(15, 0.f);
    cam_data[0] = 1.f; // example: first camera index

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value img_tensor = Ort::Value::CreateTensor<float>(memory_info, image_data.data(), image_data.size(), input_shape.data(), input_shape.size());
    Ort::Value cam_tensor = Ort::Value::CreateTensor<float>(memory_info, cam_data.data(), cam_data.size(), cam_shape.data(), cam_shape.size());

    const char* input_names[] = {"input", "camidx"};
    const char* output_names[] = {"output"};
    std::array<Ort::Value, 2> inputs = {std::move(img_tensor), std::move(cam_tensor)};

    auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs.data(), inputs.size(), output_names, 1);

    float* result = outputs.front().GetTensorMutableData<float>();
    std::cout << "First output value: " << result[0] << std::endl;

    return 0;
}
