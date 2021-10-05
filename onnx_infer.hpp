#pragma once
#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<vector>

extern clock_t startTime, endTime;



class OnnxInfer {
public:
    const wchar_t *onnx_file;
    Ort::Session *session;
    std::vector<const char *> input_node_names;
    std::vector<const char *> output_node_names;
    std::vector<float *> outputs;
    std::vector<std::vector<int>> output_shapes;
    std::vector<Ort::Value> ort_inputs;
    Ort::Env *env;
    OnnxInfer(const wchar_t *onnx_file) {

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::SessionOptions session_options;
        this->onnx_file = onnx_file;
        session_options.SetIntraOpNumThreads(4);

        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        this->env=new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test5");
        this->session = new Ort::Session(*this->env, onnx_file, session_options);

        for (int i = 0; i < this->session->GetInputCount(); i++) {
            this->input_node_names.push_back(session->GetInputName(i, allocator));
        }
        for (int i = 0; i < this->session->GetOutputCount(); i++) {
            this->output_node_names.push_back(session->GetOutputName(i, allocator));
        }
        this->outputs.resize(session->GetOutputCount());
        this->output_shapes.resize(session->GetOutputCount());
    }

    ~OnnxInfer() {
        if (this->session != nullptr) {
            delete this->session;
        }
        if (this->env != nullptr){
            delete this->env;
        }
    }

    void infer(const cv::Mat &blob) {
        auto mask_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        int num = 1;
        std::vector<int64_t> shape;
        for (int i = 0; i < blob.dims; i++) {
            num *= blob.size[i];
            shape.push_back(blob.size[i]);
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mask_memory_info, (float *) blob.data, num,
                                                                  shape.data(), blob.dims);

        this->ort_inputs.clear();
        this->ort_inputs.push_back(std::move(input_tensor));

        startTime = clock();
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, this->input_node_names.data(),
                                           ort_inputs.data(),
                                           ort_inputs.size(), this->output_node_names.data(),
                                           this->output_node_names.size());
        endTime = clock();
        std::cout << "onnx infer time:" << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        for (int i = 0; i < output_tensors.size(); i++) {
            this->outputs[i] = (float *) output_tensors[i].GetTensorMutableData<float>();
            this->output_shapes[i].clear();
            for (int j = 0; j < output_tensors[i].GetTensorTypeAndShapeInfo().GetDimensionsCount(); j++) {
                this->output_shapes[i].push_back((int) output_tensors[i].GetTensorTypeAndShapeInfo().GetShape()[j]);
            }
        }
    }
};