#pragma once

#include <iostream>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<vector>
#include "process.hpp"
#include "onnx_infer.hpp"

class SimplePose {
public:
    OnnxInfer *onnx_infer;

    SimplePose(const wchar_t *onnx_file) {
        this->onnx_infer = new OnnxInfer(onnx_file);
    }

    void infer(std::vector<cv::Mat> &frames, std::vector<std::vector<bbox_keypoints>> &res, int input_w, int input_h,
               std::vector<cv::Mat> &trans_vs) {
        cv::Mat blob;
        preprocess(blob, frames, input_w, input_h);
        std::cout << blob.size << std::endl;
        this->onnx_infer->infer(blob);
        postprocess_simplepose(res, this->onnx_infer->outputs, this->onnx_infer->output_shapes, trans_vs);
    }

    ~SimplePose() {
        if (this->onnx_infer != nullptr) {
            delete this->onnx_infer;
        }
    }
};