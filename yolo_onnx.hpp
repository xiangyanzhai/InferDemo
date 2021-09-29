#pragma once

#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<vector>
#include "process.hpp"
#include "onnx_infer.hpp"

clock_t startTime, endTime;

class YOLO {
public:
    OnnxInfer *onnx_infer;

    YOLO(const wchar_t *onnx_file) {
        this->onnx_infer = new OnnxInfer(onnx_file);
    }

    void infer(std::vector<cv::Mat> &frames, std::vector<std::vector<bbox>> &res, int input_w, int input_h) {
        cv::Mat blob;
        preprocess_yolo(blob, frames, input_w, input_h);
        res.resize(blob.size[0]);
        for (int i = 0; i < res.size(); i++) {
            res[i].clear();
        }
        this->onnx_infer->infer(blob);
        postprocess_yolo_onnx(res, this->onnx_infer->outputs, this->onnx_infer->output_shapes);
    }

    ~YOLO() {
        if (this->onnx_infer != nullptr) {
            delete this->onnx_infer;
        }
    }
};

class YOLOFastest {
public:
    OnnxInfer *onnx_infer;

    YOLOFastest(const wchar_t *onnx_file) {
        this->onnx_infer = new OnnxInfer(onnx_file);
    }

    void infer(std::vector<cv::Mat> &frames, std::vector<std::vector<bbox>> &res, int input_w, int input_h) {
        cv::Mat blob;
        preprocess_yolo(blob, frames, input_w, input_h);
        res.resize(blob.size[0]);
        for (int i = 0; i < res.size(); i++) {
            res[i].clear();
        }
        this->onnx_infer->infer(blob);
        postprocess_yolo_fastest_onnx(res, this->onnx_infer->outputs, this->onnx_infer->output_shapes, (float) input_w,
                                      (float) input_h);
    }

    ~YOLOFastest() {
        if (this->onnx_infer != nullptr) {
            delete this->onnx_infer;
        }
    }
};