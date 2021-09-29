#pragma once

#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<vector>
#include "process.hpp"

class YOLO_cv {
public:
    std::string modelConfiguration;
    std::string modelWeights;
    int inpWidth;
    int inpHeight;
    bool equal_scale;
    cv::dnn::Net net;

    YOLO_cv(std::string modelConfiguration, std::string modelWeights, int inpWidth, int inpHeight, bool equal_scale) {

        this->modelConfiguration = modelConfiguration;
        this->modelWeights = modelWeights;
        this->inpWidth = inpWidth;
        this->inpHeight = inpHeight;
        this->equal_scale = equal_scale;
        this->net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

    }

    void infer(std::vector<cv::Mat> &frames, std::vector<std::vector<bbox>> &res, int input_w, int input_h) {
        cv::Mat blob;
        preprocess_yolo(blob, frames, input_w, input_h);
        res.resize(blob.size[0]);
        for (int i = 0; i < res.size(); i++) {
            res[i].clear();
        }
        std::vector<cv::Mat> outs;
        startTime = clock();
        this->net.setInput(blob);
        this->net.forward(outs, getOutputsNames(this->net));
        endTime = clock();
        std::cout << "The infer time is: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        postprocess_cv(res, outs);
    }
};