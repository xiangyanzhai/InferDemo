#pragma once

#include <iostream>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<vector>
#include "process.hpp"
#include "onnx_infer.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

using namespace cv;
using namespace human_pose_estimation;
human_pose_estimation::HumanPoseEstimator estimator(false);

class LightPose {
public:
    OnnxInfer *onnx_infer;

    LightPose(const wchar_t *onnx_file) {
        this->onnx_infer = new OnnxInfer(onnx_file);
    }

    LightPose(const wchar_t *onnx_file, std::string name) {
        this->onnx_infer = new OnnxInfer(onnx_file, name);
    }

    void infer(std::vector<cv::Mat> &frames, std::vector<std::vector<cv::Point2f>> &res, int input_w, int input_h) {
        cv::Mat blob;
        preprocess_openpose(blob, frames, input_w, input_h);
        this->onnx_infer->infer(blob);
        int batch = frames.size();
        float *heatmap = (float *) this->onnx_infer->outputs[2];
        float *pafs = (float *) this->onnx_infer->outputs[3];
        res.resize(batch);
        float a, b;
        std::vector<human_pose_estimation::HumanPose> poses;
        for (int i = 0; i < batch; i++) {
            res[i].clear();
            poses = estimator.estimate(&heatmap[i * 19 * 32 * 57], &pafs[i * 38 * 32 * 57]);
            for (human_pose_estimation::HumanPose const &pose: poses) {
                for (auto const &keypoint: pose.keypoints) {
                    a = keypoint.x * 2;
                    b = keypoint.y * 2;
                    if (a < 0.0 || b < 0.0) {
                        a = -1.0;
                        b = -1.0;
                    }
                    res[i].emplace_back(a, b);
                }
            }
        }
    }


    ~LightPose() {
        if (this->onnx_infer != nullptr) {
            delete this->onnx_infer;
        }
    }

};