#pragma once

#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include<vector>

struct bbox {
    double x1;
    double y1;
    double w;
    double h;
    double score;
    int cls;
};

bool my_sort(bbox &a, bbox &b) {
    return a.score > b.score;
}

void preprocess_yolo(cv::Mat &blob, std::vector<cv::Mat> &frames, int input_w, int input_h) {
    blob = cv::dnn::blobFromImages(frames, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
}

void get_res(std::vector<bbox> &res, std::vector<cv::Rect2d> &bboxes, cv::Mat &temp, int num_cls,
             float conf_thresh = 0.3, float nms_thresh = 0.45) {
    cv::Point classIdPoint;
    double confidence;
    for (int i = 0; i < num_cls; i++) {
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, temp.colRange(i, i + 1), 0.005, nms_thresh, indices);
        std::sort(indices.begin(), indices.end());
        for (int j = 0, k = 0; j < temp.rows; j++) {
            if (k < indices.size()) {
                if (j != indices[k]) {
                    temp.at<float>(j, i) = 0.0f;
                } else {
                    k++;
                }
            } else {
                temp.at<float>(j, i) = 0.0f;
            }
        }
    }
    for (int i = 0; i < temp.rows; i++) {
        minMaxLoc(temp.row(i), 0, &confidence, 0, &classIdPoint);

        if (confidence < conf_thresh) continue;

        res.push_back({bboxes[i].x, bboxes[i].y, bboxes[i].width, bboxes[i].height, confidence, classIdPoint.x});
    }
    std::sort(res.begin(), res.end(), my_sort);
}

void postprocess_yolo_onnx(std::vector<std::vector<bbox>> &res, const std::vector<float *> &output,
                           std::vector<std::vector<int>> &shapes, float conf_thresh = 0.3,
                           float nms_thresh = 0.45) {
    for (int i = 0; i < res.size(); i++) res[i].clear();
    float *boxes = output[0];
    float *confs = output[1];

    int batch = shapes[1][0];
    int num_cls = shapes[1][2];
    float max_value = -1e20;
    float conf_temp;
    cv::Mat temp;
    std::vector<cv::Mat> temp_batch(batch);
    std::vector<std::vector<cv::Rect2d>> bboxes(batch);
    std::vector<std::vector<cv::Mat>> scores(batch);

    int stride1 = shapes[0][1] * shapes[0][3];
    int stride2 = shapes[1][1] * shapes[1][2];
    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < shapes[0][1]; j++) {
            max_value = -1e20;
            for (int k = 0; k < num_cls; k++) {
                conf_temp = confs[i * stride2 + j * num_cls + k];
                if (max_value < conf_temp) {
                    max_value = conf_temp;
                }
            }
            if (max_value < conf_thresh) {
                continue;
            }
            bboxes[i].push_back(cv::Rect2d(boxes[i * stride1 + j * 4 + 0], boxes[i * stride1 + j * 4 + 1],
                                           boxes[i * stride1 + j * 4 + 2] - boxes[i * stride1 + j * 4 + 0],
                                           boxes[i * stride1 + j * 4 + 3] - boxes[i * stride1 + j * 4 + 1]));
            temp = cv::Mat(1, num_cls, CV_32F);
            for (int k = 0; k < num_cls; k++) {
                temp.at<float>(0, k) = confs[i * stride2 + j * num_cls + k];
            }
            scores[i].push_back(temp);

        }
    }
    for (int i = 0; i < batch; i++) {
        if (bboxes[i].empty()) continue;
        cv::vconcat(scores[i], temp);
        get_res(res[i], bboxes[i], temp, num_cls, conf_thresh, nms_thresh);
    }
}

void postprocess_yolo_fastest_onnx(std::vector<std::vector<bbox>> &res, std::vector<float *> &output,
                                   std::vector<std::vector<int>> &shapes, const float &input_w,
                                   const float &input_h, float conf_thresh = 0.3, float nms_thresh = 0.45) {
    for (int i = 0; i < res.size(); i++) res[i].clear();
    float *out = output[0];
    int batch = shapes[0][0];
    int num_cls = shapes[0][2] - 5;
    float conf_temp;
    int stride = shapes[0][1] * shapes[0][2];
    cv::Mat temp;
    std::vector<cv::Mat> temp_batch(batch);
    std::vector<std::vector<cv::Rect2d>> bboxes(batch);
    std::vector<std::vector<cv::Mat>> scores(batch);
    float x1, y1, x2, y2;

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < shapes[0][1]; j++) {
            conf_temp = out[i * stride + j * (num_cls + 5) + 4];
            if (conf_temp < conf_thresh) {
                continue;
            }
            x1 = out[i * stride + j * (num_cls + 5) + 0];
            y1 = out[i * stride + j * (num_cls + 5) + 1];
            x2 = out[i * stride + j * (num_cls + 5) + 2];
            y2 = out[i * stride + j * (num_cls + 5) + 3];
            if (x1 < 0) x1 = 0.0f;
            if (x1 > input_w) x1 = input_w;
            if (x2 < 0) x2 = 0.0f;
            if (x2 > input_w) x2 = input_w;
            if (y1 < 0) y1 = 0.0f;
            if (y1 > input_h) y1 = input_h;
            if (y2 < 0) y2 = 0.0f;
            if (y2 > input_h) y2 = input_h;

            bboxes[i].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            temp = cv::Mat(1, num_cls, CV_32F);
            for (int k = 0; k < num_cls; k++) {
                temp.at<float>(0, k) = out[i * stride + j * (num_cls + 5) + 5 + k] * conf_temp;
            }
            scores[i].push_back(temp);

        }
    }
    for (int i = 0; i < batch; i++) {
        if (bboxes[i].empty()) continue;
        cv::vconcat(scores[i], temp);
        get_res(res[i], bboxes[i], temp, num_cls, conf_thresh, nms_thresh);
    }
}

void postprocess_cv(std::vector<std::vector<bbox>> &res, const std::vector<cv::Mat> &outs, float conf_thresh = 0.3,
                    float nms_thresh = 0.45) {
    for (int i = 0; i < res.size(); i++) res[i].clear();
    int batch, num_cls;
    if (outs[0].dims == 3) {
        batch = outs[0].size[0];
        num_cls = outs[0].size[2] - 5;
    } else {
        batch = 1;
        num_cls = outs[0].size[1] - 5;
    }

    std::vector<cv::Mat> temp_batch(batch);
    std::vector<std::vector<cv::Rect2d>> bboxes(batch);
    std::vector<std::vector<cv::Mat>> scores(batch);
    int n, stride;
    float x, y, w, h;
    cv::Mat score;
    for (auto out: outs) {
        if (out.dims == 3) {
            n = out.size[1];
            stride = out.size[1] * out.size[2];
        } else {
            n = out.size[0];
            stride = out.size[0] * out.size[1];
        }
        float *data = (float *) out.data;
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < n; j++) {
                if (data[i * stride + j * (num_cls + 5) + 4] < conf_thresh) continue;
                x = data[i * stride + j * (num_cls + 5) + 0];
                y = data[i * stride + j * (num_cls + 5) + 1];
                w = data[i * stride + j * (num_cls + 5) + 2];
                h = data[i * stride + j * (num_cls + 5) + 3];
                bboxes[i].push_back(cv::Rect2d(x - w / 2, y - h / 2, w, h));
                score = cv::Mat(1, num_cls, CV_32F);
                for (int k = 0; k < num_cls; k++) {
                    score.at<float>(0, k) = data[i * stride + j * (num_cls + 5) + 5 + k];
                }
                scores[i].push_back(score);
            }
        }

    }
    cv::Mat temp;
    for (int i = 0; i < batch; i++) {
        if (bboxes[i].empty()) continue;
        cv::vconcat(scores[i], temp);
        get_res(res[i], bboxes[i], temp, num_cls, conf_thresh, nms_thresh);
    }


}

std::vector<std::string> getOutputsNames(const cv::dnn::Net &net) {
    static std::vector<std::string> names;
    if (names.empty()) {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}