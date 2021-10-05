#pragma once
#define M_PI       3.14159265358979323846

#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include<vector>

const float mean_[3] = {0.485, 0.456, 0.406};
const float std_[3] = {0.229, 0.224, 0.225};
struct bbox {
    double x1;
    double y1;
    double w;
    double h;
    double score;
    int cls;
};
struct bbox_keypoints : bbox {
    float keypoints[17][3];
};
struct point {
    float a;
    float b;
};

bool my_sort(bbox &a, bbox &b) {
    return a.score > b.score;
}

void preprocess_yolo(cv::Mat &blob, std::vector<cv::Mat> &frames, int input_w, int input_h) {
    blob = cv::dnn::blobFromImages(frames, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
}

void preprocess(cv::Mat &blob, std::vector<cv::Mat> &frames, int input_w, int input_h) {
    blob = cv::dnn::blobFromImages(frames, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
    float *ptr = (float *) blob.data;
    for (int i = 0; i < frames.size(); i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < input_w * input_h; k++) {
                ptr[i * 3 * input_w * input_h + j * input_w * input_h + k] =
                        (ptr[i * 3 * input_w * input_h + j * input_w * input_h + k] - mean_[j]) / std_[j];
            }
        }
    }
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

//void xywh2cs(float x, float y, float w, float h, std::array<float, 2> &center, std::array<float, 2> &scale) {
//    center[0] = x + w * 0.5;
//    center[1] = y + h * 0.5;
//
//    float aspect_ratio = 0.75;
//    float pixel_std = 200;
//    if (w > aspect_ratio * h) {
//        h = w * 1.0 / aspect_ratio;
//    } else {
//        w = h * aspect_ratio;
//    }
//    scale[0] = (w * 1.0 / pixel_std);
//    scale[1] = (h * 1.0 / pixel_std);
//    if (center[0] != -1) {
//        scale[0] = scale[0] * 1.25;
//        scale[1] = scale[1] * 1.25;
//    }
//
//}
//
//void get_dir(std::array<float, 2> &src_point, float rot_rad, std::array<float, 2> &src_result) {
//    float sn = sin(rot_rad);
//    float cs = cos(rot_rad);
//    src_result[0] = src_point[0] * cs - src_point[1] * sn;
//    src_result[1] = src_point[0] * sn + src_point[1] * cs;
//}
//
//void get_3rd_point(std::array<std::array<float, 2>, 3> &src) {
//    src[2][0] = src[1][0] - src[0][1] + src[1][1];
//    src[2][1] = src[1][1] + src[0][0] - src[1][0];
//}
//
//auto get_affine_transform(std::array<float, 2> &center, std::array<float, 2> &scale, float rot,
//                          std::array<int, 2> &output_size, std::array<float, 2> shift, bool inv) {
//    std::array<float, 2> scale_tmp = {};
//    scale_tmp[0] = scale[0] * 200;
//    scale_tmp[1] = scale[1] * 200;
//    float src_w = scale_tmp[0];
//    int dst_w = output_size[0];
//    int dst_h = output_size[1];
//    float rot_rad = M_PI * rot / 180;
//
//    std::array<float, 2> src_dir = {};
//    std::array<float, 2> src_point = {0, static_cast<float>(src_w * -0.5)};
//    get_dir(src_point, rot_rad, src_dir);
//
//    std::array<float, 2> dst_dir = {0, static_cast<float>(dst_w * -0.5)};
//    std::array<std::array<float, 2>, 3> src = {};
//    std::array<std::array<float, 2>, 3> dst = {};
//    src[0][0] = center[0] + scale_tmp[0] * shift[0];
//    src[0][1] = center[1] + scale_tmp[1] * shift[1];
//    src[1][0] = center[0] + scale_tmp[0] * shift[0] + src_dir[0];
//    src[1][1] = center[1] + scale_tmp[1] * shift[1] + src_dir[1];
//
//    dst[0] = {static_cast<float>(dst_w * 0.5), static_cast<float>(dst_h * 0.5)};
//    dst[1] = {static_cast<float>(dst_w * 0.5 + dst_dir[0]), static_cast<float>(dst_h * 0.5 + dst_dir[1])};
//
//    get_3rd_point(src);
//    get_3rd_point(dst);
//
//    cv::Point2f point_src[3] = {};
//    cv::Point2f point_dst[3] = {};
//    for (int i = 0; i < 3; i++) {
//        point_src[i] = cv::Point2f(src[i][0], src[i][1]);
//        point_dst[i] = cv::Point2f(dst[i][0], dst[i][1]);
//    }
//    if (inv) {
//        return cv::getAffineTransform(point_dst, point_src);
//    } else {
//        return cv::getAffineTransform(point_src, point_dst);
//    }
//}

void xywh2cs(float x, float y, float w, float h, float center[2], float scale[2]) {
    center[0] = x + w * 0.5f;
    center[1] = y + h * 0.5f;

    float aspect_ratio = 0.75;
    float pixel_std = 200;
    if (w > aspect_ratio * h) {
        h = w * 1.0f / aspect_ratio;
    } else {
        w = h * aspect_ratio;
    }
    scale[0] = (w * 1.0f / pixel_std);
    scale[1] = (h * 1.0f / pixel_std);
    if (center[0] != -1) {
        scale[0] = scale[0] * 1.25f;
        scale[1] = scale[1] * 1.25f;
    }
}

void get_dir(const float src_point[2], float rot_rad, float src_result[2]) {
    float sn = sin(rot_rad);
    float cs = cos(rot_rad);
    src_result[0] = src_point[0] * cs - src_point[1] * sn;
    src_result[1] = src_point[0] * sn + src_point[1] * cs;
}

void get_3rd_point(float src[3][2]) {
    src[2][0] = src[1][0] - src[0][1] + src[1][1];
    src[2][1] = src[1][1] + src[0][0] - src[1][0];
}

auto get_affine_transform(const float center[2], const float scale[2], float rot,
                          const int output_size[2], const float shift[2], bool inv) {
    float scale_tmp[2] = {};
    scale_tmp[0] = scale[0] * 200;
    scale_tmp[1] = scale[1] * 200;
    float src_w = scale_tmp[0];
    int dst_w = output_size[0];
    int dst_h = output_size[1];
    float rot_rad = M_PI * rot / 180;

    float src_dir[2] = {};
    float src_point[2] = {0, static_cast<float>(src_w * -0.5)};
    get_dir(src_point, rot_rad, src_dir);

    float dst_dir[2] = {0, static_cast<float>(dst_w * -0.5)};
    float src[3][2] = {};
    float dst[3][2] = {};
    src[0][0] = center[0] + scale_tmp[0] * shift[0];
    src[0][1] = center[1] + scale_tmp[1] * shift[1];
    src[1][0] = center[0] + scale_tmp[0] * shift[0] + src_dir[0];
    src[1][1] = center[1] + scale_tmp[1] * shift[1] + src_dir[1];

    dst[0][0] = static_cast<float>(dst_w * 0.5);
    dst[0][1] = static_cast<float>(dst_h * 0.5);

    dst[1][0] = static_cast<float>(dst_w * 0.5 + dst_dir[0]);
    dst[1][1] = static_cast<float>(dst_h * 0.5 + dst_dir[1]);

    get_3rd_point(src);
    get_3rd_point(dst);

    cv::Point2f point_src[3] = {};
    cv::Point2f point_dst[3] = {};
    for (int i = 0; i < 3; i++) {
        point_src[i] = cv::Point2f(src[i][0], src[i][1]);
        point_dst[i] = cv::Point2f(dst[i][0], dst[i][1]);
    }
    if (inv) {
        return cv::getAffineTransform(point_dst, point_src);
    } else {
        return cv::getAffineTransform(point_src, point_dst);
    }
}

float sign(float x) {
    if (x >= 0)
        return 1.0f;
    else
        return -1.0f;
}

void postprocess_simplepose_one(bbox_keypoints &res, float *ptr, int hm_h, int hm_w, int channel, cv::Mat &trans_v) {
    float max_val;
    int c;
    double *data = (double *) trans_v.data;
    for (int i = 0; i < channel; i++) {
        max_val = -1e20;
        c = 0;
        for (int j = 0; j < hm_h; j++) {
            for (int k = 0; k < hm_w; k++) {
                if (max_val < ptr[i * hm_h * hm_w + c]) {
                    max_val = ptr[i * hm_h * hm_w + c];
                    res.keypoints[i][0] = (float) k;
                    res.keypoints[i][1] = (float) j;
                    res.keypoints[i][2] = max_val;
                }
                c++;
            }
        }
        int a = (int) res.keypoints[i][0];
        int b = (int) res.keypoints[i][1];

        if ((a > 1 && a < hm_w - 1) && (b > 1 && b < hm_w - 1)) {
            float temp = ptr[i * hm_h * hm_w + b * hm_h + a + 1] - ptr[i * hm_h * hm_w + b * hm_h + a - 1];
            res.keypoints[i][0] += sign(temp) * 0.25f;
            temp = ptr[i * hm_h * hm_w + (b + 1) * hm_w + a] - ptr[i * hm_h * hm_w + (b - 1) * hm_w + a];
            res.keypoints[i][1] += sign(temp) * 0.25f;
        }
        res.keypoints[i][0] = res.keypoints[i][0] * data[0] + res.keypoints[i][1] * data[1] + data[2];
        res.keypoints[i][1] = res.keypoints[i][0] * data[3] + res.keypoints[i][1] * data[4] + data[5];
    }
}

void postprocess_simplepose(std::vector<std::vector<bbox_keypoints>> &res, std::vector<float *> &output,
                            std::vector<std::vector<int>> &shapes, std::vector<cv::Mat> &trans_vs) {
    float *out = (float *) output[0];
    int batch = shapes[0][0];
    int keypoints = shapes[0][1];
    int hm_h = shapes[0][2];
    int hm_w = shapes[0][3];
    int row = 0;
    int col = 0;
    for (int i = 0; i < batch; i++, col++) {
        if (col == res[row].size()) {
            row++;
            col = 0;
        }
        postprocess_simplepose_one(res[row][col], &out[i * keypoints * hm_h * hm_w], hm_h, hm_w, keypoints,
                                   trans_vs[i]);
    }
}
