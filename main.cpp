#include <iostream>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>
#include<vector>
#include "process.hpp"
#include "onnx_infer.hpp"
#include "yolo_onnx.hpp"
#include "simplepose_onnx.hpp"
#include "light_pose.hpp"
#include "yolo_cv.hpp"

std::string f2s(float a) {
    std::stringstream buf;
    buf.precision(2);
    buf.setf(std::ios::fixed);
    buf << a;
    std::string str;
    str = buf.str();
    return str;
}

void yolo_test() {
    const wchar_t *onnx_file = L"D:\\model\\yolov4_-1_3_608_608_dynamic.onnx";
    YOLO yolo(onnx_file);
    std::vector<std::string> img_files;
    std::vector<cv::Mat> frames;
    cv::VideoCapture cap("D:\\sample_video\\IMG_6278.MP4");
    cv::Mat frame;
    std::vector<std::vector<bbox>> res;
    int input_w = 608;
    int input_h = 608;
    std::string label;
    float x1, y1, w, h, score;
    int cls;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frames.clear();
        frames.push_back(frame);
        yolo.infer(frames, res, input_w, input_h);
        for (int z = 0; z < frames.size(); z++) {
            frame = frames[z];
            for (int i = 0; i < res[z].size(); i++) {
                x1 = res[z][i].x1 * frame.cols;
                y1 = res[z][i].y1 * frame.rows;
                w = res[z][i].w * frame.cols;
                h = res[z][i].h * frame.rows;
                score = res[z][i].score;
                cls = res[z][i].cls;

                cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
                              cv::Scalar(0, 255, 0), 2);
                label = "cls=" + std::to_string((int) cls) + "  score=" + f2s(score);

                cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5,
                            cv::Scalar(255, 0, 0),
                            0.5);
            }
            cv::imshow("img", frame);
            cv::waitKey(20);
        }
    }

}

void yolo_fastest_test() {
    const wchar_t *onnx_file = L"D:\\model\\rope_yolo_fastest_608x384_210827\\yolo_fastest_rope.onnx";
    YOLOFastest yolo(onnx_file);
    std::vector<std::string> img_files;
    std::vector<cv::Mat> frames;
    cv::VideoCapture cap("D:\\sample_video\\0e828de703488a119e1fe4e2a4fa9fe6.mp4");
    cv::Mat frame;
    std::vector<std::vector<bbox>> res;
    int input_w = 608;
    int input_h = 384;
    std::string label;
    float x1, y1, w, h, score;
    int cls;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frames.clear();
        frames.push_back(frame);
        yolo.infer(frames, res, input_w, input_h);
        for (int z = 0; z < frames.size(); z++) {
            frame = frames[z];
            for (int i = 0; i < res[z].size(); i++) {
                x1 = res[z][i].x1 * frame.cols / input_w;
                y1 = res[z][i].y1 * frame.rows / input_h;
                w = res[z][i].w * frame.cols / input_w;
                h = res[z][i].h * frame.rows / input_h;
                score = res[z][i].score;
                cls = res[z][i].cls;

                cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
                              cv::Scalar(0, 255, 0), 2);
                label = "cls=" + std::to_string((int) cls) + "  score=" + f2s(score);
                cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5,
                            cv::Scalar(255, 0, 0),
                            0.5);
            }
            cv::imshow("img", frame);
            cv::waitKey(20);
        }
    }
}

void yolo_cv_test() {

    std::vector<std::string> img_files;
    std::vector<cv::Mat> frames;
    cv::VideoCapture cap("D:\\sample_video\\0e828de703488a119e1fe4e2a4fa9fe6.mp4");
    cv::Mat frame;
    std::vector<std::vector<bbox>> res;
//    std::string modelConfiguration = "D:\\model\\yolov4.cfg";
//    std::string modelWeights = "D:\\model\\yolov4.weights";
    std::string modelConfiguration = "D:\\model\\rope_yolo_fastest_608x384_210827\\yolo-fastest-rope.cfg";
    std::string modelWeights = "D:\\model\\rope_yolo_fastest_608x384_210827\\yolo-fastest-rope_best.weights";
    int input_w = 608;
    int input_h = 384;
    bool equal_scale = false;
    YOLO_cv yolo(modelConfiguration, modelWeights, input_w, input_h, equal_scale);
    std::string label;
    float x1, y1, w, h, score;
    int cls;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frames.clear();
        frames.push_back(frame);
        yolo.infer(frames, res, input_w, input_h);
        for (int z = 0; z < frames.size(); z++) {
            frame = frames[z];
            for (int i = 0; i < res[z].size(); i++) {
                x1 = res[z][i].x1 * frame.cols;
                y1 = res[z][i].y1 * frame.rows;
                w = res[z][i].w * frame.cols;
                h = res[z][i].h * frame.rows;
                score = res[z][i].score;
                cls = res[z][i].cls;

                cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
                              cv::Scalar(0, 255, 0), 2);
                label = "cls=" + std::to_string((int) cls) + "  score=" + f2s(score);
                cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5,
                            cv::Scalar(255, 0, 0),
                            0.5);
            }
            cv::imshow("img", frame);
            cv::waitKey(20);
        }
    }
}

void simplepose_test() {
    const wchar_t *onnx_file = L"D:\\PycharmProjects\\pytorch_YOLOv3\\test\\yolo_fastest_body.onnx";
    YOLOFastest yolo(onnx_file, "yolo_fastest_body");
    onnx_file = L"D:\\model\\effi_coco.onnx";
    SimplePose simplepose(onnx_file, "simplepose");
    std::vector<std::string> img_files;
    std::vector<cv::Mat> frames;
    cv::VideoCapture cap("D:\\sample_video\\01.13.17.755.flv");

    std::vector<std::vector<bbox>> res;

    int input_w = 320;
    int input_h = 320;
    int output_size[2] = {192, 256};
    int heatmap_size[2] = {48, 64};
    float shift[2] = {};
    std::string label;
    float x1, y1, ww, hh, score;
    int cls;
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frames.push_back(frame);
        if (frames.size() < 2) continue;
        yolo.infer(frames, res, input_w, input_h);
        std::vector<std::vector<bbox_keypoints>> res_k(frames.size());
        std::vector<cv::Mat> t_frames;
        std::vector<cv::Mat> trans_vs;
        for (int z = 0; z < frames.size(); z++) {
            frame = frames[z];
            for (int i = 0; i < res[z].size(); i++) {
                x1 = res[z][i].x1 * frame.cols / input_w;
                y1 = res[z][i].y1 * frame.rows / input_h;
                ww = res[z][i].w * frame.cols / input_w;
                hh = res[z][i].h * frame.rows / input_h;
                score = res[z][i].score;
                cls = res[z][i].cls;
                bbox_keypoints temp;
                temp.x1 = x1;
                temp.y1 = y1;
                temp.w = ww;
                temp.h = hh;
                temp.score = score;
                temp.cls = cls;
                memset(temp.keypoints, 0, sizeof(temp.keypoints));
                res_k[z].push_back(temp);
                float center[2] = {};
                float scale[2] = {};

                xywh2cs(x1, y1, ww, hh, center, scale);
                cv::Mat trans = get_affine_transform(center, scale, 0, output_size, shift, false);
                cv::Mat trans_v = get_affine_transform(center, scale, 0, heatmap_size, shift, true);

                trans_vs.push_back(trans_v);
                cv::Mat t_img;
                cv::warpAffine(frame, t_img, trans, cv::Size(192, 256));
                t_frames.push_back(t_img);


            }

        }
        simplepose.infer(t_frames, res_k, 192, 256, trans_vs);
        for (int i = 0; i < frames.size(); i++) {
            frame = frames[i];
            for (int j = 0; j < res_k[i].size(); j++) {
                x1 = res_k[i][j].x1;
                y1 = res_k[i][j].y1;
                ww = res_k[i][j].w;
                hh = res_k[i][j].h;
                cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + ww), (int) (y1 + hh)),
                              cv::Scalar(0, 255, 0), 2);
                for (int k = 0; k < 17; k++) {
                    cv::circle(frame, cv::Point((int) res_k[i][j].keypoints[k][0], (int) res_k[i][j].keypoints[k][1]),
                               2,
                               cv::Scalar(0, 255, 0), 2);
                }
            }
            cv::imshow("img", frame);
            cv::waitKey(20);
        }
        frames.clear();
    }
}

void light_pose_test() {
    const wchar_t *onnx_file = L"D:\\model\\human-pose-estimation_db.onnx";
    LightPose lightpose(onnx_file, "lightpose");
    std::vector<std::string> img_files;
    std::vector<cv::Mat> frames;
    cv::VideoCapture cap("D:\\sample_video\\01.13.17.755.flv");

    std::vector<std::vector<cv::Point2f>> res;
    int input_w = 456;
    int input_h = 256;
    std::string label;
    float x1, y1, w, h, score;
    int cls;
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frames.push_back(frame);
        if (frames.size() < 2) continue;
        lightpose.infer(frames, res, input_w, input_h);
        for (int z = 0; z < frames.size(); z++) {
            frame = frames[z];
            for (int i = 0; i < res[z].size(); i++) {
                x1 = res[z][i].x * frame.cols / input_w;
                y1 = res[z][i].y * frame.rows / input_h;
                cv::circle(frame, cv::Point((int) x1, (int) y1), 2, cv::Scalar(0, 255, 0), 2);

            }
            cv::imshow("img", frame);
            cv::waitKey(20);
        }
        frames.clear();
    }
}

int main() {
//    yolo_test();
//    yolo_fastest_test();
//    yolo_cv_test();
//    simplepose_test();
    light_pose_test();
    return 0;
}



//
//using namespace std;
//
//std::vector<std::string> getOutputsNames(const cv::dnn::Net &net) {
//    static std::vector<std::string> names;
//    if (names.empty()) {
//        //Get the indices of the output layers, i.e. the layers with unconnected outputs
//        std::vector<int> outLayers = net.getUnconnectedOutLayers();
//
//        //get the names of all the layers in the network
//        std::vector<std::string> layersNames = net.getLayerNames();
//
//        // Get the names of the output layers in names
//        names.resize(outLayers.size());
//        for (size_t i = 0; i < outLayers.size(); ++i)
//            names[i] = layersNames[outLayers[i] - 1];
//    }
//    return names;
//}
//
////struct bbox {
////    double x1;
////    double y1;
////    double w;
////    double h;
////    double score;
////    int cls;
////};
//
////bool my_sort(bbox &a, bbox &b) {
////    return a.score > b.score;
////}
//
//class YOLO {
//public:
//    std::string modelConfiguration;
//    std::string modelWeights;
//    int inpWidth;
//    int inpHeight;
//    bool equal_scale;
//    cv::dnn::Net net;
//public:
//    YOLO(std::string modelConfiguration, std::string modelWeights, int inpWidth, int inpHeight, bool equal_scale) {
//
//        this->modelConfiguration = modelConfiguration;
//        this->modelWeights = modelWeights;
//        this->inpWidth = inpWidth;
//        this->inpHeight = inpHeight;
//        this->equal_scale = equal_scale;
//        this->net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
//        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
////    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
//
//    }
//
//    void infer(const cv::Mat &blob, std::vector<std::vector<bbox>> &res) {
//        std::vector<cv::Mat> outs;
//        clock_t startTime, endTime;
//        startTime = clock();
//        this->net.setInput(blob);
//        this->net.forward(outs, getOutputsNames(this->net));
//        endTime = clock();
//        std::cout << "The infer time is: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//        startTime = clock();
//        this->postprocess(res, outs);
//        endTime = clock();
//        std::cout << "The postprocess time is: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//    }
//
//    void get_res(std::vector<bbox> &res, std::vector<cv::Rect2d> &bboxes, cv::Mat &temp, int num_cls,
//                 float conf_thresh = 0.3, float nms_thresh = 0.45) {
//        cv::Point classIdPoint;
//        double confidence;
//        for (int i = 0; i < num_cls; i++) {
//            std::vector<int> indices;
//            cv::dnn::NMSBoxes(bboxes, temp.colRange(i, i + 1), 0.005, nms_thresh, indices);
//            std::sort(indices.begin(), indices.end());
//            for (int j = 0, k = 0; j < temp.rows; j++) {
//                if (k < indices.size()) {
//                    if (j != indices[k]) {
//                        temp.at<float>(j, i) = 0.0f;
//                    } else {
//                        k++;
//                    }
//                } else {
//                    temp.at<float>(j, i) = 0.0f;
//                }
//
//
//            }
//        }
//
//
//        for (int i = 0; i < temp.rows; i++) {
//            minMaxLoc(temp.row(i), 0, &confidence, 0, &classIdPoint);
//
//            if (confidence < conf_thresh) continue;
//
//            res.push_back({bboxes[i].x, bboxes[i].y, bboxes[i].width, bboxes[i].height, confidence, classIdPoint.x});
//        }
//        std::sort(res.begin(), res.end(), my_sort);
//
//
//    }
//
//    void postprocess(std::vector<std::vector<bbox>> &res, const std::vector<cv::Mat> &outs, float conf_thresh = 0.3,
//                     float nms_thresh = 0.45) {
//        for (int i = 0; i < res.size(); i++) res[i].clear();
//        int batch, num_cls;
//        if (outs[0].dims == 3) {
//            batch = outs[0].size[0];
//            num_cls = outs[0].size[2] - 5;
//        } else {
//            batch = 1;
//            num_cls = outs[0].size[1] - 5;
//        }
//
//        std::vector<cv::Mat> temp_batch(batch);
//        std::vector<std::vector<cv::Rect2d>> bboxes(batch);
//        std::vector<std::vector<cv::Mat>> scores(batch);
//        int n, stride;
//        float x, y, w, h;
//        cv::Mat score;
//        for (auto out: outs) {
//            if (out.dims == 3) {
//                n = out.size[1];
//                stride = out.size[1] * out.size[2];
//            } else {
//                n = out.size[0];
//                stride = out.size[0] * out.size[1];
//            }
//            float *data = (float *) out.data;
//            for (int i = 0; i < batch; i++) {
//                for (int j = 0; j < n; j++) {
//                    if (data[i * stride + j * (num_cls + 5) + 4] < conf_thresh) continue;
//                    x = data[i * stride + j * (num_cls + 5) + 0];
//                    y = data[i * stride + j * (num_cls + 5) + 1];
//                    w = data[i * stride + j * (num_cls + 5) + 2];
//                    h = data[i * stride + j * (num_cls + 5) + 3];
//                    bboxes[i].push_back(cv::Rect2d(x - w / 2, y - h / 2, w, h));
//                    score = cv::Mat(1, num_cls, CV_32F);
//                    for (int k = 0; k < num_cls; k++) {
//                        score.at<float>(0, k) = data[i * stride + j * (num_cls + 5) + 5 + k];
//                    }
//                    scores[i].push_back(score);
//                }
//            }
//
//        }
//        cv::Mat temp;
//        for (int i = 0; i < batch; i++) {
//            if (bboxes[i].empty()) continue;
//            cv::vconcat(scores[i], temp);
//            this->get_res(res[i], bboxes[i], temp, num_cls, conf_thresh, nms_thresh);
//        }
//
//
//    }
//
//};
//
//void get_res(std::vector<bbox> &res, std::vector<cv::Rect2d> &bboxes, cv::Mat &temp, int num_cls,
//             float conf_thresh = 0.3, float nms_thresh = 0.45) {
//    cv::Point classIdPoint;
//    double confidence;
//    for (int i = 0; i < num_cls; i++) {
//        std::vector<int> indices;
//        cv::dnn::NMSBoxes(bboxes, temp.colRange(i, i + 1), 0.005, nms_thresh, indices);
//        std::sort(indices.begin(), indices.end());
//        for (int j = 0, k = 0; j < temp.rows; j++) {
//            if (k < indices.size()) {
//                if (j != indices[k]) {
//                    temp.at<float>(j, i) = 0.0f;
//                } else {
//                    k++;
//                }
//            } else {
//                temp.at<float>(j, i) = 0.0f;
//            }
//
//
//        }
//    }
//
//
//    for (int i = 0; i < temp.rows; i++) {
//        minMaxLoc(temp.row(i), 0, &confidence, 0, &classIdPoint);
//
//        if (confidence < conf_thresh) continue;
//
//        res.push_back({bboxes[i].x, bboxes[i].y, bboxes[i].width, bboxes[i].height, confidence, classIdPoint.x});
//    }
//    std::sort(res.begin(), res.end(), my_sort);
//
//
//}
//
//void
//postprocess(std::vector<std::vector<bbox>> &res, std::vector<float *> &output, std::vector<std::vector<int>> &shapes,
//            float conf_thresh = 0.3,
//            float nms_thresh = 0.45) {
//    for (int i = 0; i < res.size(); i++) res[i].clear();
//    float *boxes = output[0];
//    float *confs = output[1];
//
//    int batch = shapes[1][0];
//    int num_cls = shapes[1][2];
//    float max_value = -1e20;
//    float conf_temp;
//    cv::Mat temp;
//    std::vector<cv::Mat> temp_batch(batch);
//    std::vector<std::vector<cv::Rect2d>> bboxes(batch);
//    std::vector<std::vector<cv::Mat>> scores(batch);
//
//    int stride1 = shapes[0][1] * shapes[0][3];
//    int stride2 = shapes[1][1] * shapes[1][2];
//    for (int i = 0; i < batch; i++) {
//        for (int j = 0; j < shapes[0][1]; j++) {
//            max_value = -1e20;
//            for (int k = 0; k < num_cls; k++) {
//                conf_temp = confs[i * stride2 + j * num_cls + k];
//                if (max_value < conf_temp) {
//                    max_value = conf_temp;
//                }
//            }
//            if (max_value < conf_thresh) {
//                continue;
//            }
//            bboxes[i].push_back(cv::Rect2d(boxes[i * stride1 + j * 4 + 0], boxes[i * stride1 + j * 4 + 1],
//                                           boxes[i * stride1 + j * 4 + 2] - boxes[i * stride1 + j * 4 + 0],
//                                           boxes[i * stride1 + j * 4 + 3] - boxes[i * stride1 + j * 4 + 1]));
//            temp = cv::Mat(1, num_cls, CV_32F);
//            for (int k = 0; k < num_cls; k++) {
//                temp.at<float>(0, k) = confs[i * stride2 + j * num_cls + k];
//            }
//            scores[i].push_back(temp);
//
//        }
//    }
//    for (int i = 0; i < batch; i++) {
//        if (bboxes[i].empty()) continue;
//        cv::vconcat(scores[i], temp);
//        get_res(res[i], bboxes[i], temp, num_cls, conf_thresh, nms_thresh);
//    }
//}
//
//void
//postprocess_fastest(std::vector<std::vector<bbox>> &res, std::vector<float *> &output,
//                    std::vector<std::vector<int>> &shapes, const float &input_w,
//                    const float &input_h, float conf_thresh = 0.3, float nms_thresh = 0.45) {
//    for (int i = 0; i < res.size(); i++) res[i].clear();
//    float *out = output[0];
//    int batch = shapes[0][0];
//    int num_cls = shapes[0][2] - 5;
//    float conf_temp;
//    int stride = shapes[0][1] * shapes[0][2];
//    cv::Mat temp;
//    std::vector<cv::Mat> temp_batch(batch);
//    std::vector<std::vector<cv::Rect2d>> bboxes(batch);
//    std::vector<std::vector<cv::Mat>> scores(batch);
//    float x1, y1, x2, y2;
//
//    for (int i = 0; i < batch; i++) {
//        for (int j = 0; j < shapes[0][1]; j++) {
//            conf_temp = out[i * stride + j * (num_cls + 5) + 4];
//            if (conf_temp < conf_thresh) {
//                continue;
//            }
//            x1 = out[i * stride + j * (num_cls + 5) + 0];
//            y1 = out[i * stride + j * (num_cls + 5) + 1];
//            x2 = out[i * stride + j * (num_cls + 5) + 2];
//            y2 = out[i * stride + j * (num_cls + 5) + 3];
//            if (x1 < 0) x1 = 0.0f;
//            if (x1 > input_w) x1 = input_w;
//            if (x2 < 0) x2 = 0.0f;
//            if (x2 > input_w) x2 = input_w;
//            if (y1 < 0) y1 = 0.0f;
//            if (y1 > input_h) y1 = input_h;
//            if (y2 < 0) y2 = 0.0f;
//            if (y2 > input_h) y2 = input_h;
//
//            bboxes[i].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
//            temp = cv::Mat(1, num_cls, CV_32F);
//            for (int k = 0; k < num_cls; k++) {
//                temp.at<float>(0, k) = out[i * stride + j * (num_cls + 5) + 5 + k] * conf_temp;
//            }
//            scores[i].push_back(temp);
//
//        }
//    }
//    for (int i = 0; i < batch; i++) {
//        if (bboxes[i].empty()) continue;
//        cv::vconcat(scores[i], temp);
//        get_res(res[i], bboxes[i], temp, num_cls, conf_thresh, nms_thresh);
//    }
//}
//
//void yolo_cv_test() {
//    std::string modelConfiguration = "D:\\model\\yolov4.cfg";
//    std::string modelWeights = "D:\\model\\yolov4.weights";
//    int inpWidth = 608;
//    int inpHeight = 608;
//    bool equal_scale = false;
//    YOLO yolo(modelConfiguration, modelWeights, inpWidth, inpHeight, equal_scale);
//    cv::Mat frame = cv::imread("D:\\PycharmProjects\\python_infer\\sample\\football_h.jpg");
//    std::cout << frame.size << std::endl;
//
//    vector<cv::Mat> frames;
//    frames.push_back(cv::imread("D:\\PycharmProjects\\python_infer\\sample\\football_h.jpg"));
//    frames.push_back(cv::imread("D:\\PycharmProjects\\python_infer\\sample\\football_v.jpg"));
//    cv::Mat blob = cv::dnn::blobFromImages(frames, 1 / 255.0, cv::Size(608, 608), cv::Scalar(0, 0, 0), true, false);
//
//    std::cout << blob.dims << std::endl;
//    std::cout << blob.size[0] << std::endl;
//    std::cout << blob.size[1] << std::endl;
//    std::cout << blob.size[2] << std::endl;
//    std::cout << blob.size[3] << std::endl;
//    std::vector<std::vector<bbox>> res(blob.size[0]);
//    yolo.infer(blob, res);
//    string label;
//    float x1, y1, w, h, score;
//    int cls;
//    for (int z = 0; z < blob.size[0]; z++) {
//        frame = frames[z];
//        std::cout << res[z].size() << std::endl;
//        for (int i = 0; i < res[z].size(); i++) {
//            x1 = res[z][i].x1 * frame.cols;
//            y1 = res[z][i].y1 * frame.rows;
//            w = res[z][i].w * frame.cols;
//            h = res[z][i].h * frame.rows;
//            score = res[z][i].score;
//            cls = res[z][i].cls;
//            cout << score << endl;
//
//            cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
//                          cv::Scalar(0, 255, 0), 2);
//            label = "cls=" + std::to_string((int) cls) + "  score=" + std::to_string(score);
//            cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0),
//                        0.5);
//
//        }
//        cv::imshow("img", frame);
//        cv::waitKey(2000);
//    }
//
//
//    std::cout << "Hello, World!" << std::endl;
//}
//
//struct OnnxInfo {
//    const wchar_t *onnx_file;
//    std::vector<const char *> input_node_names;
//    std::vector<const char *> output_node_names;
//
//};
//
//
//void yolo_onnx_test(const wchar_t *onnx_file, std::vector<string> img_files, float conf_thresh = 0.3,
//                    float nms_thresh = 0.45) {
//    clock_t startTime, endTime;
//    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//    Ort::SessionOptions session_options;
//    session_options.SetIntraOpNumThreads(1);
//    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//    Ort::Session session(env, onnx_file, session_options);
//    Ort::AllocatorWithDefaultOptions allocator;
//    auto mask_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//
//    OnnxInfo onnx_info;
//    onnx_info.onnx_file = onnx_file;
//    int input_w, input_h;
//    input_w = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2];
//    input_h = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[3];
//    for (int i = 0; i < session.GetInputCount(); i++) {
////        std::cout<<session.GetInputName(i,allocator)<<std::endl;
//        onnx_info.input_node_names.push_back(session.GetInputName(i, allocator));
////        for (int j = 0; j < session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetDimensionsCount(); j++) {
////            std::cout<<session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()[j]<<"  ";
////        }
////        std::cout<<std::endl;
//
////        std::cout<< typeid(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape().data()).name()  <<std::endl;
//
//
//    }
//    for (int i = 0; i < session.GetOutputCount(); i++) {
//        onnx_info.output_node_names.push_back(session.GetOutputName(i, allocator));
//    }
//
//
//    std::vector<cv::Mat> frames;
//    for (int i = 0; i < img_files.size(); i++) {
//        frames.push_back(cv::imread(img_files[i]));
//    }
//    cv::Mat blob = cv::dnn::blobFromImages(frames, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true,
//                                           false);
//
//    std::vector<std::vector<bbox>> res(blob.size[0]);
//
//    int num = 1;
//    std::vector<int64_t> b;
//    for (int i = 0; i < blob.dims; i++) {
//        num *= blob.size[i];
//        b.push_back(blob.size[i]);
//    }
//
//    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mask_memory_info, (float *) blob.data, num,
//                                                              b.data(), blob.dims);
//    std::vector<Ort::Value> ort_inputs;
//    ort_inputs.push_back(std::move(input_tensor));
//
//    startTime = clock();
//    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, onnx_info.input_node_names.data(), ort_inputs.data(),
//                                      ort_inputs.size(), onnx_info.output_node_names.data(),
//                                      onnx_info.output_node_names.size());
//    endTime = clock();
//    std::cout << "yolo onnx infer time:" << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//    std::vector<std::vector<int>> shapes(output_tensors.size());
//    std::vector<float *> outputs(output_tensors.size());
//    for (int i = 0; i < output_tensors.size(); i++) {
//
//        outputs[i] = (float *) output_tensors[i].GetTensorMutableData<float>();
//        for (int j = 0; j < output_tensors[i].GetTensorTypeAndShapeInfo().GetDimensionsCount(); j++) {
//            shapes[i].push_back((int) output_tensors[i].GetTensorTypeAndShapeInfo().GetShape()[j]);
//        }
//
//    }
//    startTime = clock();
//    postprocess(res, outputs, shapes);
//    endTime = clock();
//    std::cout << "yolo onnx postprocess time:" << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//    string label;
//    float x1, y1, w, h, score;
//    int cls;
//    cv::Mat frame;
//    for (int z = 0; z < blob.size[0]; z++) {
//        frame = frames[z];
//        std::cout << res[z].size() << std::endl;
//        for (int i = 0; i < res[z].size(); i++) {
//            x1 = res[z][i].x1 * frame.cols;
//            y1 = res[z][i].y1 * frame.rows;
//            w = res[z][i].w * frame.cols;
//            h = res[z][i].h * frame.rows;
//            score = res[z][i].score;
//            cls = res[z][i].cls;
//            cout << score << endl;
//
//            cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
//                          cv::Scalar(0, 255, 0), 2);
//            label = "cls=" + std::to_string((int) cls) + "  score=" + std::to_string(score);
//            cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0),
//                        0.5);
//
//        }
//        cv::imshow("img", frame);
//        cv::waitKey(2000);
//    }
//
//
//}
//
//void yolo_fastest_onnx_test(const wchar_t *onnx_file, std::vector<string> img_files, int input_w, int input_h,
//                            float conf_thresh = 0.3, float nms_thresh = 0.45) {
//    clock_t startTime, endTime;
//    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//    Ort::SessionOptions session_options;
//    session_options.SetIntraOpNumThreads(1);
//    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//    Ort::Session session(env, onnx_file, session_options);
//    Ort::AllocatorWithDefaultOptions allocator;
//    auto mask_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//
//    OnnxInfo onnx_info;
//    onnx_info.onnx_file = onnx_file;
//
//
//    for (int i = 0; i < session.GetInputCount(); i++) {
////        std::cout<<session.GetInputName(i,allocator)<<std::endl;
//        onnx_info.input_node_names.push_back(session.GetInputName(i, allocator));
////        for (int j = 0; j < session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetDimensionsCount(); j++) {
////            std::cout<<session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()[j]<<"  ";
////        }
////        std::cout<<std::endl;
//
////        std::cout<< typeid(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape().data()).name()  <<std::endl;
//
//
//    }
//    for (int i = 0; i < session.GetOutputCount(); i++) {
//        onnx_info.output_node_names.push_back(session.GetOutputName(i, allocator));
//    }
//
//
//    std::vector<cv::Mat> frames;
//    for (int i = 0; i < img_files.size(); i++) {
//        frames.push_back(cv::imread(img_files[i]));
//    }
//    cv::Mat blob = cv::dnn::blobFromImages(frames, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true,
//                                           false);
//
//    std::vector<std::vector<bbox>> res(blob.size[0]);
//
//    int num = 1;
//    std::vector<int64_t> b;
//    for (int i = 0; i < blob.dims; i++) {
//        num *= blob.size[i];
//        b.push_back(blob.size[i]);
//    }
//
//    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mask_memory_info, (float *) blob.data, num,
//                                                              b.data(), blob.dims);
//    std::vector<Ort::Value> ort_inputs;
//    ort_inputs.push_back(std::move(input_tensor));
//
//    startTime = clock();
//    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, onnx_info.input_node_names.data(), ort_inputs.data(),
//                                      ort_inputs.size(), onnx_info.output_node_names.data(),
//                                      onnx_info.output_node_names.size());
//    endTime = clock();
//    std::cout << "yolo onnx infer time:" << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//    std::vector<std::vector<int>> shapes(output_tensors.size());
//    std::vector<float *> outputs(output_tensors.size());
//    for (int i = 0; i < output_tensors.size(); i++) {
//
//        outputs[i] = (float *) output_tensors[i].GetTensorMutableData<float>();
//        for (int j = 0; j < output_tensors[i].GetTensorTypeAndShapeInfo().GetDimensionsCount(); j++) {
//            shapes[i].push_back((int) output_tensors[i].GetTensorTypeAndShapeInfo().GetShape()[j]);
//        }
//
//    }
//    startTime = clock();
//    postprocess_fastest(res, outputs, shapes, (float) input_w, (float) input_h);
//    endTime = clock();
//    std::cout << "yolo onnx postprocess time:" << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//
//    string label;
//    float x1, y1, w, h, score;
//    int cls;
//    cv::Mat frame;
//    for (int z = 0; z < blob.size[0]; z++) {
//        frame = frames[z];
//
//        for (int i = 0; i < res[z].size(); i++) {
//            x1 = res[z][i].x1 * frame.cols / input_w;
//            y1 = res[z][i].y1 * frame.rows / input_h;
//            w = res[z][i].w * frame.cols / input_w;
//            h = res[z][i].h * frame.rows / input_h;
//            score = res[z][i].score;
//            cls = res[z][i].cls;
//
//            cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
//                          cv::Scalar(0, 255, 0), 2);
//            label = "cls=" + std::to_string((int) cls) + "  score=" + std::to_string(score);
//            cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5,
//                        cv::Scalar(255, 0, 0),
//                        0.5);
//
//        }
//        cv::imshow("img", frame);
//        cv::waitKey(2000);
//
//    }
//}
//
//void yolo_fastest_video() {
//
//    const wchar_t *onnx_file = L"D:\\model\\yolov4_-1_3_608_608_dynamic.onnx";
//    onnx_file = L"D:\\model\\rope_yolo_fastest_608x384_210827\\yolo_fastest_rope.onnx";
//    clock_t startTime, endTime;
//    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//    Ort::SessionOptions session_options;
//    session_options.SetIntraOpNumThreads(1);
//    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//    Ort::Session session(env, onnx_file, session_options);
//    Ort::AllocatorWithDefaultOptions allocator;
//    auto mask_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//
//    OnnxInfo onnx_info;
//    onnx_info.onnx_file = onnx_file;
//
//
//    for (int i = 0; i < session.GetInputCount(); i++) {
//        onnx_info.input_node_names.push_back(session.GetInputName(i, allocator));
//    }
//    for (int i = 0; i < session.GetOutputCount(); i++) {
//        onnx_info.output_node_names.push_back(session.GetOutputName(i, allocator));
//    }
//
//
//    int input_w = 608, input_h = 384;
//
//    cv::Mat blob;
//
//    cv::VideoCapture cap("D:\\sample_video\\0e828de703488a119e1fe4e2a4fa9fe6.mp4");
//    cv::Mat frame;
//
//    std::vector<std::vector<bbox>> res(1);
//    while (true) {
//        cap >> frame;
//        if (frame.empty()) break;
//        std::cout << frame.size << std::endl;
//        blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true,
//                                      false);
//        std::cout << blob.size << std::endl;
//        int num = 1;
//        std::vector<int64_t> b;
//        for (int i = 0; i < blob.dims; i++) {
//            num *= blob.size[i];
//            b.push_back(blob.size[i]);
//        }
//        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mask_memory_info, (float *) blob.data, num,
//                                                                  b.data(), blob.dims);
//        std::vector<Ort::Value> ort_inputs;
//        ort_inputs.push_back(std::move(input_tensor));
//
//        startTime = clock();
//
//        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, onnx_info.input_node_names.data(),
//                                          ort_inputs.data(),
//                                          ort_inputs.size(), onnx_info.output_node_names.data(),
//                                          onnx_info.output_node_names.size());
//        endTime = clock();
//        std::cout << "yolo onnx infer time:" << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//        std::vector<std::vector<int>> shapes(output_tensors.size());
//        std::vector<float *> outputs(output_tensors.size());
//        for (int i = 0; i < output_tensors.size(); i++) {
//
//            outputs[i] = (float *) output_tensors[i].GetTensorMutableData<float>();
//            for (int j = 0; j < output_tensors[i].GetTensorTypeAndShapeInfo().GetDimensionsCount(); j++) {
//                shapes[i].push_back((int) output_tensors[i].GetTensorTypeAndShapeInfo().GetShape()[j]);
//            }
//
//        }
//        startTime = clock();
//
//        postprocess_fastest(res, outputs, shapes, (float) input_w, (float) input_h);
//        endTime = clock();
//        std::cout << "yolo onnx postprocess time:" << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s"
//                  << std::endl;
//
//        string label;
//        float x1, y1, w, h, score;
//        int cls;
//
//        int z = 0;
//        for (int i = 0; i < res[z].size(); i++) {
//            x1 = res[z][i].x1 * frame.cols / input_w;
//            y1 = res[z][i].y1 * frame.rows / input_h;
//            w = res[z][i].w * frame.cols / input_w;
//            h = res[z][i].h * frame.rows / input_h;
//            score = res[z][i].score;
//            cls = res[z][i].cls;
//            cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
//                          cv::Scalar(0, 255, 0), 2);
//            label = "cls=" + std::to_string((int) cls) + "  score=" + std::to_string(score);
//            cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5,
//                        cv::Scalar(255, 0, 0),
//                        0.5);
//
//
//        }
//
//        cv::imshow("img", frame);
//        cv::waitKey(20);
//    }
//}
//
//void simplepose_video() {
//
//    const wchar_t *onnx_file = L"D:\\model\\yolov4_-1_3_608_608_dynamic.onnx";
//    onnx_file = L"D:\\model\\rope_yolo_fastest_608x384_210827\\yolo_fastest_rope.onnx";
//    clock_t startTime, endTime;
//    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//    Ort::SessionOptions session_options;
//    session_options.SetIntraOpNumThreads(1);
//    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//    Ort::Session session(env, onnx_file, session_options);
//    Ort::AllocatorWithDefaultOptions allocator;
//    auto mask_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//
//    OnnxInfo onnx_info;
//    onnx_info.onnx_file = onnx_file;
//
//
//    for (int i = 0; i < session.GetInputCount(); i++) {
//        onnx_info.input_node_names.push_back(session.GetInputName(i, allocator));
//    }
//    for (int i = 0; i < session.GetOutputCount(); i++) {
//        onnx_info.output_node_names.push_back(session.GetOutputName(i, allocator));
//    }
//
//
//    int input_w = 608, input_h = 384;
//
//    cv::Mat blob;
//
//    cv::VideoCapture cap("D:\\sample_video\\0e828de703488a119e1fe4e2a4fa9fe6.mp4");
//    cv::Mat frame;
//
//    std::vector<std::vector<bbox>> res(1);
//    while (true) {
//        cap >> frame;
//        if (frame.empty()) break;
//        std::cout << frame.size << std::endl;
//        blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true,
//                                      false);
//        std::cout << blob.size << std::endl;
//        int num = 1;
//        std::vector<int64_t> b;
//        for (int i = 0; i < blob.dims; i++) {
//            num *= blob.size[i];
//            b.push_back(blob.size[i]);
//        }
//        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mask_memory_info, (float *) blob.data, num,
//                                                                  b.data(), blob.dims);
//        std::vector<Ort::Value> ort_inputs;
//        ort_inputs.push_back(std::move(input_tensor));
//
//        startTime = clock();
//
//        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, onnx_info.input_node_names.data(),
//                                          ort_inputs.data(),
//                                          ort_inputs.size(), onnx_info.output_node_names.data(),
//                                          onnx_info.output_node_names.size());
//        endTime = clock();
//        std::cout << "yolo onnx infer time:" << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
//        std::vector<std::vector<int>> shapes(output_tensors.size());
//        std::vector<float *> outputs(output_tensors.size());
//        for (int i = 0; i < output_tensors.size(); i++) {
//
//            outputs[i] = (float *) output_tensors[i].GetTensorMutableData<float>();
//            for (int j = 0; j < output_tensors[i].GetTensorTypeAndShapeInfo().GetDimensionsCount(); j++) {
//                shapes[i].push_back((int) output_tensors[i].GetTensorTypeAndShapeInfo().GetShape()[j]);
//            }
//
//        }
//        startTime = clock();
//
//        postprocess_fastest(res, outputs, shapes, (float) input_w, (float) input_h);
//        endTime = clock();
//        std::cout << "yolo onnx postprocess time:" << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s"
//                  << std::endl;
//
//        string label;
//        float x1, y1, w, h, score;
//        int cls;
//
//        int z = 0;
//        for (int i = 0; i < res[z].size(); i++) {
//            x1 = res[z][i].x1 * frame.cols / input_w;
//            y1 = res[z][i].y1 * frame.rows / input_h;
//            w = res[z][i].w * frame.cols / input_w;
//            h = res[z][i].h * frame.rows / input_h;
//            score = res[z][i].score;
//            cls = res[z][i].cls;
//            cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
//                          cv::Scalar(0, 255, 0), 2);
//            label = "cls=" + std::to_string((int) cls) + "  score=" + std::to_string(score);
//            cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5,
//                        cv::Scalar(255, 0, 0),
//                        0.5);
//
//
//        }
//
//        cv::imshow("img", frame);
//        cv::waitKey(20);
//    }
//}
//
//int main() {
////    yolo_fastest_video();
//    int input_w = 608;
//    int input_h = 384;
//    OnnxInfer rope(L"D:\\model\\rope_yolo_fastest_608x384_210827\\yolo_fastest_rope.onnx");
////    OnnxInfer rope(L"D:\\model\\yolov4_-1_3_608_608_dynamic.onnx");
//
//    cv::Mat frame = cv::imread("D:\\PycharmProjects\\python_infer\\sample\\rope_h.jpg");
//    cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true,
//                                          false);
//    rope.infer(blob);
//
//    for (int i = 0; i < rope.output_shapes.size(); i++) {
//        for (int j = 0; j < rope.output_shapes[i].size(); j++) {
//            std::cout << rope.output_shapes[i][j] << " ";
//        }
//    }
//    std::vector<std::vector<bbox>> res(1);
//    postprocess_fastest(res, rope.outputs, rope.output_shapes, (float) input_w, (float) input_h);
//    std::cout<<"aaaaa end aaaaaa"<<std::endl;
//    string label;
//    float x1, y1, w, h, score;
//    int cls;
//    int z=0;
//    for (int i = 0; i < res[z].size(); i++) {
//        x1 = res[z][i].x1 * frame.cols / input_w;
//        y1 = res[z][i].y1 * frame.rows / input_h;
//        w = res[z][i].w * frame.cols / input_w;
//        h = res[z][i].h * frame.rows / input_h;
//        score = res[z][i].score;
//        cls = res[z][i].cls;
//        cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
//                      cv::Scalar(0, 255, 0), 2);
//        label = "cls=" + std::to_string((int) cls) + "  score=" + std::to_string(score);
//        cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5,
//                    cv::Scalar(255, 0, 0),
//                    0.5);
//
//
//    }
////
//    cv::imshow("img", frame);
//    cv::waitKey(200000);
//
//
//
////    const wchar_t *onnx_file = L"D:\\model\\yolov4_-1_3_608_608_dynamic.onnx";
//////    onnx_file = L"D:\\model\\rope_yolo_fastest_608x384_210827\\yolo_fastest_rope.onnx";
////    std::vector<string> img_files;
//////    img_files.push_back("D:\\PycharmProjects\\python_infer\\sample\\football_v.jpg");
//////    img_files.push_back("D:\\PycharmProjects\\python_infer\\sample\\football_v.jpg");
////    img_files.push_back("D:\\PycharmProjects\\python_infer\\sample\\rope_h.jpg");
//////    img_files.push_back("D:\\PycharmProjects\\python_infer\\sample\\football_h.jpg");
////    int input_w = 608;
////    int input_h = 384;
////    yolo_cv_test();
//////    yolo_onnx_test(onnx_file, img_files);
//////    yolo_fastest_onnx_test(onnx_file, img_files, input_w, input_h);
//////    yolo_fastest_video();
//    return 0;
//}