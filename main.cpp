#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include<vector>

using namespace std;

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

class YOLO {
public:
    std::string modelConfiguration;
    std::string modelWeights;
    int inpWidth;
    int inpHeight;
    bool equal_scale;
    cv::dnn::Net net;
public:
    YOLO(std::string modelConfiguration, std::string modelWeights, int inpWidth, int inpHeight, bool equal_scale) {

        this->modelConfiguration = modelConfiguration;
        this->modelWeights = modelWeights;
        this->inpWidth = inpWidth;
        this->inpHeight = inpHeight;
        this->equal_scale = equal_scale;
        this->net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

    }

    void infer(const cv::Mat &blob, std::vector<std::vector<bbox>> &res) {
        std::vector<cv::Mat> outs;
        clock_t startTime, endTime;
        startTime = clock();
        this->net.setInput(blob);
        this->net.forward(outs, getOutputsNames(this->net));
        endTime = clock();
        std::cout << "The infer time is: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
        startTime = clock();
        this->postprocess(res, outs);
        endTime = clock();
        std::cout << "The postprocess time is: " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
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

    void postprocess(std::vector<std::vector<bbox>> &res, const std::vector<cv::Mat> &outs, float conf_thresh = 0.3,
                     float nms_thresh = 0.45) {
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
            this->get_res(res[i], bboxes[i], temp, num_cls, conf_thresh, nms_thresh);
        }


    }

};

void yolo_cv_test() {
    std::string modelConfiguration = "C:\\Users\\admin\\Desktop\\yolov4\\yolov4.cfg";
    std::string modelWeights = "C:\\Users\\admin\\Desktop\\yolov4\\yolov4.weights";
    int inpWidth = 608;
    int inpHeight = 608;
    bool equal_scale = false;
    YOLO yolo(modelConfiguration, modelWeights, inpWidth, inpHeight, equal_scale);
    cv::Mat frame = cv::imread("D:\\PycharmProjects\\python_infer\\sample\\football_h.jpg");
    std::cout << frame.size << std::endl;

    vector<cv::Mat> frames;
    frames.push_back(cv::imread("D:\\PycharmProjects\\python_infer\\sample\\football_h.jpg"));
    frames.push_back(cv::imread("D:\\PycharmProjects\\python_infer\\sample\\football_v.jpg"));
    cv::Mat blob = cv::dnn::blobFromImages(frames, 1 / 255.0, cv::Size(608, 608), cv::Scalar(0, 0, 0), true, false);

    std::cout << blob.dims << std::endl;
    std::cout << blob.size[0] << std::endl;
    std::cout << blob.size[1] << std::endl;
    std::cout << blob.size[2] << std::endl;
    std::cout << blob.size[3] << std::endl;
    std::vector<std::vector<bbox>> res(blob.size[0]);
    yolo.infer(blob, res);
    string label;
    float x1, y1, w, h, score;
    int cls;
    for (int z = 0; z < blob.size[0]; z++) {
        frame = frames[z];
        std::cout << res[z].size() << std::endl;
        for (int i = 0; i < res[z].size(); i++) {
            x1 = res[z][i].x1 * frame.cols;
            y1 = res[z][i].y1 * frame.rows;
            w = res[z][i].w * frame.cols;
            h = res[z][i].h * frame.rows;
            score = res[z][i].score;
            cls = res[z][i].cls;
            cout << score << endl;

            cv::rectangle(frame, cv::Point((int) x1, (int) y1), cv::Point((int) (x1 + w), (int) (y1 + h)),
                          cv::Scalar(0, 255, 0), 2);
            label = "cls=" + std::to_string((int) cls) + "  score=" + std::to_string(score);
            cv::putText(frame, label, cv::Point((int) x1, (int) y1), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0),
                        0.5);

        }
        cv::imshow("img", frame);
        cv::waitKey(2000);
    }


    std::cout << "Hello, World!" << std::endl;
}

int main() {
    yolo_cv_test();
    return 0;
}