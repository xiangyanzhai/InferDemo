
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "include/postprocess_ocr.h"
#include "include/utility.h"
#include "include/dirent.h"
#include <io.h>
#include "onnx_infer.hpp"

class CRNN {
public:
    OnnxInfer *onnx_infer;
    int input_w;
    int input_h;
    std::vector<std::string> label_list_;

    CRNN(const wchar_t *onnx_file, int inp_w, int inp_h) {
        input_w = inp_w;
        input_h = inp_h;
        this->onnx_infer = new OnnxInfer(onnx_file);
        this->label_list_ = PaddleOCR::Utility::ReadDict(
                "D:\\PycharmProjects\\Demo\\horizon\\pp_ocr_v2_rec\\en_dict.txt");
        this->label_list_.insert(this->label_list_.begin(),
                                 "#"); // blank char for ctc
        this->label_list_.push_back(" ");
        std::cout << "aaaaaaaaaa:" << this->label_list_.size() << std::endl;
    }

    void preprocess(cv::Mat &blob, cv::Mat &frame, int input_w, int input_h) {
        blob = cv::dnn::blobFromImage(frame, 0.017352073251188378,
                                      cv::Size(input_w, input_h),
                                      cv::Scalar(127.5, 127.5, 127.5),
                                      false, false);
    }

    void infer(cv::Mat &frame,std::string &res,float &res_score) {
        cv::Mat blob;
        int frame_w = frame.cols;
        int frame_h = frame.rows;
        int new_w = ((float) frame_w / frame_h * input_h);

        new_w = min(new_w, input_w);

        cv::resize(frame, frame, cv::Size(new_w, input_h));

        cv::copyMakeBorder(frame, frame, 0, 0, 0,
                           int(input_w - new_w), cv::BORDER_CONSTANT,
                           {127,127,127});

        preprocess(blob, frame, input_w, input_h);
        this->onnx_infer->infer(blob);

        auto predict_batch = this->onnx_infer->outputs[0];
        auto predict_shape = this->onnx_infer->output_shapes[0];

        std::string str_res;
        int argmax_idx;
        int last_index = 0;
        float score = 0.f;
        int count = 0;
        float max_value = 0.0f;
        int m=0;
        for (int n = 0; n < predict_shape[1]; n++) {
            // get idx
            argmax_idx = int(PaddleOCR::Utility::argmax(
                    &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
                    &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]+11]));
            // get score
            max_value = float(*std::max_element(
                    &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
                    &predict_batch[(m * predict_shape[1] + n)*predict_shape[2]+11]));

            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                score += max_value;
                count += 1;
                str_res += label_list_[argmax_idx];
            }
            last_index = argmax_idx;
        }
        if (count==0){
            return ;
        }
        score /= count;

        res=str_res;
        res_score=score;
        std::cout<<"aaaaaaaaaaaaaaaaaaaaaaaa:"<<str_res.size()<<std::endl;

    }

    ~CRNN() {
        if (this->onnx_infer != nullptr) {
            delete this->onnx_infer;
        }
    }
};

bool endsWith(std::string const &str, std::string const &suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

int main() {
    const wchar_t *crnn_model_file = L"D:\\PycharmProjects\\PaddleOCR\\rec_v2.onnx";
    CRNN crnn(crnn_model_file, 64, 32);
    std::string img_dir = "D:\\datasets\\run_val\\image\\crop_img";
    std::vector<std::string> names;
    DIR *dir = opendir(img_dir.c_str());
    struct dirent *file;
    while ((file = readdir(dir)) != NULL) {
//        std::cout << file->d_name << std::endl;
        names.push_back(file->d_name);
    }
    std::cout << names.size() << std::endl;
    std::string S="";
    for (int i = 0; i < names.size(); i++) {
        if (!endsWith(names[i], "jpg")) continue;
        std::string img_file = img_dir + "\\" + names[i];

        cv::Mat img = cv::imread(img_file);
        std::string res="";
        float score=0.0f;
        crnn.infer(img,res,score);
        std::cout<<res<<"        "<<score<<std::endl;
        S=S+names[i]+" "+res+" "+std::to_string(score)+"\n";
    }
    std::ofstream  ofs;
    ofs.open("c_res.txt");
    ofs<<S.c_str();
    ofs.close();
    std::cout << "******************************" << std::endl;

}
