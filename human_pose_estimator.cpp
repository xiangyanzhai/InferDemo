// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>


#include "human_pose_estimator.hpp"
#include "peak.hpp"

namespace human_pose_estimation {
const size_t HumanPoseEstimator::keypointsNumber = 18;

HumanPoseEstimator::HumanPoseEstimator(bool enablePerformanceReport)
{
pad = cv::Vec4i::all(0);
stride = 8;

upsampleRatio = 4;

minPeaksDistance = 3.0f;

midPointsScoreThreshold = 0.05f;

foundMidPointsRatioThreshold = 0.8f;

minJointsNumber = 3;

minSubsetScore = 0.2f;
}

std::vector<HumanPose> HumanPoseEstimator::estimate(const float* heatMapsData, const float* pafsData) {
    

    int keypointsNumber = 18;
    int heatMapDims2 = 32;
    int heatMapDims3 = 57;
    int pafsBlob1 = 38;
    cv::Size imageSize(256,456);

    std::vector<HumanPose> poses = postprocess(
        heatMapsData,
        heatMapDims2 * heatMapDims3,
        keypointsNumber,
        pafsData,
        heatMapDims2 * heatMapDims3,
        pafsBlob1,
        heatMapDims3, heatMapDims2, imageSize);
    return poses;
}

void HumanPoseEstimator::preprocess(const cv::Mat& image, float* buffer) const {
    cv::Mat resizedImage;
    double scale = inputLayerSize.height / static_cast<double>(image.rows);
    cv::resize(image, resizedImage, cv::Size(), scale, scale, cv::INTER_CUBIC);
    cv::Mat paddedImage;
    cv::copyMakeBorder(resizedImage, paddedImage, pad(0), pad(2), pad(1), pad(3),
                       cv::BORDER_CONSTANT, meanPixel);
    std::vector<cv::Mat> planes(3);
    cv::split(paddedImage, planes);
    for (size_t pId = 0; pId < planes.size(); pId++) {
        cv::Mat dst(inputLayerSize.height, inputLayerSize.width, CV_32FC1,
                    reinterpret_cast<void*>(
                        buffer + pId * inputLayerSize.area()));
        planes[pId].convertTo(dst, CV_32FC1);
    }
}

std::vector<HumanPose> HumanPoseEstimator::postprocess(
        const float* heatMapsData, const int heatMapOffset, const int nHeatMaps,
        const float* pafsData, const int pafOffset, const int nPafs,
        const int featureMapWidth, const int featureMapHeight,
        const cv::Size& imageSize) const {
    std::vector<cv::Mat> heatMaps(nHeatMaps);
    for (size_t i = 0; i < heatMaps.size(); i++) {
        heatMaps[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                              reinterpret_cast<void*>(
                                  const_cast<float*>(
                                      heatMapsData + i * heatMapOffset)));
    }
    
    resizeFeatureMaps(heatMaps);
    
    std::vector<cv::Mat> pafs(nPafs);
    for (size_t i = 0; i < pafs.size(); i++) {
        pafs[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                          reinterpret_cast<void*>(
                              const_cast<float*>(
                                  pafsData + i * pafOffset)));
    }
    resizeFeatureMaps(pafs);
    //printf("heatMaps.height = %d\n",heatMaps[0].size().height);
    //printf("heatMaps.width = %d\n",heatMaps[0].size().width);
    std::vector<HumanPose> poses = extractPoses(heatMaps, pafs);
    //correctCoordinates(poses, heatMaps[0].size(), imageSize);
    return poses;
}

class FindPeaksBody: public cv::ParallelLoopBody {
public:
    FindPeaksBody(const std::vector<cv::Mat>& heatMaps, float minPeaksDistance,
                  std::vector<std::vector<Peak> >& peaksFromHeatMap)
        : heatMaps(heatMaps),
          minPeaksDistance(minPeaksDistance),
          peaksFromHeatMap(peaksFromHeatMap) {}

    virtual void operator()(const cv::Range& range) const {
        for (int i = range.start; i < range.end; i++) {
            findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i);
        }
    }

private:
    const std::vector<cv::Mat>& heatMaps;
    float minPeaksDistance;
    std::vector<std::vector<Peak> >& peaksFromHeatMap;
};

std::vector<HumanPose> HumanPoseEstimator::extractPoses(
        const std::vector<cv::Mat>& heatMaps,
        const std::vector<cv::Mat>& pafs) const {
    std::vector<std::vector<Peak> > peaksFromHeatMap(heatMaps.size());
    FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())),
                      findPeaksBody);
    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
        peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
        for (auto& peak : peaksFromHeatMap[heatmapId]) {
            peak.id += peaksBefore;
        }
    }
    
    std::vector<HumanPose> poses = groupPeaksToPoses(
                peaksFromHeatMap, pafs, keypointsNumber, midPointsScoreThreshold,
                foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);
    return poses;
}

void HumanPoseEstimator::resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const {
	for (auto& featureMap : featureMaps) {
        cv::resize(featureMap, featureMap, cv::Size(),
                   upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
    }
}

void HumanPoseEstimator::correctCoordinates(std::vector<HumanPose>& poses,
                                            const cv::Size& featureMapsSize,
                                            const cv::Size& imageSize) const {
    CV_Assert(stride % upsampleRatio == 0);

    cv::Size fullFeatureMapSize = featureMapsSize * stride / upsampleRatio;

    float scaleX = imageSize.width /
            static_cast<float>(fullFeatureMapSize.width - pad(1) - pad(3));
    float scaleY = imageSize.height /
            static_cast<float>(fullFeatureMapSize.height - pad(0) - pad(2));
    for (auto& pose : poses) {
        for (auto& keypoint : pose.keypoints) {
            if (keypoint != cv::Point2f(-1, -1)) {
                keypoint.x *= stride / upsampleRatio;
                keypoint.x -= pad(1);
                keypoint.x *= scaleX;

                keypoint.y *= stride / upsampleRatio;
                keypoint.y -= pad(0);
                keypoint.y *= scaleY;
            }
        }
    }
}

bool HumanPoseEstimator::inputWidthIsChanged(const cv::Size& imageSize) {
    double scale = static_cast<double>(inputLayerSize.height) / static_cast<double>(imageSize.height);
    cv::Size scaledSize(static_cast<int>(cvRound(imageSize.width * scale)),
                        static_cast<int>(cvRound(imageSize.height * scale)));
    cv::Size scaledImageSize(std::max(scaledSize.width, inputLayerSize.height),
                             inputLayerSize.height);
    int minHeight = std::min(scaledImageSize.height, scaledSize.height);
    scaledImageSize.width = static_cast<int>(std::ceil(
                scaledImageSize.width / static_cast<float>(stride))) * stride;
    pad(0) = static_cast<int>(std::floor((scaledImageSize.height - minHeight) / 2.0));
    pad(1) = static_cast<int>(std::floor((scaledImageSize.width - scaledSize.width) / 2.0));
    pad(2) = scaledImageSize.height - minHeight - pad(0);
    pad(3) = scaledImageSize.width - scaledSize.width - pad(1);
    if (scaledSize.width == (inputLayerSize.width - pad(1) - pad(3))) {
        return false;
    }

    inputLayerSize.width = scaledImageSize.width;
    return true;
}

HumanPoseEstimator::~HumanPoseEstimator() {
    try {
        ;
    }
    catch (...) {
        ;
    }
}
}  // namespace human_pose_estimation


namespace human_pose_estimation_25 {
const size_t HumanPoseEstimator::keypointsNumber = 25;

HumanPoseEstimator::HumanPoseEstimator(bool enablePerformanceReport)
{
pad = cv::Vec4i::all(0);
stride = 8;

upsampleRatio = 4;

minPeaksDistance = 3.0f;

midPointsScoreThreshold = 0.05f;

foundMidPointsRatioThreshold = 0.8f;

minJointsNumber = 3;

minSubsetScore = 0.2f;
}

std::vector<HumanPose> HumanPoseEstimator::estimate(const float* heatMapsData, const float* pafsData) {


    int keypointsNumber = 25;
    int heatMapDims2 = 45;
    int heatMapDims3 = 80;
    int pafsBlob1 = 52;
    cv::Size imageSize(360,640);

    std::vector<HumanPose> poses = postprocess(
        heatMapsData,
        heatMapDims2 * heatMapDims3,
        keypointsNumber,
        pafsData,
        heatMapDims2 * heatMapDims3,
        pafsBlob1,
        heatMapDims3, heatMapDims2, imageSize);
    return poses;
}

void HumanPoseEstimator::preprocess(const cv::Mat& image, float* buffer) const {
    cv::Mat resizedImage;
    double scale = inputLayerSize.height / static_cast<double>(image.rows);
    cv::resize(image, resizedImage, cv::Size(), scale, scale, cv::INTER_CUBIC);
    cv::Mat paddedImage;
    cv::copyMakeBorder(resizedImage, paddedImage, pad(0), pad(2), pad(1), pad(3),
                       cv::BORDER_CONSTANT, meanPixel);
    std::vector<cv::Mat> planes(3);
    cv::split(paddedImage, planes);
    for (size_t pId = 0; pId < planes.size(); pId++) {
        cv::Mat dst(inputLayerSize.height, inputLayerSize.width, CV_32FC1,
                    reinterpret_cast<void*>(
                        buffer + pId * inputLayerSize.area()));
        planes[pId].convertTo(dst, CV_32FC1);
    }
}

std::vector<HumanPose> HumanPoseEstimator::postprocess(
        const float* heatMapsData, const int heatMapOffset, const int nHeatMaps,
        const float* pafsData, const int pafOffset, const int nPafs,
        const int featureMapWidth, const int featureMapHeight,
        const cv::Size& imageSize) const {
    std::vector<cv::Mat> heatMaps(nHeatMaps);
    for (size_t i = 0; i < heatMaps.size(); i++) {
        heatMaps[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                              reinterpret_cast<void*>(
                                  const_cast<float*>(
                                      heatMapsData + i * heatMapOffset)));
    }

    resizeFeatureMaps(heatMaps);

    std::vector<cv::Mat> pafs(nPafs);
    for (size_t i = 0; i < pafs.size(); i++) {
        pafs[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                          reinterpret_cast<void*>(
                              const_cast<float*>(
                                  pafsData + i * pafOffset)));
    }
    resizeFeatureMaps(pafs);
    //printf("heatMaps.height = %d\n",heatMaps[0].size().height);
    //printf("heatMaps.width = %d\n",heatMaps[0].size().width);
    std::vector<HumanPose> poses = extractPoses(heatMaps, pafs);
    //correctCoordinates(poses, heatMaps[0].size(), imageSize);
    return poses;
}

class FindPeaksBody: public cv::ParallelLoopBody {
public:
    FindPeaksBody(const std::vector<cv::Mat>& heatMaps, float minPeaksDistance,
                  std::vector<std::vector<Peak> >& peaksFromHeatMap)
        : heatMaps(heatMaps),
          minPeaksDistance(minPeaksDistance),
          peaksFromHeatMap(peaksFromHeatMap) {}

    virtual void operator()(const cv::Range& range) const {
        for (int i = range.start; i < range.end; i++) {
            findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i);
        }
    }

private:
    const std::vector<cv::Mat>& heatMaps;
    float minPeaksDistance;
    std::vector<std::vector<Peak> >& peaksFromHeatMap;
};

std::vector<HumanPose> HumanPoseEstimator::extractPoses(
        const std::vector<cv::Mat>& heatMaps,
        const std::vector<cv::Mat>& pafs) const {
    std::vector<std::vector<Peak> > peaksFromHeatMap(heatMaps.size());
    FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())),
                      findPeaksBody);
    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
        peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
        for (auto& peak : peaksFromHeatMap[heatmapId]) {
            peak.id += peaksBefore;
        }
    }

    std::vector<HumanPose> poses = groupPeaksToPoses(
                peaksFromHeatMap, pafs, keypointsNumber, midPointsScoreThreshold,
                foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);
    return poses;
}

void HumanPoseEstimator::resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const {
	for (auto& featureMap : featureMaps) {
        cv::resize(featureMap, featureMap, cv::Size(),
                   upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
    }
}

void HumanPoseEstimator::correctCoordinates(std::vector<HumanPose>& poses,
                                            const cv::Size& featureMapsSize,
                                            const cv::Size& imageSize) const {
    CV_Assert(stride % upsampleRatio == 0);

    cv::Size fullFeatureMapSize = featureMapsSize * stride / upsampleRatio;

    float scaleX = imageSize.width /
            static_cast<float>(fullFeatureMapSize.width - pad(1) - pad(3));
    float scaleY = imageSize.height /
            static_cast<float>(fullFeatureMapSize.height - pad(0) - pad(2));
    for (auto& pose : poses) {
        for (auto& keypoint : pose.keypoints) {
            if (keypoint != cv::Point2f(-1, -1)) {
                keypoint.x *= stride / upsampleRatio;
                keypoint.x -= pad(1);
                keypoint.x *= scaleX;

                keypoint.y *= stride / upsampleRatio;
                keypoint.y -= pad(0);
                keypoint.y *= scaleY;
            }
        }
    }
}

bool HumanPoseEstimator::inputWidthIsChanged(const cv::Size& imageSize) {
    double scale = static_cast<double>(inputLayerSize.height) / static_cast<double>(imageSize.height);
    cv::Size scaledSize(static_cast<int>(cvRound(imageSize.width * scale)),
                        static_cast<int>(cvRound(imageSize.height * scale)));
    cv::Size scaledImageSize(std::max(scaledSize.width, inputLayerSize.height),
                             inputLayerSize.height);
    int minHeight = std::min(scaledImageSize.height, scaledSize.height);
    scaledImageSize.width = static_cast<int>(std::ceil(
                scaledImageSize.width / static_cast<float>(stride))) * stride;
    pad(0) = static_cast<int>(std::floor((scaledImageSize.height - minHeight) / 2.0));
    pad(1) = static_cast<int>(std::floor((scaledImageSize.width - scaledSize.width) / 2.0));
    pad(2) = scaledImageSize.height - minHeight - pad(0);
    pad(3) = scaledImageSize.width - scaledSize.width - pad(1);
    if (scaledSize.width == (inputLayerSize.width - pad(1) - pad(3))) {
        return false;
    }

    inputLayerSize.width = scaledImageSize.width;
    return true;
}

HumanPoseEstimator::~HumanPoseEstimator() {
    try {
        ;
    }
    catch (...) {
        ;
    }
}
}  // namespace human_pose_estimation
