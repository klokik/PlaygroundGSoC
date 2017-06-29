#include <iostream>
#include <numeric>

#include <opencv2/opencv.hpp>

#define USE_PCL 0

#if USE_PCL

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#else

#include "libicp/src/icpPointToPoint.h"

#endif

using Silhouette = std::vector<cv::Point2f>;

cv::Point2f operator*(cv::Mat &M, const cv::Point2f &pt);

Silhouette normalizeSilhouette(Silhouette &shape) {
  Silhouette result(shape);

  cv::Point2f mean = std::accumulate(shape.begin(), shape.end(), cv::Point2f()) * (1.f / shape.size());

  float std_dev = 0;

  for (auto &pt : result) {
    pt = pt - mean;
    std_dev += std::pow(cv::norm(pt), 2);
  }

  std_dev = std::sqrt(std_dev / shape.size());

  for (auto &pt : result)
    pt *= 1.f / std_dev;

  return result;
}

#if USE_PCL
void silhouetteToPC(Silhouette &sil, pcl::PointCloud<pcl::PointXYZ> &pc) {
  pc.width = sil.size();
  pc.height = 1;
  pc.is_dense = false;

  pc.resize(pc.width * pc.height);

  for (size_t i = 0; i < sil.size(); ++i) {
    pc.points[i] = {sil[i].x, sil[i].y, 0};
  }
}

void PCToSilhouette(pcl::PointCloud<pcl::PointXYZ> &pc, Silhouette &sil) {
  sil.clear();

  assert(pc.height == 1);

  for (size_t i = 0; i < pc.width; ++i) {
    sil.push_back(cv::Point2f(pc.points[i].x, pc.points[i].y));
  }
}

std::pair<Silhouette, float> fitICP(Silhouette &test, Silhouette &model) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cl_test(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cl_model(new pcl::PointCloud<pcl::PointXYZ>);

  silhouetteToPC(test, *cl_test);
  silhouetteToPC(model, *cl_model);

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cl_test);
  icp.setInputTarget(cl_model);

  pcl::PointCloud<pcl::PointXYZ> cl_result;

  icp.align(cl_result);

  assert(icp.hasConverged());

  float score = icp.getFitnessScore();
  std::cout << "Score: " << score << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;

  Silhouette result;
  PCToSilhouette(cl_result, result);

  return {result, score};
}

#else

std::pair<Silhouette, float> fitICP(Silhouette &test, Silhouette &model) {

  std::vector<double> test_arr;
  std::vector<double> model_arr;

  for (auto &pt : test) {
    test_arr.push_back(pt.x);
    test_arr.push_back(pt.y);
  }

  for (auto &pt : model) {
    model_arr.push_back(pt.x);
    model_arr.push_back(pt.y);
  }

  int dim = 2; // 2D

  Matrix rot = Matrix::eye(2); // libipc matrix
  Matrix trans(2,1);           // libipc matrix

  IcpPointToPoint icp(&test_arr[0], test.size(), dim);
  float score = icp.fit(&model_arr[0], model.size(), rot, trans, -1);

  std::cout << "Score: " << score << std::endl;

  cv::Mat cv_trf(3, 3, CV_64FC1, cv::Scalar(0.f));
  cv_trf.at<double>(0, 0) = rot.val[0][0];
  cv_trf.at<double>(0, 1) = rot.val[1][0];
  cv_trf.at<double>(1, 0) = rot.val[0][1];
  cv_trf.at<double>(1, 1) = rot.val[1][1];

  cv_trf.at<double>(0, 2) = 0; //trans.val[0][0];
  cv_trf.at<double>(1, 2) = 0; //trans.val[1][0];
  cv_trf.at<double>(2, 2) = 1.f;

  std::cout << cv_trf << std::endl;

  Silhouette result;
  for (const auto &pt : test) {
    cv::Point2f ptf = pt;
    result.push_back(cv_trf * std::ref(ptf));
  }

  return {result, score};
}
#endif

const auto pi = 3.1415926f;

cv::Point2f operator*(cv::Mat &M, const cv::Point2f &pt) {
  cv::Mat_<double> vec(3, 1);

  vec(0, 0) = pt.x;
  vec(1, 0) = pt.y;
  vec(2, 0) = 1.f;

  cv::Mat_<double> dst = M*vec;

  return cv::Point2f(dst(0, 0), dst(1, 0));
}

Silhouette getTestSilhouette() {
  Silhouette result;

  cv::Point2f offset(50, 120);

  int N = 100;

  cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(), 30, 1);

  for (int i = 0; i < N; ++i) {
    float theta = 2*pi * i / N + pi/3;
    cv::Point2f pt(std::cos(theta)*2, std::sin(theta));
    pt *= 1+std::sin(theta*10)*0.1f;

    pt = rot*pt;

    result.push_back(pt*30 + offset);
  }

  return result;
}

Silhouette getModelSilhouette() {
  Silhouette result;

  cv::Point2f offset(128, 128);

  int N = 50;

  for (int i = 0; i < N; ++i) {
    float theta = 2*pi * i / N;
    cv::Point2f pt(std::cos(theta)*2, std::sin(theta));

    result.push_back(pt*20 + offset);
  }

  return result;
}

void drawSilhouette(cv::Mat &mat, Silhouette &sil) {
  for (auto &ptf : sil) {
    cv::circle(mat, ptf, 1, cv::Scalar(255));
  }
}

cv::Rect_<float> getBoundingRect(Silhouette &sil) {
  cv::Rect_<float> b_rect;

  auto h_it = std::minmax_element(sil.begin(), sil.end(),
    [](const cv::Point2f &a, const cv::Point2f &b) {
      return a.x < b.x;});
  auto v_it = std::minmax_element(sil.begin(), sil.end(),
    [](const cv::Point2f &a, const cv::Point2f &b) {
      return a.y < b.y;});

  b_rect.x = h_it.first->x;
  b_rect.y = v_it.first->y;
  b_rect.width = h_it.second->x - b_rect.x;
  b_rect.height = v_it.second->y - b_rect.y;

  return b_rect;
}

int main() {
  cv::Mat test_sm = cv::Mat::zeros(256, 256, CV_8UC1);
  cv::Mat model_sm = cv::Mat::zeros(256, 256, CV_8UC1);

  auto test_s = getTestSilhouette();
  auto model_s = getModelSilhouette();

  drawSilhouette(test_sm, test_s);
  drawSilhouette(model_sm, model_s);

  cv::imshow("Test Silhouette", test_sm);
  cv::imshow("Model Silhouette", model_sm);

  test_s = normalizeSilhouette(test_s);
  model_s = normalizeSilhouette(model_s);

  auto fitted__score = fitICP(test_s, model_s);

  Silhouette soup(fitted__score.first);
  soup.insert(soup.end(), model_s.begin(), model_s.end());

  int scale = 240;
  auto t_box = getBoundingRect(soup);
  for (auto &pt : soup)
    pt = (pt - t_box.tl())*scale;

  cv::Mat tnsm = cv::Mat::eye(t_box.height*scale, t_box.width*scale, CV_8UC1);
  drawSilhouette(tnsm, soup);

  cv::imshow("Shifted, scaled, rotated", tnsm);

  while ((cv::waitKey(0) & 255) != 27);

  return 0;
}
