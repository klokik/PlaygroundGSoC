#include <iostream>
#include <numeric>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>


using Silhouette = std::vector<cv::Point2f>;

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

  for (size_t i = 0; i < sil.size(); ++i) {
    sil.push_back(cv::Point2f(pc.points[i].x, pc.points[i].y));
  }
}

Silhouette fitICP(Silhouette &test, Silhouette &model) {
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

  std::cout << "Score: " << icp.getFitnessScore() << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;

  Silhouette result;
  PCToSilhouette(cl_result, result);

  return result;
}

const auto pi = 3.1415926f;

Silhouette getTestSilhouette() {
  Silhouette result;

  cv::Point2f offset(50, 120);

  int N = 100;

  for (int i = 0; i < N; ++i) {
    float theta = 2*pi * i / N + pi/3;
    cv::Point2f pt(std::cos(theta), std::sin(theta)*2);
    pt *= 1+std::sin(theta)+std::sin(theta*10)*0.1f;

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

  fitICP(test_s, model_s);

  Silhouette soup(test_s);
  soup.insert(soup.end(), model_s.begin(), model_s.end());

  int scale = 240;
  auto t_box = getBoundingRect(soup);
  for (auto &pt : soup)
    pt = (pt - t_box.tl())*scale;

  cv::Mat tnsm = cv::Mat::eye(t_box.height*scale, t_box.width*scale, CV_8UC1);
  drawSilhouette(tnsm, soup);

  cv::imshow("Shifted, scaled", tnsm);

  while (cv::waitKey(0) != 27);

  return 0;
}
