#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>


using Triangle = std::vector<int>;

struct Mesh {
  std::vector<cv::Point3f> points;
  std::vector<Triangle> triangles;
};

struct Pose {
  cv::Vec3f rot;
  cv::Vec3f trans;
};

using Silhouette = std::vector<cv::Point2i>;

struct Footprint {
  cv::Mat img;
  Silhouette contour;
  Pose pose;
};

class EdgeModel {
  public: void addFootprint(cv::Mat &footprint, const Silhouette &contour, const Pose &pose) {
    this->items.push_back({footprint.clone(), contour, pose});
  }

  public: std::vector<Footprint> items;
};

struct Camera {
  cv::Mat matrix = cv::Mat::zeros(3, 3, CV_32FC1);
  std::vector<float> ks;
};

Mesh getRandomMesh(void) {
  std::vector<cv::Point3f> points; 
  std::vector<Triangle> triangles; 

  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> distrib(-1, 1);

  for (int i = 0; i < 1000; ++i)
    points.push_back(cv::Point3f(distrib(rng)+1, distrib(rng)*4-5, distrib(rng)));

  return {points, triangles};
}

Mesh readTrainingMesh(std::string _filename) {
  std::vector<cv::Point3f> points; 
  std::vector<Triangle> triangles; 

  std::ifstream ifs(_filename);

  enum class PLYSection : int { HEADER=0, VERTEX, FACE};
  std::map<PLYSection, int> counts;

  PLYSection cur_section = PLYSection::HEADER;
  for (std::string line; std::getline(ifs, line);) {
    if (cur_section == PLYSection::HEADER) {
      if (line.find("element face") == 0)
        counts[PLYSection::FACE] = std::atoi(line.substr(line.rfind(" ")).c_str());
      if (line.find("element vertex") == 0)
        counts[PLYSection::VERTEX] = std::atoi(line.substr(line.rfind(" ")).c_str());
      if (line.find("end_header") == 0) {
        cur_section = PLYSection::VERTEX;
        std::cout << "Vertices: " << counts[PLYSection::VERTEX] << std::endl;
        std::cout << "Faces: " << counts[PLYSection::FACE] << std::endl;
      }
    }
    else if (cur_section == PLYSection::VERTEX) {
      if (0 < counts[cur_section]--) {
        std::istringstream iss(line);

        cv::Point3f pt;
        iss >> pt.x >> pt.y >> pt.z;

        points.push_back(pt);
      }
      else
        cur_section = PLYSection::FACE;
    }
    if (cur_section == PLYSection::FACE) {
      if (0 == counts[cur_section]--)
        break;

      std::istringstream iss(line);

      int n_verts, i1, i2, i3;
      iss >> n_verts >> i1 >> i2 >> i3;
      assert(n_verts == 3);

      triangles.push_back({i1, i2, i3});
    }
  }

  std::cout << "Vertices left: " << counts[PLYSection::VERTEX] << std::endl;
  std::cout << "Faces left: " << counts[PLYSection::FACE] << std::endl;

  return {points, triangles};
}

Footprint getFootprint(Mesh mesh, Pose pose, Camera cam, int im_size) {
  // project points on a plane
  std::vector<cv::Point2f> points2d;
  cv::projectPoints(mesh.points, pose.rot, pose.trans, cam.matrix, cam.ks, points2d);

  // find points2d bounding rect
  cv::Rect_<float> b_rect;
  // b_rect = cv::boundingRect2f(points2d); // available since 2.4.something
  auto h_it = std::minmax_element(points2d.begin(), points2d.end(),
    [](const cv::Point2f &a, const cv::Point2f &b) {
      return a.x < b.x;});
  auto v_it = std::minmax_element(points2d.begin(), points2d.end(),
    [](const cv::Point2f &a, const cv::Point2f &b) {
      return a.y < b.y;});

  b_rect.x = h_it.first->x;
  b_rect.y = v_it.first->y;
  b_rect.width = h_it.second->x - b_rect.x;
  b_rect.height = v_it.second->y - b_rect.y;

  auto larger_size = std::max(b_rect.width, b_rect.height);

  auto rate = static_cast<float>(im_size)/larger_size;
  cv::Size fp_mat_size(b_rect.width*rate + 1, b_rect.height*rate + 1);

  cv::Mat footprint = cv::Mat::zeros(fp_mat_size, CV_8UC1);

  // std::cout << fp_mat_size << std::endl;
  // std::cout << b_rect << std::endl;
  // map point onto plane
  for (auto &point : points2d) {
    cv::Point2i xy = (point - b_rect.tl())*rate;

    assert(xy.x >= 0);
    assert(xy.y >= 0);
    assert(xy.x <= footprint.cols);
    assert(xy.y <= footprint.rows);

    xy.x = std::min(xy.x, footprint.cols-1);
    xy.y = std::min(xy.y, footprint.rows-1);

    point = xy;
  }

  for (const auto &tri : mesh.triangles) {
    std::vector<cv::Point2i> poly{
        points2d[tri[0]],
        points2d[tri[1]],
        points2d[tri[2]]};

    cv::fillConvexPoly(footprint, poly, cv::Scalar(255));
  }

  int margin = 4;
  cv::copyMakeBorder(footprint, footprint, margin, margin, margin, margin,
    cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0));

  cv::Mat tmp = footprint.clone();
  std::vector<Silhouette> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(tmp, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  assert(contours.size() == 1);

  cv::Mat mkernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::morphologyEx(footprint, footprint, cv::MORPH_GRADIENT, mkernel,
    cv::Point(-1,-1), 1);

  cv::imshow("footprint", footprint);
  cv::waitKey(100);

  return {footprint, contours[0], pose};
}

EdgeModel getSampledFootprints(Mesh &mesh, Camera &cam, int im_size,
  int rot_samples, int trans_samples) {

  EdgeModel e_model;

  auto it = std::max_element(mesh.points.cbegin(), mesh.points.cend(),
    [](const cv::Point3f &a, const cv::Point3f &b) {
      return cv::norm(a) < cv::norm(b); });
  float mlen = cv::norm(*it);

  const auto pi = 3.1415926f;
  for (int r_ax_i = 0; r_ax_i < rot_samples; ++r_ax_i) {
    float axis_inc = pi * r_ax_i / rot_samples;
    cv::Vec3f axis{std::cos(axis_inc), std::sin(axis_inc), 0};

    for (int r_ang_i = 0; r_ang_i < rot_samples; ++r_ang_i ) {
      float theta = pi * r_ang_i / rot_samples;
      auto rodrigues = axis*theta;

      std::cout << "Sample (" << r_ax_i << ";" << r_ang_i << ")" << std::endl;

      // avoid translation sampling for now
      Pose mesh_pose{rodrigues, {0,0,-mlen*3.f}};

      auto footprint = getFootprint(mesh, mesh_pose, cam, im_size);
      e_model.addFootprint(footprint.img, footprint.contour, footprint.pose);
    }
  }

  return e_model;
}

int main() {
  Mesh test_mesh = readTrainingMesh("monkey.ply");

  int im_size = 64;
  Camera camera;
  camera.matrix.at<float>(0, 0) = 1;
  camera.matrix.at<float>(1, 1) = 1;
  camera.matrix.at<float>(2, 2) = 1;
  camera.matrix.at<float>(0, 2) = 0;//im_size/2;
  camera.matrix.at<float>(1, 2) = 0;//im_size/2;

  int rot_samples = 10;
  int trans_samples = 1;

  auto edge_model = getSampledFootprints(test_mesh, camera, im_size, rot_samples, trans_samples);

  auto seg_size = (im_size+9);
  cv::Mat whole_image = cv::Mat::zeros(seg_size*rot_samples, seg_size*rot_samples, CV_8UC1);

  int i = 0;
  for (auto kv : edge_model.items) {
    auto ix = (i % rot_samples)*seg_size;
    auto iy = (i / rot_samples)*seg_size;
    i++;

    kv.img.copyTo(whole_image.colRange(ix, ix+kv.img.cols).
      rowRange(iy, iy+kv.img.rows));
  }

  cv::imshow("Silhouettes", whole_image);
  while ((cv::waitKey(100) % 255) != 27);

  return 0;
}