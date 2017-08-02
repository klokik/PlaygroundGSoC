#include <iostream>
#include <numeric>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/norms.h>
// #include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>


using Silhouette = std::vector<cv::Point2f>;

using Triangle = std::vector<int>;

int scale = 6;


struct Mesh {
  std::vector<cv::Point3f> points;
  std::vector<cv::Vec3f> normals;
  std::vector<Triangle> triangles;
};

::Mesh readTrainingMesh(std::string _filename) {
  std::vector<cv::Point3f> points;
  std::vector<cv::Vec3f> normals;
  std::vector<Triangle> triangles;

  std::string filename = _filename;
  std::ifstream ifs(filename);

  if (!ifs.good())
    throw std::runtime_error("File '"+filename+"' not found");

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
        std::cout << "Vertices/normals: " << counts[PLYSection::VERTEX] << std::endl;
        std::cout << "Faces: " << counts[PLYSection::FACE] << std::endl;
      }
    }
    else if (cur_section == PLYSection::VERTEX) {
      if (0 < counts[cur_section]) {
        std::istringstream iss(line);

        cv::Point3f pt;
        cv::Point3f nrm;
        iss >> pt.x >> pt.y >> pt.z >> nrm.x >> nrm.y >> nrm.z;

        points.push_back(pt);
        normals.push_back(nrm);
        --counts[cur_section];
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

  assert(counts[PLYSection::VERTEX] == 0);
  assert(counts[PLYSection::FACE] == 0);

  return {points, normals, triangles};
}

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

  assert(pc.height == 1);

  for (size_t i = 0; i < pc.width; ++i) {
    sil.push_back(cv::Point2f(pc.points[i].x, pc.points[i].y));
  }
}

/*Silhouette fitICP(Silhouette &test, Silhouette &model) {
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
}*/

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

  for (int i = 0; i < N*1.f; ++i) {
    float theta = 2*pi * i / N + pi/3;
    cv::Point2f pt(std::cos(theta)*2, std::sin(theta));
    pt *= 1+std::sin(theta*10)*0.1f;

    pt = rot*pt;

    result.push_back(pt*30 + offset);
  }

  return result;
}

/*Silhouette getModelSilhouette() {
  Silhouette result;

  cv::Point2f offset(128, 128);
  // cv::Point2f offset(0, 0);

  int N = 500;

  for (int i = 0; i < N; ++i) {
    // float theta = 2*pi * i / N;
    // cv::Point2f pt(std::cos(theta)*2, std::sin(theta));

    cv::Point2f pt; 
    float l = (i*1.f/N-0.5f)*2;
    switch (i % 4) {
    case 0: pt.x = 1;
            pt.y = l;
      break;
    case 1: pt.x = -1;
            pt.y = l;
      break;
    case 2: pt.x = l;
            pt.y = 1;
      break;
    case 3: pt.x = l;
            pt.y = -1;
      break;
    }

    result.push_back(pt*20 + offset);
  }

  return result;
}*/

cv::Mat init_pose = (cv::Mat_<double>(6, 1) << 0,0.3,0, 0.00,-0.06,-0.13);

Silhouette getModelSilhouette() {
  std::ifstream ifs("edges.txt");

  size_t N {0};
  ifs >> N;

  Silhouette result;
  for (size_t i = 0; i < N; ++i) {
    cv::Point2f pt;
    ifs >> pt.x >> pt.y;

    result.push_back(pt);
  }

  cv::Mat params = cv::Mat::zeros(6, 1, CV_64FC1);
  for (int i = 0; i < 6; ++i)
    ifs >> params.at<double>(i);

  ifs.close();

  init_pose = params;
  return result;
}

void drawSilhouette(cv::Mat &mat, Silhouette &sil, size_t num1 = 0) {
  size_t i {0};

  for (auto &ptf : sil) {
    cv::Scalar color = cv::Scalar(0, 0, 255);
    if (i > num1)
      color = cv::Scalar(0, 255, 0);
    cv::circle(mat, ptf, 1, color);

    ++i;
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

pcl::PointXYZ getNearestPoint(pcl::KdTree<pcl::PointXYZ> &template_kdtree, const pcl::PointXYZ &pt) {
  std::vector<int> indices;
  std::vector<float> l2_sqr_distances;

  assert(template_kdtree.nearestKSearch(pt, 1, indices, l2_sqr_distances) == 1);

  auto cloud = template_kdtree.getInputCloud();
  auto out_pt = cloud->points[indices.front()];

  return out_pt;
}

pcl::PointXYZ transform(cv::Vec3f params, pcl::PointXYZ &point) {
  cv::Mat transformation = cv::getRotationMatrix2D(cv::Point2f(0, 0), params[0]*180/3.1415926, 1.);
  transformation.convertTo(transformation, CV_32FC1);
  transformation.at<float>(0, 2) = params[1];
  transformation.at<float>(1, 2) = params[2];

  cv::Mat vec = (cv::Mat_<float>(3, 1) << point.x, point.y, 1.f);

  cv::Mat out = transformation * vec;

  pcl::PointXYZ result;
  result.x = out.at<float>(0, 0);
  result.y = out.at<float>(1, 0);
  result.z = 0;

  return result;
}

pcl::PointCloud<pcl::PointXYZ> transform(cv::Vec3f params, pcl::PointCloud<pcl::PointXYZ> &cloud) {
  cv::Mat transformation = cv::getRotationMatrix2D(cv::Point2f(0, 0), params[0]*180/3.1415926, 1.);
  transformation.convertTo(transformation, CV_32FC1);
  transformation.at<float>(0, 2) = params[1];
  transformation.at<float>(1, 2) = params[2];

  pcl::PointCloud<pcl::PointXYZ> transformed(cloud);
  for (auto &pt : transformed.points) {
    cv::Mat vec = (cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.f);

    cv::Mat out = transformation * vec;
    pt.x = out.at<float>(0, 0);
    pt.y = out.at<float>(1, 0);
  }

  return transformed;
}

pcl::PointCloud<pcl::PointXYZ> projectSurfacePoints(cv::Mat &params, ::Mesh &mesh) {
  ::Silhouette points_2d;
  cv::Mat cam_mat = cv::Mat::eye(3, 3, CV_32FC1);
  cam_mat = (cv::Mat_<double>(3, 3) << 570.3422241210938, 0.0, 319.5, 0.0, 570.3422241210938, 239.5, 0.0, 0.0, 1.0);
  cv::Vec3f rot(params.at<double>(0,0), params.at<double>(1,0), params.at<double>(2,0));
  cv::Vec3f trans(params.at<double>(3,0), params.at<double>(4,0), params.at<double>(5,0));

  cv::projectPoints(mesh.points, rot, trans, cam_mat, {}, points_2d);

  pcl::PointCloud<pcl::PointXYZ> result;
  silhouetteToPC(points_2d, result);

  return result;
}

Silhouette fitICP2d(Silhouette &test, Silhouette &model) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cl_test(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cl_model(new pcl::PointCloud<pcl::PointXYZ>);

  silhouetteToPC(test, *cl_test);
  silhouetteToPC(model, *cl_model);

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cl_model);

  cv::Vec3f params (0.9f, 0.5, -1);

  Silhouette result;

  for (size_t i = 0; ; ++i) {

    // params[0] = 3.1415*i/10;

    auto getErr = [] (cv::Vec3f params, pcl::PointCloud<pcl::PointXYZ> &_test, pcl::KdTree<pcl::PointXYZ> &kdtree) {
      auto transformed_cloud = transform(params, _test);
      double sum {0};
      int num {0};
      for (auto &pt : transformed_cloud.points) {
        pcl::PointXYZ cpt = getNearestPoint(kdtree, pt);

        float residual = pcl::L2_Norm_SQR(pt.data, cpt.data, 3);
        if (residual < 0.7) {
          sum += residual;
          ++num;
        }
      }
      if (num == 0)
        return std::numeric_limits<double>::max();
      else
        return std::sqrt(sum) / num;
    };

    double ref_err = getErr(params, *cl_test, kdtree);
    std::cout << "Matching error: " << ref_err << std::endl;
    std::cout << "params: " << params << std::endl;

    pcl::PointCloud<pcl::PointXYZ> transformed = transform(params, *cl_test);

    // find derivative/jacobian
    {
      double h = 1e-3;
      float learning_rate = 1e-1;

      int dof = 3;

      cv::Mat jacobian = cv::Mat::zeros(cl_test->width, dof, CV_64FC1);

      cv::Mat residuals(cl_test->width, 1, CV_64FC1);
      for (int k = 0; k < cl_test->width; ++k) {
        auto pt_ref = transform(params, cl_test->points[k]);
        auto clpt_ref = getNearestPoint(kdtree, pt_ref);

        double d1 = pcl::L2_Norm(pt_ref.data, clpt_ref.data, 3);

        residuals.at<double>(k, 0) = d1;
      }

      for (int j = 0; j < dof; ++j) {
        cv::Vec3f offset(0, 0, 0);
        offset[j] = h;
        cv::Vec3f params_plus = params + offset;
        cv::Vec3f params_minus = params - offset;
        // double err_plus = getErr(params + offset, *cl_test, kdtree);
        // double err_minus = getErr(params - offset, *cl_test, kdtree);

        // float dEdPj = (err_plus - err_minus) / (2 * h);
        // std::cout << "dEdPj: " << dEdPj << std::endl;

        // params[j] += -dEdPj*ref_err*learning_rate;

        for (int k = 0; k < cl_test->width; ++k) {
          if (residuals.at<double>(k, 0) > 0.7)
            continue;

          auto pt_plus = transform(params_plus, cl_test->points[k]);
          auto pt_minus = transform(params_minus, cl_test->points[k]);

          auto clpt_plus = getNearestPoint(kdtree, pt_plus);
          auto clpt_minus = getNearestPoint(kdtree, pt_minus);

          double d1 = pcl::L2_Norm_SQR(pt_plus.data, clpt_plus.data, 3);
          double d2 = pcl::L2_Norm_SQR(pt_minus.data, clpt_minus.data, 3);

          double dEk_dPj = (d1 - d2) / (2 * h);

          jacobian.at<double>(k, j) = dEk_dPj;
        }
      }
      jacobian = jacobian / cv::norm(jacobian, cv::NORM_INF);
      // std::cout << jacobian << std::endl;
      // break;

      if (cv::countNonZero(jacobian) == 0) {
        std::cout << "Failed" << std::endl;
        break;
      }

      cv::Mat pinv_jacobian = jacobian.inv(cv::DECOMP_SVD);
      cv::Mat delta_pose_mat = pinv_jacobian * residuals;

      double step_size = cv::norm(delta_pose_mat);
      std::cout << "Step: " << step_size << std::endl;

      if (step_size < 0.03) {
        std::cout << "Done in " << i << " iterations" << std::endl;
        break;
      }

      params -= cv::Vec3f(delta_pose_mat.at<double>(0, 0), delta_pose_mat.at<double>(1, 0), delta_pose_mat.at<double>(2, 0)) * learning_rate;
    }

    result.clear();
    for (auto &pt : transformed.points)
      result.push_back(cv::Point2f(pt.x, pt.y));

    // draw
    {
      Silhouette soup(result);
      soup.insert(soup.end(), model.begin(), model.end());

      int scale = 240;
      auto t_box = getBoundingRect(soup);
      for (auto &pt : soup)
        pt = (pt - t_box.tl())*scale;

      cv::Mat tnsm = cv::Mat::eye(t_box.height*scale, t_box.width*scale, CV_8UC3);
      drawSilhouette(tnsm, soup);

      cv::imshow("2d-3d icp", tnsm);

      if ((cv::waitKey(32) & 255) == 27)
        break;
    }
  }

  return result;
}

Silhouette fitICP3d(::Mesh &mesh, Silhouette &model) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cl_model(new pcl::PointCloud<pcl::PointXYZ>);

  silhouetteToPC(model, *cl_model);

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cl_model);

  cv::Mat params = init_pose;

  Silhouette result;
  double last_err = 0;
  size_t stall_counter = 0;

  for (int i = 0; ; ++i) {
    pcl::PointCloud<pcl::PointXYZ> projected_2d = projectSurfacePoints(params, mesh);

    result.clear();
    for (auto &pt : projected_2d.points)
      result.push_back(cv::Point2f(pt.x, pt.y));

    // draw
    {
      Silhouette soup(result);
      soup.insert(soup.end(), model.begin(), model.end());

      auto t_box = getBoundingRect(soup);
      std::cout << t_box << std::endl;
      for (auto &pt : soup)
        pt = (pt - t_box.tl())*scale;

      cv::Mat tnsm = cv::Mat::eye(t_box.height*scale, t_box.width*scale, CV_8UC3);
      drawSilhouette(tnsm, soup, result.size());

      cv::imshow("2d-3d icp", tnsm);

      if ((cv::waitKey(32) & 255) == 27)
        break;
    }

    auto getErr = [] (cv::Mat params, ::Mesh &mesh, pcl::KdTree<pcl::PointXYZ> &kdtree) {
      auto transformed_cloud = projectSurfacePoints(params, mesh);
      double sum {0};
      int num {0};
      for (auto &pt : transformed_cloud.points) {
        pcl::PointXYZ cpt = getNearestPoint(kdtree, pt);

        float residual = pcl::L2_Norm_SQR(pt.data, cpt.data, 3);
        // std::cout << residual << " ";
        if (residual < 0.9) {
          sum += residual;
          ++num;
        }
      }
      std::cout << num << "/" <<  transformed_cloud.width << std::endl;
      if (num == 0)
        return std::numeric_limits<double>::max();
      else
        return std::sqrt(sum) / num;
    };

    double ref_err = getErr(params, mesh, kdtree);

    double epsilon = 1e-5;
    if (std::abs(last_err - ref_err) < epsilon) {
      if (stall_counter == 5) {
        std::cout << "Done in " << i << " iterations" << std::endl;
        break;
      }
      stall_counter++;
    }
    else
      stall_counter = 0;

    last_err = ref_err;

    int dof = 6;
    double h = 1e-3;
    float learning_rate = 1e-4;

#if 0
    learning_rate = 0.2;
    for (int j = 0; j < dof; ++j) {
      cv::Mat offset = cv::Mat::zeros(params.size(), CV_64FC1);
      offset.at<double>(j) = h;

      cv::Mat params_plus = params + offset;
      cv::Mat params_minus = params - offset;

      double err_plus = getErr(params + offset, mesh, kdtree);
      double err_minus = getErr(params - offset, mesh, kdtree);

      float dEdPj = (err_plus - err_minus) / (2 * h);
      std::cout << "dEdPj: " << dEdPj << std::endl;

      params.at<double>(j) += -dEdPj*ref_err*learning_rate;
    }
#else
    {
      cv::Mat jacobian = cv::Mat::zeros(projected_2d.width, dof, CV_64FC1);

      cv::Mat residuals(projected_2d.width, 1, CV_64FC1);
      for (int k = 0; k < projected_2d.width; ++k) {
        auto pt_ref = projected_2d.points[k];
        auto clpt_ref = getNearestPoint(kdtree, pt_ref);

        double d1 = pcl::L2_Norm(pt_ref.data, clpt_ref.data, 3);

        residuals.at<double>(k, 0) = d1;
      }

      for (int j = 0; j < dof; ++j) {
        cv::Mat offset = cv::Mat::zeros(params.size(), CV_64FC1);
        offset.at<double>(j) = h;

        cv::Mat params_plus = params + offset;
        cv::Mat params_minus = params - offset;

        auto cloud_plus = projectSurfacePoints(params_plus, mesh);
        auto cloud_minus = projectSurfacePoints(params_minus, mesh);

        for (int k = 0; k < projected_2d.width; ++k) {
          if (residuals.at<double>(k, 0) > 5)//0.7)
            continue;

          auto clpt_plus = getNearestPoint(kdtree, cloud_plus.points[k]);
          auto clpt_minus = getNearestPoint(kdtree, cloud_minus.points[k]);

          double d1 = pcl::L2_Norm_SQR(cloud_plus.points[k].data, clpt_plus.data, 3);
          double d2 = pcl::L2_Norm_SQR(cloud_minus.points[k].data, clpt_minus.data, 3);

          double dEk_dPj = (d1 - d2) / (2 * h);

          jacobian.at<double>(k, j) = dEk_dPj;
        }
      }
      jacobian = jacobian / cv::norm(jacobian, cv::NORM_INF);

      if (cv::countNonZero(jacobian) == 0) {
        std::cout << "Failed" << std::endl;
        break;
      }

      cv::Mat delta_pose_mat;
      cv::solve(jacobian, residuals, delta_pose_mat, cv::DECOMP_SVD);

      params -= delta_pose_mat * learning_rate;
    }
#endif

    std::cout << "Pose: " << params << std::endl;
    std::cout << "Error: " << ref_err << std::endl;
  }

  return result;
}

int main() {
  cv::Mat test_sm = cv::Mat::zeros(480, 640, CV_8UC3);
  cv::Mat model_sm = cv::Mat::zeros(480, 640, CV_8UC3);

  // auto test_s = getTestSilhouette();
  auto model_s = getModelSilhouette();

  // drawSilhouette(test_sm, test_s);
  drawSilhouette(model_sm, model_s);

  cv::imshow("Test Silhouette", test_sm);
  cv::imshow("Model Silhouette", model_sm);

  // test_s = normalizeSilhouette(test_s);
  model_s = model_s;//normalizeSilhouette(model_s);

  // auto fitted = fitICP(test_s, model_s);
  // Silhouette fitted = fitICP2d(test_s, model_s);
  ::Mesh mesh = readTrainingMesh("pokal_edge.ply");
  auto fitted = fitICP3d(mesh, model_s);

  Silhouette soup(fitted);
  soup.insert(soup.end(), model_s.begin(), model_s.end());

  auto t_box = getBoundingRect(soup);
  for (auto &pt : soup)
    pt = (pt - t_box.tl())*scale;

  cv::Mat tnsm = cv::Mat::eye(t_box.height*scale, t_box.width*scale, CV_8UC3);
  drawSilhouette(tnsm, soup);

  cv::imshow("2d-3d", tnsm);

  while ((cv::waitKey(0) & 255) != 27);

  return 0;
}
