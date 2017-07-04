
#include <iostream>
#include <ifstream>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>


using Triangle = std::vector<int>;

struct Plane {
	cv::Vec3f normal;
	float d;
};

struct Pose {
  cv::Vec3f rot;
  cv::Vec3f trans;
};

// using Silhouette = std::vector<cv::Point2i>;
using Silhouettef = std::vector<cv::Point2f>;

struct Camera {
  cv::Mat matrix = cv::Mat::eye(3, 3, CV_32FC1);
  std::vector<float> ks;
};

struct Mesh {
  std::vector<cv::Point3f> points;
  std::vector<cv::Point3f> normals;
  std::vector<Triangle> triangles;
};

Mesh readMesh(std::string _filename) {
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
        std::cout << "Vertices/Normals: " << counts[PLYSection::VERTEX] << std::endl;
        std::cout << "Faces: " << counts[PLYSection::FACE] << std::endl;
      }
    }
    else if (cur_section == PLYSection::VERTEX) {
      if (0 < counts[cur_section]--) {
        std::istringstream iss(line);

        cv::Point3f pt;
        cv::Point3f nrm;
        iss >> pt.x >> pt.y >> pt.z >> nrm.x >> nrm.y >> nrm.z;

        points.push_back(pt);
        normals.push_back(nrm);
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

  std::cout << "Vertices/Normals left: " << counts[PLYSection::VERTEX] << std::endl;
  std::cout << "Faces left: " << counts[PLYSection::FACE] << std::endl;

  return {points, triangles};
}

cv::Mat mergeProcrustesTransform(cv::Vec2f procr_shift, float procr_scale, cv::Mat procr_rot) {
	cv::Mat hom_2d_transform = cv::Mat::eye(3, 3, CV_32FC1);

	cv::Mat rot = procr_rot;
	cv::Mat scale = (cv::Mat(3, 3, CV_32FC1) << procr_scale, 0, 0, 0, procr_scale, 0, 0 , 0, 1);
	cv::Mat shift = (cv::Mat(3, 3, CV_32FC1) << 0, 0, procr_shift(0), 0, 0, procr_shift(1), 0 , 0, 1);

	hom_2d_transform = shift * scale * rot;

	return hom_2d_transform;
}

// cv::Mat getHomSilhouetteTransform(Pose sil_pose, cv::Mat procr_trans) {	
// }

// cv::Vec2f centerOf(Sillhouettef &sil) {
// 	throw std::runtime_error("Not implemented");
// }

float getPlaneDepth(Plane &plane, cv::Point2f uv) {
	throw std::runtime_error("Not implemented");
}

std::Pair<cv::Mat, cv::Vec3f> map2dto3d(Camera &cam, Sillhouettef &sil, Pose &sil_pose, cv::Mat transf_2d, Plane &plane) {
	// cv::Mat similarity_trans = getHomSilhouetteTransform(sil_pose, transf_2d);
	cv::Mat similarity_trans = transf_2d;

	assert(similarity_trans.at<float>(3, 3) == 1);
	cv::Point2f uv(similarity_trans.at<float>(0, 2), similarity_trans.at<float>(1, 2));

	float plane_z = getPlaneDepth(plane, uv);

	cv::Mat K = cam.matrix;

	cv::Mat R_z = cv::Mat::eye(3, 3, CV_32FC1);
	float t_x{0}, t_y{0}, t_z{1};
	// TODO: solve equation (1/z) * similarity_trans * K * (x, y , plane_z).t() = 1/(plane_z + t_z) * K * R_z * (x, y, plane_z).t() + (t_x, t_y, t_z)
	// for R_z, t_x, t_y, t_z;

	cv::Mat R_y_R_x;
	cv::Rodrigues(sil_pose.rot, R_y_R_x);

	cv::Mat rot = R_z * R_y_R_x;
	cv::Vec3f shift(t_x, t_y, t_z);

	return {rot, shift};
}

void drawMesh(cv::Mat &dst, Camera &cam, Mesh &mesh, cv::Mat cam_sp_transform) {
	std::vector<cv::Point3f> vertice;
	std::vector<cv::Point3f> normal;

	vertice.reserve(mesh.points.size());
	normal.reserve(mesh.normals.size());

	assert(vertice.size() == normal.size());

	for (const auto &vtx : mesh.points)
		vertice.push_back(cam_sp_transform * vtx);

	for (const auto &nrm : mesh.normals)
		normal.push_back(cam_sp_transform * nrm);

	std::vector<cv::Point2f> vertice_2d;

	// FIXME: shift and scale points to dst size
	cv::Mat draw_cam_matrix = cv::Mat::eye(3, 3, CV_32FC1);
	draw_cam_matrix.at<float>()
	cv::projectPoints(vertice, cv::Vec3f(), cv::Vec3f(), draw_cam_matrix, {}, vertice_2d);

	cv::Vec3f light = cv::normalize(cv::Vec3f(-1, -1, -1));

	for (const auto &tri : mesh.triangles) {
		cv::Vec3f avg_normal = (normals[tri(0)] + normals[tri(1)] + normals[tri(2)]) / 3;

		cv::Scalar color = cv::Scalar(255, 255, 255) * 0.8 * avg_normal.dot(light);

    std::vector<cv::Point2i> poly{
        vertice_2d[tri[0]],
        vertice_2d[tri[1]],
        vertice_2d[tri[2]]};

    cv::fillConvexPoly(dst, poly, color);
	}
}

Pose getPoseEstimation()

int main() {
	cv::Mat rgb = cv::imread("color_4.png");

	Mesh cup_mesh = readMesh("cup.ply");

	Pose sil_pose;
	Silhouettef sil;
	Silhouettef segment;

	cv::Vec2f shift;
	cv::Vec3f rot;
	float scale;

	std::tie(shift, scale, rot) = procrFit(sil, segment);
	cv::Mat transform_2d = mergeProcrustesTransform(shift, scale, rot);

	Camera cam;
	Plane plane;

	auto pose_3d = map2dto3d(cam, sil, sil_pose, transform_2d, plane);

	// FIXME: full matrix, not just rotation part
	drawMesh(rgb, cam, cup_mesh, pose_3d.first);

	cv::imshow("Mesh", rgb);

	while ((cv::waitKey(100) & 255 ) != 27);

  return 0;
}