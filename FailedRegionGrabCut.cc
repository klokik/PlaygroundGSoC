
#include <iostream>

#include <opencv2/opencv.hpp>

extern int depth_height;
extern int depth_width;
extern float depth_data[];

int main() {
	cv::Mat depth(depth_height, depth_width, CV_32FC1, &depth_data[0]);
	depth.convertTo(depth, CV_16UC1, 65535./2);
	cv::imwrite("/tmp/depth.tiff", depth);
	// cv::Mat depth = cv::imread("depth_2.png", cv::IMREAD_GRAYSCALE);
	depth.convertTo(depth, CV_32FC1, 1./255, 0);
	// cv::Mat im_mask = cv::imread("registrationMask.png", cv::IMREAD_GRAYSCALE);
	cv::Mat rgb = cv::imread("color_2.png");//"image.png");

	cv::threshold(depth, depth, 0.05, 1.0, cv::THRESH_BINARY);
	// cv::threshold(depth, depth, 0, 1.0, cv::THRESH_BINARY);
	depth.convertTo(depth, CV_8UC1, 255., 0.);

	// cv::bitwise_or(im_mask, depth, depth);
	cv::bitwise_not(depth, depth);

	cv::Mat contours_mat = depth.clone();
	std::vector<std::vector<cv::Point>> contours;
	// cv::findContours(contours_mat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	// for (auto &contour : contours) {
	// 	std::vector<std::vector<cv::Point>> contour_list{contour};
	// 	cv::fillPoly(depth, contour_list, cv::Scalar(255, 255, 255));
	// }

	cv::imshow("Depth", depth);
	cv::Mat morph_cl_op;
	cv::Mat mkernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(depth, depth, cv::MORPH_CLOSE, mkernel, cv::Point(-1,-1), 12);
	cv::imshow("DepthCL", depth);

	cv::morphologyEx(depth, depth, cv::MORPH_OPEN, mkernel, cv::Point(-1,-1), 6);
	cv::imshow("DepthCLOP", depth);

	// cv::morphologyEx(depth, depth, cv::MORPH_CLOSE, mkernel, cv::Point(-1,-1), 15);
	// cv::imshow("DepthCLOPCL", depth);

	contours_mat = depth.clone();
	std::vector<std::vector<cv::Point>> hulls;
	cv::findContours(contours_mat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	for (auto &contour : contours) {
		std::vector<cv::Point> hull;
		cv::convexHull(contour, hull);
		hulls.push_back(hull);
		std::vector<std::vector<cv::Point>> contour_list{contour};
		cv::fillPoly(depth, contour_list, cv::Scalar(255, 255, 255));
	}
	// cv::drawContours(rgb, hulls, -1, cv::Scalar(0, 0, 255), 2);
	// cv::fillPoly(depth, hulls, cv::Scalar(255, 255, 255));
	// cv::fillPoly(depth, contours, cv::Scalar(255, 255, 255));
	cv::imshow("ConvHull", depth);

	// ROI's and corresponding masks for grabCut
	contours.clear();
	cv::Mat depth_eroded, depth_dilated;
	cv::erode(depth, depth_eroded, mkernel, cv::Point(-1, -1), 6);
	cv::dilate(depth, depth_dilated, mkernel, cv::Point(-1, -1), 12);

	contours_mat = depth.clone();
	cv::findContours(contours_mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	std::vector<cv::Rect> rois;
	std::vector<cv::Mat> roi_masks;
	for (auto &contour : contours) {
		if (cv::contourArea(contour) < 100)
			continue;

		constexpr int margin = 20;
		cv::Rect roi = cv::boundingRect(contour);
		roi.x = std::max(0, roi.x - margin);
		roi.y = std::max(0, roi.y - margin);
		roi.width = std::min(depth.cols - roi.x, roi.width + 2*margin);
		roi.height = std::min(depth.rows - roi.y, roi.height + 2*margin);

		cv::Mat c_mask(depth.size(), CV_8UC1, cv::GC_BGD);
		c_mask(roi).setTo(cv::GC_PR_BGD, depth_dilated(roi));
		c_mask(roi).setTo(cv::GC_PR_FGD, depth(roi));
		c_mask(roi).setTo(cv::GC_FGD, depth_eroded(roi));

		rois.push_back(roi);
		roi_masks.push_back(c_mask(roi));

		// cv::imshow("roi", c_mask(roi)*255.f/3);
		// cv::waitKey();
	}

	cv::Mat refined_depth(depth.size(), CV_8UC1, cv::GC_BGD);
	for (size_t i = 0; i < rois.size(); ++i) {
		auto &roi = rois[i];
		auto &gc_mask = roi_masks[i];

		cv::Mat bgd_model, fgd_model;
		cv::grabCut(rgb(roi), gc_mask, cv::Rect(),
			bgd_model, fgd_model, 2, cv::GC_INIT_WITH_MASK);

		cv::Mat refined_depth_roi = refined_depth(roi);
		cv::Mat copy_mask = (gc_mask != cv::GC_BGD) & (gc_mask != cv::GC_PR_BGD);
		gc_mask.copyTo(refined_depth_roi, copy_mask);

/*		cv::imshow("mask_rgb", rgb(roi));
		cv::Mat tmp;
		rgb(roi).copyTo(tmp, copy_mask);
		cv::imshow("masked", tmp);
		cv::waitKey();*/
	}

	refined_depth = ((refined_depth == cv::GC_FGD)
		| (refined_depth == cv::GC_PR_FGD));
	cv::imshow("gc_refined", refined_depth);

	contours_mat = refined_depth.clone();
	cv::findContours(contours_mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	cv::drawContours(rgb, contours, -1, cv::Scalar(0, 255, 255), 1);

	// cv::imshow("RGB", rgb);
	// cv::imshow("RMask", im_mask);
	// cv::imshow("RMask", im_mask);
	// cv::imshow("DepthM", depth);

	cv::Mat masked_rgb;
	refined_depth.convertTo(masked_rgb, CV_32FC1, 0.7/255, 0.3);
	cv::Mat masked_rgb3;
	cv::Mat l[] = {masked_rgb, masked_rgb, masked_rgb};
	cv::merge(l, 3, masked_rgb3);

	cv::multiply(masked_rgb3, rgb, masked_rgb3, 1./255, CV_32F);
	cv::imshow("mRGB", masked_rgb3);

	while(cv::waitKey(100) != 27);

	return 0;
}
