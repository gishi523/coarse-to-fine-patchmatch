#include <opencv2/opencv.hpp>
#include <chrono>
#include "coarse_to_fine_patchmatch.h"

void drawOpticalFlow(cv::Mat& img, const cv::Mat4f& matches, float maxnorm = -1)
{
	const int w = img.cols;
	const int h = img.rows;
	const int nmatches = matches.rows;

	if (maxnorm < 0)
	{
		for (int i = 0; i < nmatches; i++)
		{
			const cv::Vec4f& match = matches(i);
			const float fx = match[2] - match[0];
			const float fy = match[3] - match[1];
			maxnorm = std::max(maxnorm, std::hypotf(fx, fy));
		}
	}

	const float INV_2PI = static_cast<float>(1 / CV_2PI);
	const int radius = 1;
	for (int i = 0; i < nmatches; i++)
	{
		const cv::Vec4f& match = matches(i);
		const float fx = match[2] - match[0];
		const float fy = match[3] - match[1];

		// convert flow angle to hue
		float angle = INV_2PI * atan2f(fy, fx);
		if (angle < 0.f) angle += 1.f;
		const uchar hue = static_cast<uchar>(180 * angle);

		// convert flow norm to saturation
		const float norm = std::hypotf(fx, fy) / maxnorm;
		const uchar sat = static_cast<uchar>(255 * norm);

		// draw each match as a 3x3 color block
		for (int dy = -radius; dy <= radius; dy++)
		{
			for (int dx = -radius; dx <= radius; dx++)
			{
				const int x = std::max(0, std::min(static_cast<int>(match[0] + dx + 0.5f), w - 1));
				const int y = std::max(0, std::min(static_cast<int>(match[1] + dy + 0.5f), h - 1));
				img.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(hue, sat, 255);
			}
		}
	}

	cv::cvtColor(img, img, cv::COLOR_HSV2BGR);
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cout << "usage: " << argv[0] << " image1 image2" << std::endl;
		return 0;
	}

	const cv::Mat I1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	const cv::Mat I2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

	CoarseToFinePatchMatch cpm;

	const auto t1 = std::chrono::system_clock::now();

	const cv::Mat4f matches = cpm.compute(I1, I2);

	const auto t2 = std::chrono::system_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "cpm computation time: " << duration << "[msec]" << std::endl;

	cv::Mat draw = cv::Mat::zeros(I1.size(), CV_8UC3);
	drawOpticalFlow(draw, matches);
	cv::imshow("image1", I1);
	cv::imshow("image2", I2);
	cv::imshow("optical flow", draw);
	cv::waitKey(0);

	return 0;
}