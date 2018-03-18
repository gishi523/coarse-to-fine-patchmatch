#ifndef __COARSE_TO_FINE_PATCHMATCH_H__
#define __COARSE_TO_FINE_PATCHMATCH_H__

#include <opencv2/opencv.hpp>

/** @brief Coarse-to-Fine PatchMatch class.

The class implements the Coarse-to-Fine PatchMatch Optical Flow
described in "Yinlin Hu: Efficient Coarse-to-Fine PatchMatch for Large Displacement Optical Flow".
*/
class CoarseToFinePatchMatch
{
public:

	struct Parameters
	{
		int step;
		int maxIters;
		float stopIterRatio;
		float scaleStep;
		int maxDisp;
		int checkTh;
		int borderWidth;

		// default settings
		Parameters()
		{
			step = 3;
			maxIters = 8;
			scaleStep = 0.5f;
			stopIterRatio = 0.05f;
			maxDisp = 400;
			checkTh = 3;
			borderWidth = 5;
		}
	};

	CoarseToFinePatchMatch(const Parameters& param = Parameters());

	/** @brief Calculates an optical flow using the Coarse-to-Fine PatchMatch
	@param I1 first 8-bit input image.
	@param I2 second input image of the same size and the same type as I1.
	@return output Nx1 vector of correspondances; each row represents "x1,y1,x2,y2".
	*/
	cv::Mat4f compute(const cv::Mat& I1, const cv::Mat& I2);

private:
	Parameters param_;
};

#endif // !__COARSE_TO_FINE_PATCHMATCH_H__