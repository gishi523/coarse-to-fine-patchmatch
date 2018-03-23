#include "coarse_to_fine_patchmatch.h"
#include <opencv2/xfeatures2d.hpp>

#ifdef WITH_SSE
#include <emmintrin.h>
#endif

static const int NUM_NEIGHBORS = 8;
static const int NEIGHBOR_DX[NUM_NEIGHBORS] = { 0, 0, 1, -1, -1, -1, 1, 1 };
static const int NEIGHBOR_DY[NUM_NEIGHBORS] = { -1, 1, 0, 0, -1, 1, -1, 1 };

template <class T>
static inline const T& clamp(const T& v, const T& lo, const T& hi)
{
	return std::max(lo, std::min(v, hi));
}

static std::vector<double> makeScales(double scaleStep, int nscales)
{
	std::vector<double> scales(nscales);
	double scale = 1;
	for (int i = 0; i < nscales; i++)
	{
		scales[i] = scale;
		scale *= scaleStep;
	}
	return scales;
}

static void GaussianPyrDown(const cv::Mat& src, cv::Mat& dst, double sigma, double scale)
{
	cv::Mat tmp;
	cv::GaussianBlur(src, tmp, cv::Size(), sigma);
	cv::resize(tmp, dst, cv::Size(), scale, scale);
}

static void constructPyramid(const cv::Mat& img, std::vector<cv::Mat>& pyramid, float ratio, int minWidth)
{
	// the ratio cannot be arbitrary numbers
	if (ratio > 0.98f || ratio < 0.4f)
		ratio = 0.75f;

	// first decide how many levels
	const int nscales = static_cast<int>(log(1. * minWidth / img.cols) / log(ratio));
	const std::vector<double> scales = makeScales(ratio, nscales);

	pyramid.resize(nscales);
	img.copyTo(pyramid[0]);

	const double sigma0 = (1 / ratio - 1);
	const int n = static_cast<int>(log(0.25) / log(ratio));
	for (int s = 1; s < nscales; s++)
	{
		const double sigma = s <= n ? s * sigma0 : n * sigma0;
		const double scale = s <= n ? scales[s] : scales[n];
		const cv::Mat& src = s <= n ? img : pyramid[s - n];
		GaussianPyrDown(src, pyramid[s], sigma, scale);
	}
}

static int makeSeedsAndNeighbors(int w, int h, int step, cv::Mat2f& seeds, cv::Mat1i& neighbors)
{
	const int gridw = w / step;
	const int gridh = h / step;
	const int nseeds = gridw * gridh;

	const int ofsx = (w - (gridw - 1) * step) / 2;
	const int ofsy = (h - (gridh - 1) * step) / 2;

	seeds.create(nseeds, 1);
	neighbors.create(nseeds, NUM_NEIGHBORS);
	neighbors = -1;

	for (int i = 0; i < nseeds; i++)
	{
		const int x = i % gridw;
		const int y = i / gridw;

		const float seedx = static_cast<float>(x * step + ofsx);
		const float seedy = static_cast<float>(y * step + ofsy);
		seeds(i) = cv::Vec2f(seedx, seedy);

		int nbidx = 0;
		for (int n = 0; n < NUM_NEIGHBORS; n++)
		{
			const int nbx = x + NEIGHBOR_DX[n];
			const int nby = y + NEIGHBOR_DY[n];
			if (nbx < 0 || nbx >= gridw || nby < 0 || nby >= gridh)
				continue;
			neighbors(i, nbidx++) = nby * gridw + nbx;
		}
	}

	return nseeds;
}

static cv::Point2f intersection(const cv::Point2f& u1, const cv::Point2f& u2, const cv::Point2f& v1, const cv::Point2f& v2)
{
	cv::Point2d ans = u1;
	const double t = ((u1.x - v1.x) * (v1.y - v2.y) - (u1.y - v1.y) * (v1.x - v2.x)) /
		((u1.x - u2.x) * (v1.y - v2.y) - (u1.y - u2.y) * (v1.x - v2.x));
	ans.x += (u2.x - u1.x) * t;
	ans.y += (u2.y - u1.y) * t;
	return cv::Point2f(ans);
}

// circle center containing a triangular
static cv::Point2f circumcenter(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c)
{
	cv::Point2f ua, ub, va, vb;
	ua.x = (a.x + b.x) / 2;
	ua.y = (a.y + b.y) / 2;
	ub.x = ua.x - a.y + b.y;
	ub.y = ua.y + a.x - b.x;
	va.x = (a.x + c.x) / 2;
	va.y = (a.y + c.y) / 2;
	vb.x = va.x - a.y + c.y;
	vb.y = va.y + a.x - c.x;
	return intersection(ua, ub, va, vb);
}

static double dist(const cv::Point2f& p1, const cv::Point2f& p2)
{
	return cv::norm(p2 - p1);
}

static float minimalCircle(const cv::Point2f p[], int n)
{
	const double eps = 1e-6;

	// center and radius of the circle
	cv::Point2f o;
	double r;

	int i, j, k;
	o = p[0];
	r = 0;
	for (i = 1; i < n; i++)
	{
		if (dist(p[i], o) - r > eps)
		{
			o = p[i];
			r = 0;

			for (j = 0; j < i; j++)
			{
				if (dist(p[j], o) - r > eps)
				{
					o.x = 0.5f * (p[i].x + p[j].x);
					o.y = 0.5f * (p[i].y + p[j].y);

					r = dist(o, p[j]);

					for (k = 0; k < j; k++)
					{
						if (dist(o, p[k]) - r > eps)
						{
							o = circumcenter(p[i], p[j], p[k]);
							r = dist(o, p[k]);
						}
					}
				}
			}
		}
	}
	return static_cast<float>(r);
}

static void updateSearchRadius(const cv::Mat2f& flow, const cv::Mat1i& neighbors, cv::Mat1f& searchRadius)
{
	const int nseeds = flow.rows;
	cv::Point2f flows[NUM_NEIGHBORS + 1];
	for (int i = 0; i < nseeds; i++)
	{
		flows[0] = flow(i);

		int count = 1;
		for (int n = 0; n < NUM_NEIGHBORS && neighbors(i, n) >= 0; n++)
			flows[count++] = flow(neighbors(i, n));

		searchRadius(i) = minimalCircle(flows, count);
	}
}

struct MatchingCost
{
	MatchingCost(const cv::Mat& desc1, const cv::Mat& desc2, const cv::Size& imgSize)
		: desc1(desc1), desc2(desc2), w(imgSize.width), h(imgSize.height), ch(desc1.cols)
	{
		CV_Assert(desc1.type() == CV_8U && desc2.type() == CV_8U);
		CV_Assert(desc1.size() == desc2.size());
#ifdef WITH_SSE
		CV_Assert(ch * 255 < std::numeric_limits<ushort>::max()); // confirm maximum SAD does not exceed 16bit
#endif
	}

	inline int compute(const cv::Vec2f& pt, const cv::Vec2f& flow) const
	{
		const int x1 = clamp(static_cast<int>(pt[0] + 0.5f), 0, w - 1);
		const int y1 = clamp(static_cast<int>(pt[1] + 0.5f), 0, h - 1);
		const int x2 = clamp(static_cast<int>(pt[0] + flow[0] + 0.5f), 0, w - 1);
		const int y2 = clamp(static_cast<int>(pt[1] + flow[1] + 0.5f), 0, h - 1);

		const int idx1 = y1 * w + x1;
		const int idx2 = y2 * w + x2;

		const uchar* ptr1 = desc1.ptr<uchar>(idx1);
		const uchar* ptr2 = desc2.ptr<uchar>(idx2);

#ifdef WITH_SSE
		const __m128i* vptr1 = reinterpret_cast<const __m128i*>(ptr1);
		const __m128i* vptr2 = reinterpret_cast<const __m128i*>(ptr2);
		__m128i vsum = _mm_setzero_si128();
		for (int i = 0; i < ch / 16; i++)
		{
			const __m128i v1 = _mm_loadu_si128(vptr1++);
			const __m128i v2 = _mm_loadu_si128(vptr2++);
			const __m128i vsad = _mm_sad_epu8(v1, v2);
			vsum = _mm_adds_epu16(vsum, vsad);
		}

		int diff = _mm_extract_epi16(vsum, 0) + _mm_extract_epi16(vsum, 4);
		for (int i = (ch & ~0x0f); i < ch; i++)
			diff += std::abs(ptr1[i] - ptr2[i]);
#else
		int diff = 0;
		for (int i = 0; i < ch; i++)
			diff += std::abs(ptr1[i] - ptr2[i]);
#endif

		return diff;
	}

	const cv::Mat& desc1;
	const cv::Mat& desc2;
	int w, h, ch;
};

class RandomFlow
{
public:
	RandomFlow(uint64 seed = 0) : rng_(seed) {}
	inline cv::Vec2f generate(int radius)
	{
		const float x = static_cast<float>(rng_.uniform(-radius, radius + 1));
		const float y = static_cast<float>(rng_.uniform(-radius, radius + 1));
		return cv::Vec2f(x, y);
	}
private:
	cv::RNG rng_;
};

class Propagation
{
public:

	Propagation(int maxIters = 8, double stopIterRatio = 0.05) : maxIters_(maxIters), stopIterRatio_(stopIterRatio) {}

	void operator()(cv::Mat2f& flow, const cv::Mat& desc1, const cv::Mat& desc2, const cv::Size& imgSize,
		const cv::Mat2f& seeds, const cv::Mat1i& neighbors, const cv::Mat1f& radius)
	{
		CV_Assert(desc1.type() == CV_8U && desc2.type() == CV_8U);

		const double eps = 1e-6;
		const int nseeds = seeds.rows;

		RandomFlow rf;
		MatchingCost C(desc1, desc2, imgSize);
		std::vector<int> visited(nseeds), bestCosts(nseeds);

		// init cost
		for (int i = 0; i < nseeds; i++)
			bestCosts[i] = C.compute(seeds(i), flow(i));

		double lastUpdateRatio = 2;
		for (int iter = 0; iter < maxIters_; iter++)
		{
			int updateCount = 0;
			memset(visited.data(), 0, sizeof(int) * nseeds);

			int i0 = 0, i1 = nseeds, step = 1;
			if (iter % 2 == 1)
			{
				i0 = nseeds - 1; i1 = -1; step = -1;
			}

			for (int ic = i0; ic != i1; ic += step)
			{
				bool updated = false;
				const cv::Vec2f pt = seeds(ic);

				// Propagation: Improve current guess by trying instead correspondences from neighbors
				for (int n = 0; n < NUM_NEIGHBORS && neighbors(ic, n) >= 0; n++)
				{
					const int in = neighbors(ic, n);
					if (!visited[in])
						continue;

					const cv::Vec2f fc = flow(ic);
					const cv::Vec2f ft = flow(in);
					const cv::Vec2f df = ft - fc;
					if (fabs(df[0]) < eps && fabs(df[1]) < eps)
						continue;

					const int cost = C.compute(pt, ft);
					if (cost < bestCosts[ic])
					{
						bestCosts[ic] = cost;
						flow(ic) = ft;
						updated = true;
					}
				}

				// Random search: Improve current guess by searching in boxes
				// of exponentially decreasing size around the current best guess.
				for (int mag = cvRound(radius(ic)); mag >= 1; mag /= 2)
				{
					/* Sampling window */
					const cv::Vec2f fc = flow(ic);
					const cv::Vec2f ft = fc + rf.generate(mag);
					const cv::Vec2f df = ft - fc;
					if (fabs(df[0]) < eps && fabs(df[1]) < eps)
						continue;

					const int cost = C.compute(pt, ft);
					if (cost < bestCosts[ic])
					{
						bestCosts[ic] = cost;
						flow(ic) = ft;
						updated = true;
					}
				}

				visited[ic] = 1;
				if (updated)
					updateCount++;

			}

			const double updateRatio = 1. * updateCount / nseeds;
			if (updateRatio < stopIterRatio_ || lastUpdateRatio - updateRatio < 0.01)
				break;

			lastUpdateRatio = updateRatio;
		}
	}

private:
	int maxIters_;
	double stopIterRatio_;
};

class PyramidRandomSearch
{

public:

	using Parameters = CoarseToFinePatchMatch::Parameters;

	PyramidRandomSearch(const std::vector<cv::Mat>& I1s, const std::vector<cv::Mat>& I2s, const cv::Mat2f& seeds,
		const cv::Mat1i& neighbors, const Parameters& param = Parameters()) : neighbors_(neighbors), param_(param)
	{
		init(I1s, I2s, seeds);
	}

	void init(const std::vector<cv::Mat>& I1s, const std::vector<cv::Mat>& I2s, const cv::Mat2f& seeds)
	{
		const int nseeds = seeds.rows;
		const int nscales = static_cast<int>(I1s.size());
		const std::vector<double> scales = makeScales(param_.scaleStep, nscales);

		seeds_.assign(nscales, cv::Mat2f(nseeds, 1));
		sizes_.assign(nscales, cv::Size());
		maxRadius_.assign(nscales, 0);

		for (int s = 0; s < nscales; s++)
		{
			const int w = I1s[s].cols;
			const int h = I1s[s].rows;
			seeds_[s] = cv::Mat2f(scales[s] * seeds);
			sizes_[s] = cv::Size(w, h);
			maxRadius_[s] = std::min(cvRound(param_.maxDisp * scales[s]), 32);
		}

		initRadius_ = cvRound(param_.maxDisp * scales[nscales - 1]);
	}

	cv::Mat2f compute(const std::vector<cv::Mat>& desc1, const std::vector<cv::Mat>& desc2)
	{
		const int nscales = static_cast<int>(desc1.size());
		const int nseeds = seeds_[0].rows;

		// random Initialization on coarsest level
		RandomFlow rf;
		cv::Mat2f flow(nseeds, 1);
		for (int i = 0; i < nseeds; i++)
			flow(i) = rf.generate(initRadius_);

		// set the radius of coarsest level
		cv::Mat1f radius(nseeds, 1);
		radius = static_cast<float>(initRadius_);

		Propagation propagate(param_.maxIters, param_.stopIterRatio);

		// coarse-to-fine
		for (int s = nscales - 1; s >= 0; s--)
		{
			propagate(flow, desc1[s], desc2[s], sizes_[s], seeds_[s], neighbors_, radius);
			if (s > 0)
			{
				updateSearchRadius(flow, neighbors_, radius);

				// scale the radius accordingly
				radius = cv::max(1, cv::min(radius, maxRadius_[s]));
				radius *= (1. / param_.scaleStep);
				flow *= (1. / param_.scaleStep);
			}
		}

		return flow;
	}

private:
	std::vector<cv::Mat2f> seeds_;
	std::vector<cv::Size> sizes_;
	std::vector<int> maxRadius_;
	cv::Mat1i neighbors_;
	int initRadius_;
	Parameters param_;
};

static int crossCheck(const cv::Mat2f& seeds, const cv::Mat2f& flow1, const cv::Mat2f& flow2, std::vector<int>& valid,
	int w, int h, int step, int maxDisp, int checkThreshold, int borderWidth)
{
	const int nseeds = seeds.rows;
	const int radius = step / 2;
	const cv::Rect region(borderWidth, borderWidth, w - 2 * borderWidth, h - 2 * borderWidth);

	cv::Mat1i labels(h, w);
	for (int i = 0; i < nseeds; i++)
	{
		const int x0 = static_cast<int>(seeds(i)[0]);
		const int y0 = static_cast<int>(seeds(i)[1]);
		for (int dy = -radius; dy <= radius; dy++)
		{
			for (int dx = -radius; dx <= radius; dx++)
			{
				const int x = clamp(x0 + dx, 0, w - 1);
				const int y = clamp(y0 + dy, 0, h - 1);
				labels(y, x) = i;
			}
		}
	}

	valid.assign(nseeds, 1);
	int nvalids = 0;
	for (int i = 0; i < nseeds; i++)
	{
		const cv::Vec2f f1 = flow1(i);
		const cv::Point pt1(seeds(i));
		const cv::Point pt2(seeds(i) + f1);

		if (!region.contains(pt1) || !region.contains(pt2) || cv::norm(f1) > maxDisp)
		{
			valid[i] = 0;
			continue;
		}

		const cv::Vec2f f2 = flow2(labels(pt2));
		if (cv::norm(f1 + f2) > checkThreshold)
		{
			valid[i] = 0;
			continue;
		}

		nvalids++;
	}

	return nvalids;
}

CoarseToFinePatchMatch::CoarseToFinePatchMatch(const Parameters& param) : param_(param)
{
}

cv::Mat4f CoarseToFinePatchMatch::compute(const cv::Mat& I1, const cv::Mat& I2)
{
	CV_Assert(I1.size() == I2.size() && I1.type() == CV_8U && I2.type() == CV_8U);

	const int h = I1.rows;
	const int w = I1.cols;
	const int step = param_.step;

	// construct pyramid
	std::vector<cv::Mat> I1s, I2s;
	constructPyramid(I1, I1s, param_.scaleStep, 30);
	constructPyramid(I2, I2s, param_.scaleStep, 30);
	const int nscales = static_cast<int>(I1s.size());

	// get feature
	auto daisy = cv::xfeatures2d::DAISY::create(5, 3, 4, 8, cv::xfeatures2d::DAISY::NRM_FULL, cv::noArray(), false, false);
	std::vector<cv::Mat> desc1(nscales), desc2(nscales);
	for (int s = 0; s < nscales; s++)
	{
		daisy->compute(I1s[s], desc1[s]);
		daisy->compute(I2s[s], desc2[s]);
		desc1[s].convertTo(desc1[s], CV_8U, 255);
		desc2[s].convertTo(desc2[s], CV_8U, 255);
	}

	// make seeds and neighbors
	cv::Mat2f seeds;
	cv::Mat1i neighbors;
	const int nseeds = makeSeedsAndNeighbors(w, h, step, seeds, neighbors);

	// pyramid random search
	PyramidRandomSearch search(I1s, I2s, seeds, neighbors, param_);
	const cv::Mat2f flow1 = search.compute(desc1, desc2);
	const cv::Mat2f flow2 = search.compute(desc2, desc1);

	// cross check
	std::vector<int> valid;
	const int nmatches = crossCheck(seeds, flow1, flow2, valid, w, h, step, param_.maxDisp, param_.checkTh, param_.borderWidth);

	// flow to match
	cv::Mat4f matches(nmatches, 1);
	int idx = 0;
	for (int i = 0; i < nseeds; i++)
	{
		if (valid[i])
		{
			matches(idx)[0] = seeds(i)[0];
			matches(idx)[1] = seeds(i)[1];
			matches(idx)[2] = seeds(i)[0] + flow1(i)[0];
			matches(idx)[3] = seeds(i)[1] + flow1(i)[1];
			idx++;
		}
	}

	return matches;
}