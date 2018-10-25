// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#ifndef TUM_HELPER_CUH
#define TUM_HELPER_CUH

#include <cuda_runtime.h>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>


// CUDA utility functions

// OpenCV image conversion
// interleaved to layered
void convertMatToLayered(float *aOut, const cv::Mat &mIn);

// layered to interleaved
void convertLayeredToMat(cv::Mat &mOut, const float *aIn);

// show image
void showImage(std::string title, const cv::Mat &mat, int x, int y);

// measuring time
class Timer
{
public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
        cudaDeviceSynchronize();
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
private:
	clock_t tStart;
	bool running;
	float sec;
};

#endif
