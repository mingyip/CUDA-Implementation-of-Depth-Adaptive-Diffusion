// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################
#include "helper.cuh"

#include <cstdlib>
#include <iostream>
#include <sstream>

// OpenCV image conversion: layered to interleaved
void convertLayeredToInterleaved(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (!aOut || !aIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    if (nc == 1)
    {
        memcpy(aOut, aIn, w*h*sizeof(float));
        return;
    }

    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[(nc-1-c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
            }
        }
    }
}
void convertLayeredToMat(cv::Mat &mOut, const float *aIn)
{
    convertLayeredToInterleaved((float*)mOut.data, aIn, mOut.cols, mOut.rows, mOut.channels());
}


// OpenCV image conversion: interleaved to layered
void convertInterleavedToLayered(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (!aOut || !aIn)
    {
        std::cerr << "arrays not allocated!" << std::endl;
        return;
    }

    if (nc == 1)
    {
        memcpy(aOut, aIn, w*h*sizeof(float));
        return;
    }

    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + (size_t)w*y)];
            }
        }
    }
}

void convertMatToLayered(float *aOut, const cv::Mat &mIn)
{
    convertInterleavedToLayered(aOut, (float*)mIn.data, mIn.cols, mIn.rows, mIn.channels());
}


// show cv:Mat in OpenCV GUI
void showImage(std::string title, const cv::Mat &mat, int x, int y)
{
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_AUTOSIZE);
    cv::moveWindow(wTitle, x, y);
    cv::imshow(wTitle, mat);
}




