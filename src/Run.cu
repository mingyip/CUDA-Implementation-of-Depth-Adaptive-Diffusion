// ########################################################################
// Practical Course: GPU Programming in Computer Vision
// Technical University of Munich, Computer Vision Group
// ########################################################################

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "helper.cuh"
#include "lib/cuda/cuda_utils.cuh"
#include "lib/cuda/compute_rho.cuh"
#include "lib/cuda/construct_image_from_phi.cuh"
#include "lib/cuda/diffusion.cuh"
#include "lib/cuda/iterate_update.cuh"
#include "lib/cuda/update_p.cuh"
#include "lib/cuda/update_phi.cuh"

int main(int argc, char **argv)
{
    Timer total_timer;
    total_timer.start();
    // parse command line parameters
    const char *params = {
        "{i|image1| |input image1}"
        "{j|image2| |input image2}"
        "{w|bw|false|load input image as grayscale/black-white}"
        "{n|iter|1500|iterations}"
        "{e|eiter|50|energy iter}"
        "{l|lambda|50.0|lambda}"
        "{b|nblabels|25|nblabels}"
        "{g|gamma|1|step gamma}"
        "{m|bluriter|100|blur iter}"
        "{p|para|0.00002|stop parameter}"
        "{t|threshold|20|threshold}"};
    cv::CommandLineParser cmd(argc, argv, params);

    std::string inputImage1 = cmd.get<std::string>("image1");
    std::string inputImage2 = cmd.get<std::string>("image2");
    bool gray = cmd.get<bool>("bw");
    size_t max_iter = (size_t)cmd.get<int>("iter");
    std::cout << "max iterations: " << max_iter << std::endl;
    size_t energy_iter = (size_t)cmd.get<int>("eiter");
    std::cout << "energy calculation frequency: " << energy_iter << std::endl;
    float lambda = cmd.get<float>("lambda");
    std::cout << "lambda: " << lambda << std::endl;
    int nblabels = cmd.get<int>("nblabels");
    std::cout << "nblabels: " << nblabels << std::endl;
    float gamma = cmd.get<float>("gamma");
    std::cout << "gamma: " << gamma << std::endl;
    int bluriter = cmd.get<int>("bluriter");
    std::cout << "bluriter: " << bluriter << std::endl;
    float stop_para = cmd.get<float>("para");
    std::cout << "stop parameter: " << stop_para << std::endl;
    int threshold = cmd.get<int>("threshold");
    std::cout << "threshold: " << threshold << std::endl;

    float tau_d = 1 / sqrt(3);
    float tau_p = 1 / sqrt(3);

    // Read images
    cv::Mat mIn1;
    cv::Mat mIn2;
    mIn1 = cv::imread(inputImage1.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
    mIn2 = cv::imread(inputImage2.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
    if (mIn1.empty()) {
        std::cerr << "ERROR: Could not retrieve frame " << inputImage1 << std::endl;
        return 1;
    }
    if (mIn2.empty()) {
        std::cerr << "ERROR: Could not retrieve frame " << inputImage2 << std::endl;
        return 1;
    }
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn1.convertTo(mIn1, CV_32F, 1. / 255);
    mIn2.convertTo(mIn2, CV_32F, 1. / 255);

    // get image dimensions
    int w = mIn1.cols;        // width
    int h = mIn1.rows;        // height
    int nc = mIn1.channels(); // number of channels
    std::cout << "Image: " << w << " x " << h << std::endl;

    uint1 dim_1d = make_uint1(nblabels);
    uint2 dim_2d = make_uint2(w, h);
    uint3 dim_3d1 = make_uint3(w, h, nc);
    uint3 dim_3d2 = make_uint3(w, h, nblabels);
    uint4 dim_4d = make_uint4(w, h, nblabels, 3);

    // initialize CUDA contextkernelGaussTensor
    cudaDeviceSynchronize();
    CUDA_CHECK;

    // ### Set the output image format
    cv::Mat mOut(h, w, CV_32FC1);

    // ### Allocate arrays
    float *imgIn1 = imgIn1 = new float[w * h * nc];
    float *imgIn2 = imgIn2 = new float[w * h * nc];
    float *imgRho = imgRho = new float[w * h * nblabels];
    float *imgPhi = imgPhi = new float[w * h * nblabels]();
    float *imgP = imgP = new float[w * h * nblabels * 3];
    float *imgOut = imgOut = new float[w * h];

    // init raw input image array (and convert to layered)
    convertMatToLayered(imgIn1, mIn1);
    convertMatToLayered(imgIn2, mIn2);

    // declare view3d objects
    View3D<float> Il(imgIn1, dim_3d1);
    View3D<float> Ir(imgIn2, dim_3d1);
    View3D<float> rho(imgRho, dim_3d2);
    View2D<float> image(imgOut, dim_2d);
    View3D<float> Phi(imgPhi, dim_3d2);
    View4D<float> P(imgP, dim_4d);
    View3D<float> Phi_iter(imgPhi, dim_3d2);
    View4D<float> P_iter(imgP, dim_4d);

    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++)
            imgPhi[x + w * y] = 1;

    // Compute Rho
    compute_rho(Il, Ir, w, h, nblabels, gamma, lambda, rho);
    cudaThreadSynchronize();

    // Iteratively Update phi and p until converged or reached the maximum iteration.
    Timer timer;
    timer.start();
    iterate_update(Phi, P, rho, tau_d, tau_p, gamma, Phi_iter, P_iter, max_iter, energy_iter, stop_para);
    cudaThreadSynchronize();
    timer.end();
    std::cout << "Time taken for iterate_update: " << timer.get() * 1000 << " ms" << std::endl;

    // Construct A Disparity Image From Phi.
    construct_image_from_phi(Phi, image, nblabels, gamma);
    cudaThreadSynchronize();
    convertLayeredToMat(mOut, imgOut);
    showImage("Phi", mOut / nblabels, 100 + w, 100);
    cv::imwrite("Phi_result.png", mOut / nblabels * 255.f);

    // Diffusion.
    // Pixel with disparity less than theshold (background) is assigned with diffusion value 1
    // Pixel with high disparity value (foreground) is assigned with diffusion value 0 and therefore, no diffusion
    float *imgWeights = new float[nblabels]();
    for (int i = 0; i < nblabels; i++)
        if (i < threshold)
            imgWeights[i] = 1;
    View1D<float> weights(imgWeights, dim_1d);
    test_diffusion(Il, Phi, weights, bluriter, Ir);

    // Show Blurred Image
    convertLayeredToMat(mIn2, imgIn2);
    showImage("Blurred Image", mIn2, 100 + 2 * w, 100);
    total_timer.end();
    std::cout << "Total time lapse: " << total_timer.get() * 1000 << " ms" << std::endl;
    cv::imwrite("image_result.png", mIn2 * 255.f);
    cv::waitKey();

    delete[] imgIn1;
    delete[] imgIn2;
    delete[] imgRho;
    delete[] imgPhi;
    delete[] imgP;
    delete[] imgOut;
    delete[] imgWeights;

    // close all opencv windows
    cv::destroyAllWindows();

    return 0;
}
