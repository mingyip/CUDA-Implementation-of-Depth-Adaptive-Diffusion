#include "cuda_utils.cuh"

#include <sstream>

void cuda_check(const std::string &file, int line)
{
    static std::string prev_file;
    static int prev_line = 0;

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        std::stringstream ss;
        ss << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")\n";
        if (prev_line > 0)
            ss << "Previous CUDA call:\n" << prev_file << ", line " << prev_line << "\n";
        throw std::runtime_error(ss.str());
    }

    prev_file = file;
    prev_line = line;
}
