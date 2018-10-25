#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <stdexcept>
#include <string>

#include <assert.h>
#include <cuda_runtime.h>

//MATLAB index mapping
/*
#define C2D2IDX(_x, _y, dims) ((_y) + (_x) * (dims).y)
#define C3D2IDX(_x, _y, _z, dims) ((_y) + (_x) * (dims).y + (_z) * (dims).y * (dims).x)
#define C4D2IDX(_x, _y, _z, _c, dims) ((_y) + (_x) * (dims).y + (_z) * (dims).y * (dims).x + (_c) * (dims).y * (dims).x * (dims).z)
/*/
#define C2D2IDX(_x, _y, dims) ((_x) + (_y) * (dims).x)
#define C3D2IDX(_x, _y, _z, dims) ((_x) + (_y) * (dims).x + (_z) * (dims).y * (dims).x)
#define C4D2IDX(_x, _y, _z, _c, dims) ((_x) + (_y) * (dims).x + (_z) * (dims).y * (dims).x + (_c) * (dims).y * (dims).x * (dims).z)
//*/

//Switch to disable assertions.
//*
#define DEBUG_ASSERT(cond) ((void) 0)
/*/
#define DEBUG_ASSERT(cond) (assert(cond))
//*/

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(const std::string &file, int line);

// compute grid size from block size
inline
dim3 computeGrid1D(const dim3 &block, const int w)
{
    return dim3((w + block.x - 1) / block.x, 1, 1);
}

inline
dim3 computeGrid2D(const dim3 &block, const int w, const int h)
{
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
}

inline
dim3 computeGrid3D(const dim3 &block, const int w, const int h, const int s)
{
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, (s + block.z - 1) / block.z);
}

__host__
inline
void cudaFreeChecked(void * ptr) {
    cudaFree(ptr);
    CUDA_CHECK;
}

// Wrapper class to measure kernel run-time based on cuda events.  
class CudaTimer {
public:
    
    CudaTimer()
    {
        cudaEventCreate(&_start);
        cudaEventRecord(_start,0);
        CUDA_CHECK;
    }

    float stop() {
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        CUDA_CHECK;

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, _start, stop);
        return elapsed_time;
    }

private:
    cudaEvent_t _start;
};

/**
 * Wrapper class for global device memory.
 */
template<typename T>
class CudaGlobalMem {
public:
    CudaGlobalMem() = delete;

    //Disable copy and move constructors to avoid calling cudaFree multiple times.
    CudaGlobalMem(const CudaGlobalMem &) = delete;
    CudaGlobalMem & operator=(const CudaGlobalMem &) = delete;

    CudaGlobalMem(CudaGlobalMem &&) = delete;
    CudaGlobalMem & operator=(CudaGlobalMem &&) = delete;

    //Takes a number of elements and allocates sufficiently large memory based on T.
    CudaGlobalMem(size_t num_elems)
    : size(num_elems)
    {
        cudaMalloc(&device_ptr, sizeof(T) * size);
        CUDA_CHECK;
    }

    //Sorthand for std::vector
    template<typename ContainerT>
    CudaGlobalMem(const ContainerT & host_mem)
    : CudaGlobalMem(host_mem.size())
    {}

    //Destructor releasing the device memory
    virtual ~CudaGlobalMem() {
        cudaFree(device_ptr);
        CUDA_CHECK;
    }

    //Returns a const raw pointer to the device memory
    const T * get() const {
        return device_ptr;
    }

    //Returns a const raw pointer to the device memory
    T * get() {
        return device_ptr;
    }

    //Takes a pointer to host memory and copies it to the device memory
    void copyToDevice(const T * host_ptr) {
        cudaMemcpy(device_ptr, host_ptr, sizeof(T) * size, cudaMemcpyHostToDevice);
        CUDA_CHECK;
    }

    //Shorthand for std::vector that also checks if the host memory fits.
    template<typename ContainerT>
    void copyToDevice(const ContainerT &host_mem) {
        if (host_mem.size() != size) {
            throw std::length_error("Memory sizes do not match. gpu: " + std::to_string(size) + ", host: " + std::to_string(host_mem.size()));
        }

        copyToDevice(host_mem.data());
    }

    //Takes a pointer to host memory and copies the device memory to it.
    void copyFromDevice(T * host_ptr) const {
        cudaMemcpy(host_ptr, device_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
        CUDA_CHECK;
    }

    //Shorthand for std::vector that also checks if the device memory fits.
    template<typename ContainerT>
    void copyFromDevice(ContainerT &host_mem) const {
        if (host_mem.size() != size) {
            throw std::length_error("Memory sizes do not match. gpu: " + std::to_string(size) + ", host: " + std::to_string(host_mem.size()));
        }

        copyFromDevice(host_mem.data());
    }

private:
    size_t size;
    T * device_ptr;
};


__host__ __device__
inline
bool in_range(uint x, uint lower, uint upper) {
    return lower <= x && x < upper;
}


// View class that allows memory look-ups for 1D arrays. It can be used in host and device functions.
template<typename T>
__host__ __device__
class View1D {
public:
    //Constructor taking pointer to memory and array dimensions.
    __host__ __device__
    View1D(T *ptr, const uint1 &dims)
    : ptr(ptr)
    , dims_(dims)
    {}

    //Returns the array dimensions.
    __host__ __device__
    const uint1 & dims() const {
        return dims_;
    }

    //Returns the number of array elements.
    __host__ __device__
    size_t size() const {
        return dims_.x;
    }

    //Returns the memory pointer.
    T * data() {
        return ptr;
    }

    //Returns the memory pointer.
    const T * data() const {
        return ptr;
    }

    //Returns the element value at the given coordinates.
    __host__ __device__
    const T& operator()(uint x) const {
        DEBUG_ASSERT(in_range(x, 0, dims_.x));

        return ptr[x];
    }

    //Returns the element value at the given coordinates.
    __host__ __device__
    T& operator()(uint x) {
        DEBUG_ASSERT(in_range(x, 0, dims_.x));

        return ptr[x];
    }

private:
    T * ptr;
    uint1 dims_;
};


// View class that allows memory look-ups for 2D arrays. It can be used in host and device functions.
template<typename T>
__host__ __device__
class View2D {
public:
    //Constructor taking pointer to memory and array dimensions.
    __host__ __device__
    View2D(T *ptr, const uint2 &dims)
    : ptr(ptr)
    , dims_(dims)
    {}

    //Returns the array dimensions.
    __host__ __device__
    const uint2 & dims() const {
        return dims_;
    }

    //Returns the number of array elements.
    __host__ __device__
    size_t size() const {
        return dims_.x * dims_.y;
    }

    //Returns the memory pointer.
    T * data() {
        return ptr;
    }
    
    //Returns the memory pointer.
    const T * data() const {
        return ptr;
    }

    //Returns the element value at the given coordinates.
    __host__ __device__
    const T& operator()(uint x, uint y) const {
        DEBUG_ASSERT(in_range(x, 0, dims_.x));
        DEBUG_ASSERT(in_range(y, 0, dims_.y));

        return ptr[C2D2IDX(x, y, dims_)];
    }

    //Returns the element value at the given coordinates.
    __host__ __device__
    T& operator()(uint x, uint y) {
        DEBUG_ASSERT(in_range(x, 0, dims_.x));
        DEBUG_ASSERT(in_range(y, 0, dims_.y));

        return ptr[C2D2IDX(x, y, dims_)];
    }

private:
    T * ptr;
    uint2 dims_;
};


// View class that allows memory look-ups for 3D arrays. It can be used in host and device functions.
template<typename T>
__host__ __device__
class View3D {
public:
    //Constructor taking pointer to memory and array dimensions.
    __host__ __device__
    View3D(T *ptr, const uint3 &dims)
    : ptr(ptr)
    , dims_(dims)
    {}

    //Returns the array dimensions.
    __host__ __device__
    const uint3 & dims() const {
        return dims_;
    }

    //Returns the number of array elements.
    __host__ __device__
    size_t size() const {
        return dims_.x * dims_.y * dims_.z;
    }

    //Returns the memory pointer.
    T * data() {
        return ptr;
    }

    //Returns the memory pointer.
    const T * data() const {
        return ptr;
    }

    //Returns the element value at the given coordinates.
    __host__ __device__
    const T& operator()(uint x, uint y, uint z) const {
        DEBUG_ASSERT(in_range(x, 0, dims_.x));
        DEBUG_ASSERT(in_range(y, 0, dims_.y));
        DEBUG_ASSERT(in_range(z, 0, dims_.z));

        return ptr[C3D2IDX(x, y, z, dims_)];
    }

    //Returns the element value at the given coordinates.
    __host__ __device__
    T& operator()(uint x, uint y, uint z) {
        DEBUG_ASSERT(in_range(x, 0, dims_.x));
        DEBUG_ASSERT(in_range(y, 0, dims_.y));
        DEBUG_ASSERT(in_range(z, 0, dims_.z));

        return ptr[C3D2IDX(x, y, z, dims_)];
    }

private:
    T * ptr;
    uint3 dims_;
};



// View class that allows memory look-ups for 4D arrays. It can be used in host and device functions.
template<typename T>
__host__ __device__
class View4D {
public:
    //Constructor taking pointer to memory and array dimensions.
    __host__ __device__
    View4D(T *ptr, const uint4 &dims)
    : ptr(ptr)
    , dims_(dims)
    {}

    //Returns the array dimensions.
    __host__ __device__
    const uint4 & dims() const {
        return dims_;
    }

    //Returns the number of array elements.
    __host__ __device__
    size_t size() const {
        return dims_.x * dims_.y * dims_.z * dims_.w;
    }

    //Returns the memory pointer.
    T * data() {
        return ptr;
    }

    //Returns the memory pointer.
    const T * data() const {
        return ptr;
    }

    //Returns the element value at the given coordinates.
    __host__ __device__
    const T& operator()(uint x, uint y, uint z, uint c) const {
        DEBUG_ASSERT(in_range(x, 0, dims_.x));
        DEBUG_ASSERT(in_range(y, 0, dims_.y));
        DEBUG_ASSERT(in_range(z, 0, dims_.z));
        DEBUG_ASSERT(in_range(c, 0, dims_.w));

        return ptr[C4D2IDX(x, y, z, c, dims_)];
    }

    //Returns the element value at the given coordinates.
    __host__ __device__
    T& operator()(uint x, uint y, uint z, uint c) {
        DEBUG_ASSERT(in_range(x, 0, dims_.x));
        DEBUG_ASSERT(in_range(y, 0, dims_.y));
        DEBUG_ASSERT(in_range(z, 0, dims_.z));
        DEBUG_ASSERT(in_range(c, 0, dims_.w));

        return ptr[C4D2IDX(x, y, z, c, dims_)];
    }

private:
    T * ptr;
    uint4 dims_;
};

#endif // CUDA_UTILS_CUH
