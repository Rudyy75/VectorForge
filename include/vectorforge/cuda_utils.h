#pragma once

#ifdef VECTORFORGE_HAS_CUDA

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <iostream>

namespace vectorforge {
namespace cuda {

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::string error_msg = std::string("CUDA error in ") + __FILE__ + ":" + std::to_string(__LINE__) + \
                                    " - " + cudaGetErrorString(err); \
            throw std::runtime_error(error_msg); \
        } \
    } while (0)

// RAII wrapper for GPU memory (VRAM)
template <typename T>
class CudaDeviceBuffer {
public:
    // Allocate GPU memory
    explicit CudaDeviceBuffer(size_t count) : count_(count), ptr_(nullptr) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

    // Free GPU memory automatically when the object goes out of scope
    ~CudaDeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Prevent copying (we don't want two objects trying to free the same memory)
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    // Allow moving
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept : count_(other.count_), ptr_(other.ptr_) {
        other.count_ = 0;
        other.ptr_ = nullptr;
    }

    // Upload data from CPU (RAM) to GPU (VRAM)
    void upload(const T* host_ptr, size_t count, cudaStream_t stream = 0) {
        if (count > count_) throw std::runtime_error("Upload count exceeds buffer size");
        if (count > 0) {
            if (stream == 0) {
                CUDA_CHECK(cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream));
            }
        }
    }

    // Download data from GPU (VRAM) back to CPU (RAM)
    void download(T* host_ptr, size_t count, cudaStream_t stream = 0) const {
        if (count > count_) throw std::runtime_error("Download count exceeds buffer size");
        if (count > 0) {
            if (stream == 0) {
                CUDA_CHECK(cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
            }
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }

private:
    size_t count_;
    T* ptr_;
};

} 
} 

#endif 
