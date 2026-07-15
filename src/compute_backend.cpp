#include "vectorforge/compute_backend.h"
#include "vectorforge/cpu_backend.h"

#ifdef VECTORFORGE_HAS_CUDA
#include "vectorforge/cuda_backend.h"
#endif

namespace vectorforge {

std::unique_ptr<ComputeBackend> create_backend(const std::string& device) {
#ifdef VECTORFORGE_HAS_CUDA
    if (device == "cuda" || device == "gpu") {
        return std::make_unique<CUDABackend>();
    }
#endif
    // Fall back to CPU if requested, or if CUDA isn't compiled in
    return std::make_unique<CPUBackend>();
}

}
