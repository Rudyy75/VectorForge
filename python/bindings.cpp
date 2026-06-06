#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "vectorforge/flat_index.h"
#include "vectorforge/ivf_index.h"
#include "vectorforge/types.h"
#include "vectorforge/thread_pool.h"
#include <stdexcept>

namespace py = pybind11;
using namespace vectorforge;

// Helper to convert py::array_t to raw float*
template<typename T>
const T* get_raw_ptr(const py::array_t<T>& arr, size_t expected_dim, const char* name) {
    py::buffer_info info = arr.request();
    if (info.ndim != 1 && info.ndim != 2) {
        throw std::runtime_error(std::string(name) + " must be a 1D or 2D NumPy array");
    }
    
    size_t total_elements = 1;
    for (int i = 0; i < info.ndim; ++i) {
        total_elements *= info.shape[i];
    }
    
    if (expected_dim > 0 && total_elements % expected_dim != 0) {
        throw std::runtime_error(std::string(name) + " dimension mismatch. Expected vectors of dim " + std::to_string(expected_dim));
    }
    
    return static_cast<const T*>(info.ptr);
}

template<typename T>
size_t get_num_vectors(const py::array_t<T>& arr, size_t dim) {
    py::buffer_info info = arr.request();
    size_t total_elements = 1;
    for (int i = 0; i < info.ndim; ++i) {
        total_elements *= info.shape[i];
    }
    return total_elements / dim;
}

PYBIND11_MODULE(vectorforge, m) {
    m.doc() = "VectorForge: High-Performance Vector Search Engine";

    py::enum_<MetricType>(m, "MetricType")
        .value("L2", MetricType::L2)
        .value("IP", MetricType::IP)
        .value("Cosine", MetricType::Cosine)
        .export_values();

    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("id", &SearchResult::id)
        .def_readonly("distance", &SearchResult::distance)
        .def("__repr__", [](const SearchResult& sr) {
            return "<SearchResult id=" + std::to_string(sr.id) + " dist=" + std::to_string(sr.distance) + ">";
        });
        
    py::class_<ThreadPool>(m, "ThreadPool")
        .def(py::init<size_t>(), py::arg("num_threads"));

    // FlatIndex
    py::class_<FlatIndex>(m, "FlatIndex")
        .def(py::init<size_t, MetricType>(), py::arg("dim"), py::arg("metric") = MetricType::L2)
        .def("add", [](FlatIndex& self, size_t id, py::array_t<float> vector) {
            const float* ptr = get_raw_ptr(vector, self.size() == 0 ? vector.size() : 0, "vector");
           
            py::buffer_info info = vector.request();
            if (info.ndim != 1) throw std::runtime_error("Vector must be 1D");
            self.add(id, static_cast<const float*>(info.ptr));
        }, py::arg("id"), py::arg("vector"))
        .def("search", [](const FlatIndex& self, py::array_t<float> query, size_t k) {
            py::buffer_info info = query.request();
            if (info.ndim != 1) throw std::runtime_error("Query must be 1D");
            
            std::vector<SearchResult> res;
            {
                py::gil_scoped_release release; // Release GIL for pure C++ execution
                res = self.search(static_cast<const float*>(info.ptr), k);
            }
            return res;
        }, py::arg("query"), py::arg("k"))
        .def("save", &FlatIndex::save, py::arg("filename"))
        .def("load", &FlatIndex::load, py::arg("filename"))
        .def("size", &FlatIndex::size);

    // IVFIndex
    py::class_<IVFIndex>(m, "IVFIndex")
        .def(py::init<size_t, size_t, MetricType>(), py::arg("dim"), py::arg("nlist"), py::arg("metric") = MetricType::L2)
        .def("train", [](IVFIndex& self, py::array_t<float> vectors, ThreadPool& pool) {
            py::buffer_info info = vectors.request();
            size_t dim = info.shape[info.ndim - 1]; // Assume last dim is vector dim
            size_t n = 1;
            for(int i=0; i<info.ndim-1; ++i) n *= info.shape[i];
            
            {
                py::gil_scoped_release release;
                self.train(static_cast<const float*>(info.ptr), n, pool);
            }
        }, py::arg("vectors"), py::arg("pool"))
        .def("add", [](IVFIndex& self, py::array_t<float> vectors, py::array_t<size_t> ids, ThreadPool* pool) {
            py::buffer_info vec_info = vectors.request();
            py::buffer_info id_info = ids.request();
            
            size_t n = id_info.shape[0];
            
            {
                py::gil_scoped_release release;
                self.add(static_cast<const float*>(vec_info.ptr), static_cast<const size_t*>(id_info.ptr), n, pool);
            }
        }, py::arg("vectors"), py::arg("ids"), py::arg("pool") = nullptr)
        .def("search", [](const IVFIndex& self, py::array_t<float> queries, size_t k, ThreadPool* pool) {
            py::buffer_info info = queries.request();
            size_t dim = info.shape[info.ndim - 1];
            size_t num_queries = 1;
            if (info.ndim == 2) num_queries = info.shape[0];
            
            std::pair<std::vector<float>, std::vector<size_t>> res;
            {
                py::gil_scoped_release release;
                res = self.search(static_cast<const float*>(info.ptr), num_queries, k, pool);
            }
            
            // Convert to numpy arrays to return
            py::array_t<float> dists_arr({num_queries, k}, res.first.data());
            py::array_t<size_t> ids_arr({num_queries, k}, res.second.data());
            
            return py::make_tuple(dists_arr, ids_arr);
        }, py::arg("queries"), py::arg("k"), py::arg("pool") = nullptr)
        .def("set_nprobe", &IVFIndex::set_nprobe, py::arg("nprobe"))
        .def("is_trained", &IVFIndex::is_trained)
        .def("save", &IVFIndex::save, py::arg("filename"))
        .def("load", &IVFIndex::load, py::arg("filename"))
        .def("size", &IVFIndex::size);
}
