#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

namespace vectorforge {

// The types of distance metrics
enum class MetricType {
    L2,      // Euclidean distance (Straight line distance)
    IP,      // Inner Product / Dot Product (Magnitude * Angle)
    Cosine   // Cosine Similarity (Angle)
};

// Represents a single vector
struct Vector {
    std::vector<float> data;
    size_t dim;

    // Constructors
    Vector() : dim(0) {}
    Vector(const std::vector<float>& d) : data(d), dim(d.size()) {}
    Vector(size_t d) : data(d, 0.0f), dim(d) {}
};

// Represents a result
struct SearchResult {
    uint64_t id;      // Unique identifier for the vector
    float distance;   // Distance to the query vector

    // Used for sorting results
    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }
};

} // namespace vectorforge
