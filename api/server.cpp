#include "crow.h"
#include <nlohmann/json.hpp>
#include "vectorforge/flat_index.h"
#include <memory>
#include <iostream>

using json = nlohmann::json;
using namespace vectorforge;

int main() {
    crow::SimpleApp app;
    
    // Create a global thread-safe index with dim 64 and L2 distance
    auto index = std::make_shared<FlatIndex>(64, MetricType::L2);

    CROW_ROUTE(app, "/stats")
    ([&index]() {
        json response;
        response["vector_count"] = index->size();
        response["dimension"] = 64;
        response["index_type"] = "FlatIndex";
        response["metric"] = "L2";
        return crow::response(200, response.dump());
    });

    CROW_ROUTE(app, "/vectors/add").methods(crow::HTTPMethod::Post)
    ([&index](const crow::request& req) {
        auto body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) {
            return crow::response(400, "Invalid JSON");
        }
        
        if (!body.contains("vectors") || !body["vectors"].is_array()) {
            return crow::response(400, "Missing 'vectors' array");
        }
        
        size_t count = 0;
        for (const auto& item : body["vectors"]) {
            if (!item.contains("id") || !item.contains("vector")) {
                continue;
            }
            
            uint64_t id = item["id"].get<uint64_t>();
            std::vector<float> vec = item["vector"].get<std::vector<float>>();
            
            if (vec.size() != 64) {
                return crow::response(400, "Vector must be 64-dimensional");
            }
            
            index->add(id, vec.data());
            count++;
        }
        
        json response;
        response["status"] = "success";
        response["added"] = count;
        return crow::response(200, response.dump());
    });

    CROW_ROUTE(app, "/search").methods(crow::HTTPMethod::Post)
    ([&index](const crow::request& req) {
        auto body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) {
            return crow::response(400, "Invalid JSON");
        }
        
        if (!body.contains("vector") || !body.contains("k")) {
            return crow::response(400, "Missing 'vector' or 'k'");
        }
        
        std::vector<float> query = body["vector"].get<std::vector<float>>();
        if (query.size() != 64) {
            return crow::response(400, "Query vector must be 64-dimensional");
        }
        
        size_t k = body["k"].get<size_t>();
        
        // Execute Search
        auto results = index->search(query.data(), k);
        
        // Format response
        json response;
        response["results"] = json::array();
        for (const auto& res : results) {
            json item;
            item["id"] = res.id;
            item["distance"] = res.distance;
            response["results"].push_back(item);
        }
        
        return crow::response(200, response.dump());
    });

    std::cout << "VectorForge REST API Server starting on port 8080" << std::endl;
    app.port(8080).multithreaded().run();
    
    return 0;
}
