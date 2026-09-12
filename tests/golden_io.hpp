#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Simple text golden format for MHD regression tests.
//
// MHD_GOLDEN 1
// key value
// ...
// field <name> <n>
// <n doubles, one per line>
// ...
struct GoldenSnapshot {
    std::map<std::string, std::string> meta;
    std::map<std::string, std::vector<double>> fields;
};

inline void write_golden(const std::string& path, const GoldenSnapshot& g)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("write_golden: cannot open " + path);
    }
    out << "MHD_GOLDEN 1\n";
    out.setf(std::ios::scientific);
    out.precision(17);
    for (const auto& kv : g.meta) {
        out << kv.first << ' ' << kv.second << '\n';
    }
    for (const auto& kv : g.fields) {
        out << "field " << kv.first << ' ' << kv.second.size() << '\n';
        for (double v : kv.second) {
            out << v << '\n';
        }
    }
}

inline GoldenSnapshot read_golden(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("read_golden: cannot open " + path);
    }
    std::string tag;
    int version = 0;
    in >> tag >> version;
    if (tag != "MHD_GOLDEN" || version != 1) {
        throw std::runtime_error("read_golden: bad header in " + path);
    }

    GoldenSnapshot g;
    std::string key;
    while (in >> key) {
        if (key == "field") {
            std::string name;
            std::size_t n = 0;
            in >> name >> n;
            std::vector<double> vals(n);
            for (std::size_t i = 0; i < n; ++i) {
                if (!(in >> vals[i])) {
                    throw std::runtime_error("read_golden: truncated field " + name);
                }
            }
            g.fields.emplace(std::move(name), std::move(vals));
        } else {
            std::string value;
            if (!(in >> value)) {
                throw std::runtime_error("read_golden: truncated meta " + key);
            }
            g.meta.emplace(std::move(key), std::move(value));
        }
    }
    return g;
}

inline double rel_l2_error(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size() || a.empty()) {
        throw std::invalid_argument("rel_l2_error: size mismatch");
    }
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    if (den == 0.0) {
        return std::sqrt(num);
    }
    return std::sqrt(num / den);
}

inline double max_abs_error(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size() || a.empty()) {
        throw std::invalid_argument("max_abs_error: size mismatch");
    }
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}
