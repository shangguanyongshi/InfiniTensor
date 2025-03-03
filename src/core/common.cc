#include "core/common.h"
#include <chrono>
#include <functional>

namespace infini {

double timeit(const std::function<void()> &func,
              const std::function<void(void)> &sync, int warmupRounds,
              int timingRounds) {
    // 预热阶段：执行 func 函数 warmupRounds 次，以确保函数达到稳定的执行状态
    for (int i = 0; i < warmupRounds; ++i)
        func();
    // 调用 sync 执行同步
    if (sync)
        sync();
    // 计时阶段：执行 func 函数 timingRounds 次，并计算平均执行时间
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timingRounds; ++i)
        func();
    if (sync)
        sync();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           timingRounds;
}

// transform vector<int> to vector<int64_t>
std::vector<int64_t> castTo64(std::vector<int> const &v32) {
    if (v32.size() == 0) {
        std::vector<int64_t> v64(1, 1);
        return v64;
    }
    std::vector<int64_t> v64(v32.size(), 1);
    for (size_t i = 0; i < v32.size(); ++i) {
        v64[i] = int64_t(v32[i]);
    }
    return v64;
}

} // namespace infini
