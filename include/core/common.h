#pragma once
#include "utils/exception.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace infini {
using std::list;
using std::map;
using std::optional;
using std::pair;
using std::set;
using std::string;
using std::tie;
using std::to_string;
using std::tuple;
using std::unordered_map;
using std::vector;

// Aliases
using dtype = float;
using HashType = uint64_t; // compatible with std::hash

// Metaprogramming utilities
#define _CAT(A, B) A##B
#define _SELECT(NAME, NUM) _CAT(NAME##_, NUM)
#define _GET_COUNT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, COUNT, ...) COUNT
#define _VA_SIZE(...) _GET_COUNT(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define _VA_SELECT(NAME, ...) _SELECT(NAME, _VA_SIZE(__VA_ARGS__))(__VA_ARGS__)

// Assert: conditions should have no side effect
#define _IT_ASSERT_2(condition, info)                                          \
    static_cast<bool>(condition)                                               \
        ? void(0)                                                              \
        : throw ::infini::Exception(                                           \
              std::string("[") + __FILE__ + ":" + std::to_string(__LINE__) +   \
              "] Assertion failed (" + #condition + "): " + info)
#define _IT_ASSERT_1(condition) _IT_ASSERT_2(condition, "")
#define IT_ASSERT(...) _VA_SELECT(_IT_ASSERT, __VA_ARGS__)

#define IT_TODO_HALT() _IT_ASSERT_2(false, "Unimplemented")
#define IT_TODO_HALT_MSG(msg) _IT_ASSERT_2(false, msg)
#define IT_ASSERT_TODO(condition) _IT_ASSERT_2(condition, "Unimplemented")
#define IT_TODO_SKIP() puts("Unimplemented " __FILE__ ":" __LINE__)

// Other utilities

// std::to_underlying is avaiable since C++23
template <typename T> auto enum_to_underlying(T e) {
    return static_cast<std::underlying_type_t<T>>(e);
}

/**
 * @brief 将 vector 转换为字符串表示形式
 * @tparam T vector 元素类型
 * @param vec 要转换的 vector
 * @return 字符串表示形式的 vector
 */
template <typename T> std::string vecToString(const std::vector<T> &vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec.at(i);
        if (i < vec.size() - 1) {
            ss << ",";
        }
    }
    ss << "]";
    return ss.str();
}

/**
 * @brief 将数组转换为字符串表示形式
 * @tparam T 数组元素类型
 * @param st 指向数组的指针
 * @param length 数组长度
 * @return std::string 数组的字符串表示形式
 */
template <typename T> std::string vecToString(const T *st, size_t length) {
    std::stringstream ss;
    ss << "[";
    size_t i = 0;
    for (i = 0; i < length; i++) {
        ss << *(st + i);
        if (i < length - 1) {
            ss << ",";
        }
    }
    ss << "]";
    return ss.str();
}

/**
 * @brief 测量指定函数的平均执行时间，该函数会先进行预热，
 *        然后测量指定函数的执行时间，并返回平均执行时间（毫秒）
 * @param func 要测量执行时间的函数
 * @param sync 用于同步的函数，可确保操作完成，可为空
 * @param warmupRounds 预热轮数，在正式计时前执行指定函数的次数
 * @param timingRounds 计时轮数，在预热后执行指定函数的次数
 * @return double 平均执行时间（毫秒）
 */
double timeit(
    const std::function<void()> &func,
    // TOLEARN: 什么情况下 sync 为空
    const std::function<void(void)> &sync = []() {}, int warmupRounds = 10,
    int timingRounds = 10);

/**
 * @brief 将 vector<int> 转换为 vector<int64_t>
 * @param v32 输入的 vector<int>
 * @return vector<int64_t> 转换后的 vector<int64_t>
 */
std::vector<int64_t> castTo64(std::vector<int> const &v32);

} // namespace infini
