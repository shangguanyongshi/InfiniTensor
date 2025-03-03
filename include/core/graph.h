#pragma once
#include "core/lazy_allocator.h"
#include "core/operator.h"
#include "core/tensor.h"
#include <algorithm>
#include <cstdint>

namespace infini {

class GraphObj : public Object {
  protected:
    Runtime runtime;
    TensorVec tensors; // 保存计算图运行中的所有输入、输出、中间态张量
    OpVec ops; // 依次保存图中的所有算子
    LazyAllocator allocator; // 用于执行给计算图分配所需要的内存

  public:
    explicit GraphObj(Runtime runtime)
        : runtime(runtime), allocator(runtime), sorted(false){};

    /**
     * @brief 使用给定的运行时和算子列表构造一个计算图
     * @param runtime 给定的运行时
     * @param ops_in 给定的算子列表
     */
    GraphObj(Runtime runtime, OpVec ops_in);

    /**
     * @brief 将所有张量、算子的信息以字符串形式返回
     * @return string 张量、算子构成的信息字符串
     */
    string toString() const override;
    Runtime getRuntime() const { return runtime; }

    /**
     * @brief 根据给定的形状和数据类型构造张量并添加到计算图中
     * @param dim 张量的形状
     * @param dtype 张量的数据类型
     * @return Tensor 返回所加入张量的拷贝
     */
    Tensor addTensor(Shape dim, DataType dtype = DataType::Float32);

    /**
     * @brief 将相同运行时中的张量添加到计算图中
     * @param tensor 要加入的张量
     * @return Tensor 返回所加入张量的拷贝
     */
    Tensor addTensor(const Tensor &tensor);

    /**
     * @brief 对指定张量列表中的每个元素使用 addTensor 方法添加到图中
     * @param tensors 要加入计算图的张量列表
     * @return TensorVec 返回所加入张量列表的拷贝
     */
    TensorVec addTensor(const TensorVec &tensors);

    /**
     * @brief 克隆指定的张量，并将克隆后的张量加入当前计算图。Clone a tensor and add it to the graph.
     */
    Tensor cloneTensor(const Tensor &tensor) {
        return addTensor(tensor->clone(runtime));
    }

    /**
     * @brief 移除指定的算子
     * @param op 待移除的算子
     */
    void removeOperator(Operator op) {
        auto it = std::find(ops.begin(), ops.end(), op);
        if (it != ops.end())
            ops.erase(it);
    }

    /**
     * @brief 移出指定的张量
     * @param tensor 待移除的张量
     */
    void removeTensor(Tensor tensor) {
        auto it = std::find(tensors.begin(), tensors.end(), tensor);
        if (it != tensors.end())
            tensors.erase(it);
    }

    void deleteConnection(Tensor tensor, Operator op);
    void addConnection(Tensor tensor, Operator op);
    void replaceConnection(Tensor oldInput, Tensor newInput, Operator op);

    Operator cloneOperator(Operator op, TensorVec inputs, TensorVec outputs) {
        auto opClone = op->clone(inputs, outputs);
        addOperatorAndConnect(opClone);
        return opClone;
    }

    /**
     * @brief 获取当前计算图中保存的所有张量
     * @return const TensorVec& 当前计算图中保存的所有张量
     */
    const TensorVec &getTensors() const { return tensors; }

    /**
     * @brief 获取当前计算图中保存的所有算子
     * @return const OpVec& 当前计算图中保存的所有算子
     */
    const OpVec &getOperators() const { return ops; }
    OpVec getComputeOps() const;
    
    /**
     * @brief 获取张量列表中第一个与给定 Fuid 相同的张量
     * @return Tensor 与给定 Fuid 相同的第一个张量
     */
    Tensor getTensor(int) const;

    /**
     * @brief 对所有算子执行拓扑排序，以方便后续按照顺序执行计算。
     * Sort the nodes in topological order. It returns true if the sorting is successful.
     * Otherwise false is returned, means that there are rings in the graph,
     * so the topological sorting fails.
     * @return true 如果已经当前拓扑已经排过序或者排序成功，返回 true
     * @return false 如果当前 ops 中的算子存在环（ops 为空或没有前置算子为 0 的算子），
     * 返回 false
     */
    bool topo_sort();

    /**
     * @brief 对计算图进行图优化
     */
    void optimize();

    /**
     * @brief 根据每个算子推断的输出张量形状，修改当前计算图中保存对应输出的张量形状
     */
    void shape_infer();
    
    /**
     * @brief 为计算图中的每个张量指定数据应该保存的位置（在张量的 data.ptr 中保存）
     * @param useNaiveAllocator 如果此参数为 true，则使用朴素的内存分配器；
     * 否则先使用 Allocator 模拟分配内存，根据模拟的结果执行实际的内存分配
     * @param memPoolSize 可以指定是否在 Allocator 中使用内存缓冲池
     */
    void dataMalloc(bool useNaiveAllocator = false, size_t memPoolSize = 0);
    
    /**
     * @brief 从给定张量 clone 一个新张量
     * @param tensor 待 clone 的张量
     */
    Tensor cloneKV(Tensor &tensor);

    void freeHeap();

    /**
     * @brief 向计算图中添加算子（适用于需要与当前图关联起来的算子）
     * Add an operator and create its outputs. Output tensor arguments should be empty Refs (e.g., nullptr).
     */
    template <typename T, typename... Args> Ref<T> addOp(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(this, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    /**
     * @brief 向计算图中添加算子（不会将当前图的指针传递给算子的构造函数）
     * Add an operator with its outputs specified.
     */
    template <typename T, typename... Args>
    Ref<T> addOpWithOutputs(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(nullptr, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    /**
     * @brief 获取当前计算图所有输入张量构成的 vector。Gets input tensors of this graph.
     */
    inline TensorVec getInputs() const {
        TensorVec ret;
        for (const auto &t : tensors)
            if (!t->getSource())
                ret.emplace_back(t);
        return ret;
    }

    /**
     * @brief 获取当前计算图所有输出张量构成的 vector。Gets output tensors of this graph.
     */
    inline TensorVec getOutputs() const {
        TensorVec ret;
        for (const auto &t : tensors)
            if (t->getTargets().empty())
                ret.emplace_back(t);
        return ret;
    }

    bool checkValid() const;

  private:
    /**
     * @brief 向计算图中添加算子，同时执行：
     * 1. 设置将输入输出张量的 targets 和 source 分别设置为当前算子
     * 2. 同时利用输入和输出张量将算子与其前继与后继算子关联
     * （即互相加入前继和后继算子的 vector 中）
     * Add reverse connections and Op relationship in ctor.
     */
    void addOperatorAndConnect(const Operator &op);

    /**
     * @brief 记录图中的算子是否已经按照拓扑排序排好
     */
    bool sorted;

    /**
     * @brief If the weight tensors are allocated.
     */
    bool weightAllocated = false;
};

} // namespace infini
