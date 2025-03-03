#include "core/graph.h"
#include "operators/reshape.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini {

GraphObj::GraphObj(Runtime runtime, OpVec ops_in)
    : runtime(runtime), allocator(runtime), sorted(false) {
    map<UidBaseType, Tensor> tensorPool;
    // Clone tensors
    for (const auto &op : ops_in) {
        // 复制当前算子所有的输入和输出张量
        for (const auto &t : op->getInputs()) {
            if (t) {
                if (tensorPool.find(t->getFuid()) == tensorPool.end())
                    tensorPool[t->getFuid()] = cloneTensor(t);
            }
        }
        for (const auto &t : op->getOutputs()) {
            if (t) {
                if (tensorPool.find(t->getFuid()) == tensorPool.end())
                    tensorPool[t->getFuid()] = cloneTensor(t);
            }
        }
    }
    // Clone operators and add connections
    for (const auto &op : ops_in) {
        TensorVec inputs, outputs;
        for (const auto &t : op->getInputs()) {
            if (t) {
                inputs.emplace_back(tensorPool.at(t->getFuid()));
            }
        }

        for (const auto &t : op->getOutputs()) {
            if (t) {
                outputs.emplace_back(tensorPool.at(t->getFuid()));
            }
        }
        addOperatorAndConnect(op->clone(inputs, outputs));
    }
}

void GraphObj::addOperatorAndConnect(const Operator &op) {
    sorted = false;
    ops.push_back(op);
    for (auto &input : op->getInputs()) {
        if (input) {
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
    }
    for (auto &output : op->getOutputs()) {
        if (output) {
            output->setSource(op);
            for (auto &succ : output->getTargets()) {
                succ->addPredecessors(op);
                op->addSuccessors(succ);
            }
        }
    }
}

string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "Graph Tensors:\n";
    for (const auto &tensor : tensors)
        oss << tensor << "\n";

    oss << "Graph operators:\n";
    for (const auto &op : ops) {
        vector<UidBaseType> preds, succs;
        for (auto &o : op->getPredecessors())
            preds.emplace_back(o->getGuid());
        for (auto &o : op->getSuccessors())
            succs.emplace_back(o->getGuid());
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds);
        oss << ", succ " << vecToString(succs);
        oss << ", " << op << "\n";
    }
    return oss.str();
}

bool GraphObj::topo_sort() {
    if (this->sorted) {
        return true;
    }
    std::vector<Operator> sorted;
    std::unordered_set<OperatorObj *> flags;
    sorted.reserve(ops.size());
    flags.reserve(ops.size());
    while (sorted.size() < ops.size()) {
        // Any node is move to sorted in this loop.
        auto modified = false;
        for (auto const &op : ops) {
            if (auto const &inputs = op->getInputs();
                flags.find(op.get()) == flags.end() &&
                std::all_of(inputs.begin(), inputs.end(),
                            [&flags](auto const &input) {
                                auto ptr = input->getSource().get();
                                return !ptr || flags.find(ptr) != flags.end();
                            })) {
                modified = true;
                sorted.emplace_back(op);
                flags.insert(op.get());
            }
        }
        if (!modified) {
            return false;
        }
    }
    this->ops = std::move(sorted);
    return this->sorted = true;
}

void GraphObj::optimize() {
    for (auto &op : ops) {
        switch (op->getOpType().underlying()) {
        default:
            break;
        }
    }
}

Tensor GraphObj::getTensor(int fuid) const {
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

void GraphObj::shape_infer() {
    for (auto &op : ops) {
        auto ans = op->inferShape();
        IT_ASSERT(ans.has_value());
        auto oldOutputs = op->getOutputs();
        IT_ASSERT(ans.value().size() == oldOutputs.size());
        // replace the old outputshape and size with new one
        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];
            auto oldShape = oldOutputs[i]->getDims();
            auto fuid = oldOutputs[i]->getFuid();
            if (newShape != oldShape) {
                auto tensor = this->getTensor(fuid);
                tensor->setShape(newShape);
            }
        }
    }
}

void GraphObj::dataMalloc(bool useNaiveAllocator, size_t memPoolSize) {
    // topological sorting first
    IT_ASSERT(topo_sort() == true);

    // 如果设置了 useNaiveAllocator 为 true，则直接为每个张量分别分配内存
    if (useNaiveAllocator) {
        // can not set memory pool when use naive allocator
        IT_ASSERT(memPoolSize == 0);
        // used for debugging memory out-of-bounds access, tensors will not be
        // released correctly
        // note: behavior may not match running in non-naive mode, and it may
        // not reproduce the bug
        for (auto &tensor : tensors) {
            if (!tensor->isWeight() ||
                (tensor->isWeight() && !weightAllocated)) {
                tensor->dataMalloc();
            }
        }
        return;
    }
    if (memPoolSize > 0) {
        allocator.setMemPool(memPoolSize);
    }
    
    // count the number of times all tensors are used
    // 记录张量被多少个算子使用
    std::unordered_map<TensorObj *, size_t> tensorToRefCount;

    // record the memory address offsets of all tensors to be allocated
    // 记录所有需要分配内存的张量（权重、输入、输出、其他）的内存地址偏移量
    std::unordered_map<TensorObj *, size_t> tensorToOffset;

    // reinit allocator
    allocator.init();

    // record all weight tensors, including weight tensors and kvcache tensors
    std::unordered_set<TensorObj *> weightTensors;

    // 此循环中：
    // 1. 对权重张量直接在 memPool 中获取偏移地址，并插入 weightTensors 集合
    // 2. 输入、输出张量在此循环中只使用 alloc 模拟分配
    // 3. 对于其他张量，获取其目标算子的个数，保存到 tensorToRefCount 集合
    // 4. 对于没有源算子的张量（是用户创建的张量），使用 alloc 模拟分配内存
    for (auto &tensor : tensors) {
        if (tensor->isWeight()) {
            // allocate memory for all weight tensors first, and this memory
            // will not be freed until the graph is destroyed
            // 模拟并记录权重张量内存分配时的偏移量
            weightTensors.insert(tensor.get());
            if (!this->weightAllocated) {
                tensorToOffset[tensor.get()] = allocator.allocWeight(tensor->getBytes());
            }
        } else if (tensor->isInput() || tensor->isOutput()) {
            // allocate memory for all input and output tensors, and this memory
            // will not be reused later
            // 模拟并记录输入和输出张量内存分配时的偏移量
            tensorToOffset[tensor.get()] = allocator.alloc(tensor->getBytes());
        } else {
            // 对于其他张量，先记录其目标算子的个数
            tensorToRefCount[tensor.get()] = tensor->getTargets().size();
            // allocate memory for all user-created tensors
            // 如果张量没有来源算子（是用户创建的张量），则进行模拟内存分配
            if (tensor.get()->getSource() == nullptr) {
                tensorToOffset[tensor.get()] = allocator.alloc(tensor->getBytes());
            }
        }
    }
    
    // if memory has not yet been allocated for weight tensors,
    // allocate memory now and do not allocate again in the future.
    // 权重张量的内存在 memPool 中保存，前面已经给 memPool 分配过内存了，
    // 这里可以直接记录并设置权重张量的实际运行时和对应的内存地址
    if (!this->weightAllocated) {
        this->weightAllocated = true;
        // only allocate once for weight tensors
        for (auto &tensor : weightTensors) {
            IT_ASSERT(tensorToOffset.find(tensor) != tensorToOffset.end());
            tensor->setDataBlob(make_ref<BlobObj>(
                tensor->runtime,
                static_cast<uint8_t *>(allocator.getWeightPtr()) + tensorToOffset[tensor]
            ));
        }
    }
    
    // traverse in topological order and simulate memory allocation
    // 按照拓扑序遍历每个算子，模拟其他类型张量内存的分配和释放，
    // 遍历到当前算子表示正在对该算子的输入应用该算子的计算逻辑，
    // 计算前应该为输出张量分配内存，计算后应该释放不再使用的输入张量的内存
    for (auto &op : ops) {
        // memory should be allocated for the op's output first
        // 1. 先获取算子的输出张量，并模拟分配内存
        auto outputs = op->getOutputs();
        for (auto &tensor : outputs) {
            if (tensor) {
                if (tensor->isOthers()) {
                    tensorToOffset[tensor.get()] = allocator.alloc(tensor->getBytes());
                }
            }
        }
        // 2. 再获取算子的输入张量，并模拟释放内存
        auto inputs = op->getInputs();
        for (auto &tensor : inputs) {
            if (tensor) {
                if (tensor->isOthers()) {
                    // 检查张量是否被其他算子使用
                    auto tensorIter = tensorToRefCount.find(tensor.get());
                    IT_ASSERT(tensorIter != tensorToRefCount.end());
                    IT_ASSERT(tensorToRefCount[tensor.get()] > 0);
                    // 如果张量被其他算子使用，则减少引用计数（表示当前算子计算完成了）
                    tensorToRefCount[tensor.get()] -= 1;
                    if (tensorToRefCount[tensor.get()] == 0) {
                        // 如果当前张量的引用计数减少到了 0，表示该张量后续不再使用了，应该释放内存
                        // indicate that this tensor will no longer be used and
                        // perform memory free
                        tensorToRefCount.erase(tensor.get());
                        allocator.free(tensorToOffset[tensor.get()],
                                       tensor->getBytes());
                    }
                }
            }
        }
    }

    // perform actual memory allocation for non-weight tensors
    // 调用 getPtr 会根据模拟结果，分配实际的内存空间，此时需要将实际内存空间 + 对应张量的偏移量
    // 保存到张量的 data.ptr 中，以保证后续计算时可以直接获取数据的保存位置
    for (auto &tensor : tensors) {
        if (!tensor->isWeight()) {
            IT_ASSERT(tensorToOffset.find(tensor.get()) != tensorToOffset.end());
            tensor->setDataBlob(make_ref<BlobObj>(
                tensor->runtime, 
                static_cast<uint8_t *>(allocator.getPtr()) + tensorToOffset[tensor.get()]
            ));
        }
    }
}

Tensor GraphObj::cloneKV(Tensor &tensor) {
    auto obj = tensor->clone();
    if (allocator.getMemPoolStatus()) {
        if (tensor->hasData()) {
            // 按照 tensor 的
            obj->setDataBlob(make_ref<BlobObj>(
                tensor->runtime,
                static_cast<uint8_t *>(allocator.getHeapPtr()) + allocator.heapAlloc(tensor->getBytes())
            ));
            obj->copyData(tensor);
        }
    } else {
        if (tensor->hasData()) {
            obj->dataMalloc();
            obj->copyData(tensor);
        }
    }
    return obj;
}

void GraphObj::freeHeap() { this->allocator.freeHeap(); }

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
    IT_ASSERT(tensor->getRuntime() == runtime,
              std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                  tensor->getRuntime()->toString() + " to " +
                  runtime->toString());
    tensors.emplace_back(tensor);
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t);
    return tensors;
}

OpVec GraphObj::getComputeOps() const {
    OpVec opList;
    for (auto op : ops)
        if (op->getOpType().isMatMulOrConv())
            opList.emplace_back(op);
    return opList;
}

void GraphObj::deleteConnection(Tensor tensor, Operator op) {
    // if op is target
    IT_ASSERT(std::find(tensor->getTargets().begin(),
                        tensor->getTargets().end(),
                        op) != tensor->getTargets().end());
    tensor->removeTarget(op);
    if (tensor->getSource()) {
        tensor->getSource()->removeSuccessors(op);
        op->removePredecessors(tensor->getSource());
    }
}

// add op as a target
void GraphObj::addConnection(Tensor tensor, Operator op) {
    tensor->addTarget(op);
    if (tensor->getSource()) {
        tensor->getSource()->addSuccessors(op);
        op->addPredecessors(tensor->getSource());
    }
}

void GraphObj::replaceConnection(Tensor oldTensor, Tensor newTensor,
                                 Operator op) {
    // op is a target of old tensor
    IT_ASSERT(std::find(oldTensor->getTargets().begin(),
                        oldTensor->getTargets().end(),
                        op) != oldTensor->getTargets().end());
    addConnection(newTensor, op);
    deleteConnection(oldTensor, op);
    op->replaceInput(oldTensor, newTensor);
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
    for (auto tensor : tensors) {
        IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                    nullptr == tensor->getSource()));
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
        }
        auto op = tensor->getSource();
        IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
    }
    for (auto op : ops) {
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
        }
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
        }
    }
    std::set<UidBaseType> s;
    // check whether two tensors with the same FUID exist
    for (auto tensor : tensors) {
        int cnt = s.count(tensor->getFuid());
        IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
        s.insert(tensor->getFuid());
    }
    return true;
}

} // namespace infini
