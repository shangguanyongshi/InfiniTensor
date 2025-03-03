#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {

/**
 * @brief 用于模拟与实际使用运行时执行张量内存、权重内存和堆内存的分配和释放，
 * 主要提供了 alloc() 和 free() 分别模拟分配和释放。模拟完成后可以使用 getPtr() 执行
 * 实际的内存分配，并返回分配的内存的起始地址
 */
class LazyAllocator {
  private:
#ifdef BUILD_TEST
    FRIEND_TEST(LazyAllocator, testMergeFreeBlocks);

    FRIEND_TEST(LazyAllocator, testAllocWithEndFreeBlock);
#endif

    Runtime runtime;

    size_t used = 0;

    size_t peak = 0; // 张量内存分配的峰值

    size_t weightPeak = 0; // 权重内存分配的峰值

    size_t heapPeak = 0; // 堆内存分配的峰值

    size_t alignment;

    bool hasMemPool = false;

    size_t memPoolSize = 0; // 预分配的内存池，在实际执行内存分配时可以直接使用

    /**
     * @brief 指向实际分配的张量内存的指针。pointer to the memory actually allocated
     */
    void *ptr = nullptr;

    /**
     * @brief 指向实际分配的权重内存的指针。pointer to the weight memory space
     */
    void *weightPtr = nullptr;

    /**
     * @brief 指向实际分配的内存池的指针。memory pool ptr
     */
    void *memPoolPtr = nullptr;

    // // a cache designed for a batch size that has already occurred
    // std::unordered_map<size_t, std::unordered_map<TensorObj *, size_t>>
    // batchsizeToTensorOffset;

    struct freeBlockInfo {
        size_t addr;
        size_t blockSize;
    };

    struct cmpFreeBlockInfo {
        bool operator()(const freeBlockInfo &a, const freeBlockInfo &b) const {
            return (a.blockSize != b.blockSize) ? (a.blockSize < b.blockSize)
                                                : (a.addr < b.addr);
        }
    };

    // free balanced tree, maintains all free memory blocks
    // 保存所有空闲的内存块（每个元素是一个 freeBlockInfo，其中保存了内存块的起始地址和大小）
    std::set<freeBlockInfo, cmpFreeBlockInfo> freeBlocks;

    // key: head address offset of the free memory block
    // value: blockSize of the block
    /**
     * @brief 保留所有空闲内存块的头地址偏移量及空闲块的大小
     */
    std::unordered_map<size_t, size_t> headAddrToBlockSize;

    // key: tail address offset of the free memory block
    // value: blockSize of the block
    /**
     * @brief 保存所有空闲内存块的尾地址偏移量及空闲块的大小
     */
    std::unordered_map<size_t, size_t> tailAddrToBlockSize;

  public:
    LazyAllocator(Runtime runtime);

    virtual ~LazyAllocator();

    /**
     * @brief 重置所有已使用、已分配、模拟峰值等内存信息为空
     */
    void init();

    /**
     * @brief 设置内存池的大小
     * @param memPoolSize 内存池的大小
     */
    void setMemPool(size_t memPoolSize);

    /**
     * @brief 获取内存池的状态
     * @return true 表示有内存池
     * @return false 表示没有内存池
     */
    bool getMemPoolStatus();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    /**
     * @brief 模拟内存分配
     * @param size 此次要分配的内存大小
     * @return size_t 返回此次分配的内存的起始地址
     */
    size_t alloc(size_t size);

    /**
     * @brief 模拟权重内存分配
     * @param size 此次要分配的内存权重的大小
     * @return size_t 返回此次分配的内存权重的起始地址
     */
    size_t allocWeight(size_t size);
    
    /**
     * @brief 模拟堆内存分配
     * @param size 此次要分配的内存堆的大小
     * @return size_t 返回此次分配的内存堆的起始地址
     */
    size_t heapAlloc(size_t size);

    void freeHeap();

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    /**
     * @brief 模拟内存释放
     * @param addr 要释放的内存块的起始地址
     * @param size 要释放的内存块的大小
     */
    void free(size_t addr, size_t size);

    /**
     * @brief 执行实际的内存分配；
     * 如果没有内存池，分配 peak 大小的内存并返回地址；
     * 如果有内存池，使用内存池中权重内存后的地址作为张量内存的保存位置；
     * perform actual memory allocation
     * @return void* 所分配张量内存的起始地址
     * pointer to the head address of the allocated memory
     */
    void *getPtr();

    // void addCache(size_t batchsize, std::unordered_map<TensorObj *, size_t>);

    // std::unordered_map<TensorObj *, size_t> getCache(size_t batchsize);

    /**
     * @brief 获取权重内存的起始地址
     * 如果没有内存池，分配 weightPeak 大小的内存并返回地址
     * 如果有内存池，使用内存池的起始地址作为权重内存的保存位置
     * @return void* 权重内存的起始地址
     */
    void *getWeightPtr();

    /**
     * @brief 获取堆内存的起始地址（直接返回内存池的起始地址作为堆内存的起始地址）
     * @return void* 堆内存的起始地址
     */
    void *getHeapPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
};

} // namespace infini
