#include "core/lazy_allocator.h"
#include <utility>

namespace infini {

// In
// cuda-c-programming-guide(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses):
// Any address of a variable residing in global memory or returned by one of the
// memory allocation routines from the driver or runtime API is always aligned
// to at least 256 bytes.
constexpr size_t alignmentInBytesForCUDA = 256;

LazyAllocator::LazyAllocator(Runtime runtime) : runtime(runtime) {
    if (runtime->isCuda()) {
        // TODO: the alignment on cuda might need further discussion
        alignment = alignmentInBytesForCUDA;
    } else {
        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        // TODO: the alignment on bang might need further discussion
        alignment = sizeof(uint64_t);
    }
}

LazyAllocator::~LazyAllocator() {
    if (this->ptr != nullptr) {
        runtime->dealloc(this->ptr);
    }
    if (this->weightPtr != nullptr) {
        runtime->dealloc(this->weightPtr);
    }
    if (this->memPoolPtr != nullptr) {
        runtime->dealloc(this->memPoolPtr);
    }
}

void LazyAllocator::init() {
    used = 0;
    peak = 0;
    freeBlocks.clear();
    headAddrToBlockSize.clear();
    tailAddrToBlockSize.clear();
    if (this->ptr != nullptr) {
        runtime->dealloc(this->ptr);
    }
    this->ptr = nullptr;
}

void LazyAllocator::setMemPool(size_t memPoolSize) {
    IT_ASSERT(memPoolSize > 0);
    if (!this->hasMemPool) {
        this->hasMemPool = true;
        this->memPoolSize = memPoolSize;
        this->memPoolPtr = runtime->alloc(memPoolSize);
    }
}

bool LazyAllocator::getMemPoolStatus() { return this->hasMemPool; }

size_t LazyAllocator::alloc(size_t size) {
    // pad the size to the multiple of alignment
    size = this->getAlignedSize(size);
    // 查找第一个大于等于 size 的空闲块
    auto it = this->freeBlocks.lower_bound(freeBlockInfo{(size_t)0, size});

    // 默认返回 peak，即当前已分配的内存的末尾地址
    size_t retAddr = this->peak;
    if (it != this->freeBlocks.end()) {
        // found an alvailable free memory block for allocation
        // 如果有大小大于等于 size 的空闲块，则直接使用该空闲块
        size_t blockSize = it->blockSize;
        // 结果的地址为该空闲块的起始地址
        retAddr = it->addr;
        // 该空闲块的尾地址
        size_t tailAddr = retAddr + size;
        // update the map of head and tail address offset of memory blocks
        // 更新保存空闲块头地址和尾地址的 map
        this->headAddrToBlockSize.erase(retAddr);
        this->tailAddrToBlockSize.erase(tailAddr);
        // memory block splitting
        // 如果空闲块的大小大于 size，则将多余的部分内存块重新插入到 freeBlocks 中，
        // 需要同时将该空闲块的起始地址和结尾地址信息保存到对应的 map 中
        if (blockSize > tailAddr - retAddr) {
            freeBlockInfo newBlock = {tailAddr,
                                      blockSize - (tailAddr - retAddr)};
            this->headAddrToBlockSize[tailAddr] = newBlock.blockSize;
            this->tailAddrToBlockSize[retAddr + blockSize] = newBlock.blockSize;
            this->freeBlocks.insert(newBlock);
        }
        // update the free balanced tree
        // 从 freeBlocks 中删除该空闲块
        this->freeBlocks.erase(it);
        // 将已分配的内存大小加到 used 中
        this->used += tailAddr - retAddr;
    } else {
        // the allocated memory space is not sufficient for reallocation, it
        // needs to be extended
        // 如果没有大于等于 size 的空闲块，则直接在 peak 处分配 size 大小的内存
        auto blockTailWithPeak = this->tailAddrToBlockSize.find(this->peak);
        if (blockTailWithPeak != this->tailAddrToBlockSize.end()) {
            // there is a free block located at the end of the currently
            // allocated memory, where this free block has its tail address as
            // 'peak'
            // 如果存在某个空闲内存块的尾地址为 peak，则直接在该空闲块的基础上扩展以获得 size 大小的内存
            // 获取该块的起始地址
            retAddr = this->peak - blockTailWithPeak->second;
            IT_ASSERT(blockTailWithPeak->second < size);
            // 在 peak 后分配当前空闲块相比于 size 的差值大小的内存
            this->peak += (size - blockTailWithPeak->second);
            // updata freeBlocks, headAddrToBlockSize and tailAddrToBlockSize
            freeBlockInfo endBlock = {retAddr, blockTailWithPeak->second};
            // 将该空闲块从 freeBlocks 中删除（已被扩展并分配给某次程序运行时）
            this->freeBlocks.erase(endBlock);
            // 从 headAddrToBlockSize 中删除该空闲块的信息
            this->headAddrToBlockSize.erase(endBlock.addr);
            // 从 tailAddrToBlockSize 中删除该空闲块的信息
            this->tailAddrToBlockSize.erase(endBlock.addr + endBlock.blockSize);
        } else {
            // 如果没有尾地址为 peak 的空闲块，则直接在 peak 处分配 size 大小的内存
            // 还需要更新 peak 的值为原有 peak 加上 size（刷新新了所使用内存的峰值）
            this->peak = this->peak + size;
        }
        // 将已分配的内存大小加到 used 中
        this->used += size;
    }

    return retAddr;
}

size_t LazyAllocator::allocWeight(size_t size) {
    IT_ASSERT(this->weightPtr == nullptr);
    size = this->getAlignedSize(size);
    size_t retAddr = this->weightPeak;
    this->weightPeak += size;
    return retAddr;
}

size_t LazyAllocator::heapAlloc(size_t size) {
    size = this->getAlignedSize(size);
    this->heapPeak += size;
    IT_ASSERT(this->memPoolSize >=
              this->weightPeak + this->peak + this->heapPeak);
    size_t retAddr = this->memPoolSize - this->heapPeak;
    return retAddr;
}

void LazyAllocator::freeHeap() { this->heapPeak = 0; }

void LazyAllocator::free(size_t addr, size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);
    // 记录该内存块的尾地址
    auto tailAddr = addr + size;
    freeBlockInfo block = {addr, tailAddr - addr};
    // 在空闲块的头和尾地址 map 中保存该空闲块的信息
    this->headAddrToBlockSize[addr] = block.blockSize;
    this->tailAddrToBlockSize[tailAddr] = block.blockSize;
    // 查找该空闲块的前一个和后一个空闲块
    auto preFreeBlockIter = this->tailAddrToBlockSize.find(addr);
    auto subFreeBlockIter = this->headAddrToBlockSize.find(tailAddr);
    if (preFreeBlockIter != this->tailAddrToBlockSize.end()) {
        // the head address of the memory block to be freed matches the end of a
        // free block, merge them together
        // 如果当前空闲块前面有空闲块，将两个空闲块合并
        size_t preBlockSize = preFreeBlockIter->second;
        this->headAddrToBlockSize.erase(block.addr);
        this->headAddrToBlockSize[block.addr - preBlockSize] += block.blockSize;
        this->tailAddrToBlockSize.erase(block.addr);
        this->tailAddrToBlockSize[tailAddr] += preBlockSize;
        block.addr -= preBlockSize;
        block.blockSize += preBlockSize;
        // delete the preceding adjacent free block
        this->freeBlocks.erase(freeBlockInfo{block.addr, preBlockSize});
    }
    if (subFreeBlockIter != this->headAddrToBlockSize.end()) {
        // the tail address of the memory block to be freed matches the start of
        // a free block, merge them together
        // 如果当前空闲块后面有空闲块，将两个空闲块合并
        auto subBlockSize = subFreeBlockIter->second;
        this->headAddrToBlockSize.erase(tailAddr);
        this->headAddrToBlockSize[block.addr] += subBlockSize;
        this->tailAddrToBlockSize.erase(tailAddr);
        this->tailAddrToBlockSize[tailAddr + subBlockSize] += block.blockSize;
        tailAddr += subBlockSize;
        block.blockSize += subBlockSize;
        // delete the succeeding adjacent memory block
        this->freeBlocks.erase(
            freeBlockInfo{tailAddr - subBlockSize, subBlockSize});
    }
    this->freeBlocks.insert(block);
    this->used -= size;
}

void *LazyAllocator::getPtr() {
    if (!hasMemPool) {
        if (this->ptr == nullptr) {
            this->ptr = runtime->alloc(this->peak);
            // #ifdef DEBUG_MODE
            //         printf("LazyAllocator really alloc non-weight: %p %lu
            //         bytes\n", this->ptr, peak);
            // #endif
        }
        return this->ptr;
    } else {
        IT_ASSERT(this->memPoolSize >= this->weightPeak + this->peak);
        return static_cast<uint8_t *>(this->memPoolPtr) + weightPeak;
    }
}

void *LazyAllocator::getWeightPtr() {
    if (!hasMemPool) {
        if (this->weightPtr == nullptr) {
            this->weightPtr = runtime->alloc(this->weightPeak);
            // #ifdef DEBUG_MODE
            //         printf("LazyAllocator really alloc weight: %p %lu
            //         bytes\n",
            //                this->weightPtr, weightPeak);
            // #endif
        }
        return this->weightPtr;
    } else {
        return this->memPoolPtr;
    }
}

void *LazyAllocator::getHeapPtr() {
    IT_ASSERT(hasMemPool);
    return this->memPoolPtr;
}

size_t LazyAllocator::getAlignedSize(size_t size) {
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void LazyAllocator::info() {
    std::cout << "Used memory: " << this->used + this->weightPeak
              << ", peak memory: " << this->peak + this->weightPeak
              << std::endl;
}

} // namespace infini
