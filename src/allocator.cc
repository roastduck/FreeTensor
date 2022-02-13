#include <malloc.h> // memalign

#include <allocator.h>

namespace ir {

bool SmallItemBlock::full() const { return items_[0].next_ == nullptr; }

SmallItem *SmallItemBlock::allocate() {
    SmallItem *item = items_[0].next_;
    items_[0].next_ = item->next_;
    return item;
}

void SmallItemBlock::deallocate(SmallItem *item) {
    item->next_ = items_[0].next_;
    items_[0].next_ = item;
}

SmallItemBlock *SmallItemBlock::newBlk() {
    SmallItemBlock *blk = (SmallItemBlock *)memalign(sizeof(SmallItemBlock),
                                                     sizeof(SmallItemBlock));
    blk->items_[SMALL_ITEM_PER_BLOCK - 1].next_ = nullptr;
    for (int i = SMALL_ITEM_PER_BLOCK - 2; i >= 0; i--) {
        blk->items_[i].next_ = &blk->items_[i + 1];
    }
    return blk;
}

void SmallItemBlock::delBlk(SmallItemBlock *blk) { free(blk); }

thread_local SmallItemAllocator SmallItemAllocator::instance_;

SmallItemAllocator::SmallItemAllocator()
    : curBlk(0), blocks_(1, SmallItemBlock::newBlk()) {}

SmallItemAllocator::~SmallItemAllocator() {
    for (auto *blk : blocks_) {
        SmallItemBlock::delBlk(blk);
    }
}

void SmallItemAllocator::lock() {
    while (spinLock_.test_and_set(std::memory_order_acquire)) {
        ; // spin
    }
}

void SmallItemAllocator::unlock() {
    spinLock_.clear(std::memory_order_release);
}

void *SmallItemAllocator::allocate() {
    lock();
    size_t nBlk = blocks_.size();
    for (size_t i = 0; i < nBlk; i++) {
        if (!blocks_[curBlk]->full()) {
            void *ret = blocks_[curBlk]->allocate();
            unlock();
            return ret;
        }
        curBlk = (curBlk + 1) % nBlk;
    }
    curBlk = nBlk;
    blocks_.emplace_back(SmallItemBlock::newBlk());
    void *ret = blocks_[curBlk]->allocate();
    unlock();
    return ret;
}

void SmallItemAllocator::deallocate(void *p) {
    lock();
    auto blk = (SmallItemBlock *)((size_t)p & ~(sizeof(SmallItemBlock) - 1));
    blk->deallocate((SmallItem *)p);
    unlock();
}

} // namespace ir
