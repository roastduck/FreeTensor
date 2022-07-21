#ifndef FREE_TENSOR_ALLOCATOR_H
#define FREE_TENSOR_ALLOCATOR_H

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace freetensor {

constexpr int SMALL_ITEM_SIZE = 64;
constexpr int SMALL_ITEM_PER_BLOCK = 16384 / SMALL_ITEM_SIZE;

union SmallItem {
    SmallItem *next_;
    uint8_t data_[SMALL_ITEM_SIZE];
};

class SmallItemBlock {
    SmallItem items_[SMALL_ITEM_PER_BLOCK];

  private:
    SmallItemBlock() = default;
    ~SmallItemBlock() = default;

  public:
    bool full() const;
    [[nodiscard]] SmallItem *allocate();
    void deallocate(SmallItem *item);

    static SmallItemBlock *newBlk();
    static void delBlk(SmallItemBlock *blk);
};
static_assert(sizeof(SmallItemBlock) == SMALL_ITEM_SIZE * SMALL_ITEM_PER_BLOCK);

class SmallItemAllocator {
    size_t curBlk;
    std::vector<SmallItemBlock *> blocks_;
    std::atomic_flag spinLock_ = ATOMIC_FLAG_INIT;

    // We must define instance_ as a static pointer of an dynamic object,
    // instead of a static object, and the dynamic object shall never be free'd.
    // Otherwise, some static variables that use the allocator may be free'd
    // after the allocator
    static thread_local SmallItemAllocator *instance_;

  private:
    void lock();
    void unlock();

  public:
    SmallItemAllocator();
    ~SmallItemAllocator();

    [[nodiscard]] void *allocate();
    void deallocate(void *p);

    static SmallItemAllocator *instance() {
        if (instance_ == nullptr) {
            instance_ = new SmallItemAllocator();
        }
        return instance_;
    }
};

template <class T> class Allocator {
    SmallItemAllocator *smallItemAllocator_;

  public:
    typedef T value_type;
    typedef std::true_type is_always_equal;

    Allocator() : smallItemAllocator_(SmallItemAllocator::instance()) {}

    template <class U> Allocator(const Allocator<U> &other) : Allocator() {}
    template <class U> Allocator(Allocator<U> &&other) : Allocator() {}

    [[nodiscard]] T *allocate(size_t n) {
        if (n * sizeof(T) > SMALL_ITEM_SIZE) {
            return (T *)malloc(n * sizeof(T));
        } else {
            return (T *)smallItemAllocator_->allocate();
        }
    }

    void deallocate(T *p, size_t n) {
        if (n * sizeof(T) > SMALL_ITEM_SIZE) {
            free(p);
        } else {
            smallItemAllocator_->deallocate(p);
        }
    }

    template <class... Args> void construct(T *p, Args &&...args) {
        ::new ((void *)p) T(std::forward<Args>(args)...);
    }
};

template <class T>
bool operator==(const Allocator<T> &lhs, const Allocator<T> &rhs) {
    return true;
}
template <class T>
bool operator!=(const Allocator<T> &lhs, const Allocator<T> &rhs) {
    return false;
}

} // namespace freetensor

#endif // FREE_TENSOR_ALLOCATOR_H
