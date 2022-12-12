#ifndef FREE_TENSOR_INTEGER_RANGE_H
#define FREE_TENSOR_INTEGER_RANGE_H

#include <iterator>
#include <type_traits>

#ifdef __CUDA_ARCH__
#define FUNC_ATTR __attribute__((always_inline)) __host__ __device__
#else
#define FUNC_ATTR __attribute__((always_inline))
#endif

template <typename IndexType, typename CounterType> class IntegerRangeIterator {
    IndexType index_, step_;
    CounterType counter_;

  public:
    typedef std::make_signed_t<CounterType> DiffType;

    FUNC_ATTR constexpr IntegerRangeIterator() {}
    FUNC_ATTR constexpr IntegerRangeIterator(auto &&index, auto &&step,
                                             auto &&counter)
        : index_(index), step_(step), counter_(counter) {}

    FUNC_ATTR constexpr auto operator*() const { return index_; }

    FUNC_ATTR friend constexpr auto
    operator==(const IntegerRangeIterator &lhs,
               const IntegerRangeIterator &rhs) {
        return lhs.counter_ == rhs.counter_;
    }
    FUNC_ATTR friend constexpr auto
    operator<=>(const IntegerRangeIterator &lhs,
                const IntegerRangeIterator &rhs) {
        return lhs.counter_ <=> rhs.counter_;
    }

    // Prefix ++/--
    FUNC_ATTR constexpr auto &operator++() {
        index_ += step_;
        counter_++;
        return *this;
    }
    FUNC_ATTR constexpr auto &operator--() {
        index_ -= step_;
        counter_--;
        return *this;
    }

    // Postfix ++/--
    FUNC_ATTR constexpr auto operator++(int) {
        auto old = *this;
        index_ += step_;
        counter_++;
        return old;
    }
    FUNC_ATTR constexpr auto operator--(int) {
        auto old = *this;
        index_ += step_;
        counter_++;
        return old;
    }

    FUNC_ATTR constexpr auto &operator+=(DiffType diff) {
        index_ += diff * step_;
        counter_ += diff;
        return *this;
    }
    FUNC_ATTR constexpr auto &operator-=(DiffType diff) {
        index_ -= diff * step_;
        counter_ -= diff;
        return *this;
    }

    FUNC_ATTR friend constexpr DiffType
    operator-(const IntegerRangeIterator &lhs,
              const IntegerRangeIterator &rhs) {
        return lhs.counter_ - rhs.counter_;
    }
    FUNC_ATTR friend constexpr auto operator+(const IntegerRangeIterator &lhs,
                                              DiffType rhs) {
        return IntegerRangeIterator(lhs.index_ + rhs * lhs.step_, lhs.step_,
                                    lhs.counter_ + rhs);
    }
    FUNC_ATTR friend constexpr auto operator-(const IntegerRangeIterator &lhs,
                                              DiffType rhs) {
        return IntegerRangeIterator(lhs.index_ - rhs * lhs.step_, lhs.step_,
                                    lhs.counter_ - rhs);
    }
    FUNC_ATTR friend constexpr auto operator+(DiffType lhs,
                                              const IntegerRangeIterator &rhs) {
        return IntegerRangeIterator(rhs.index_ + lhs * rhs.step_, rhs.step_,
                                    rhs.counter_ + lhs);
    }

    FUNC_ATTR constexpr auto operator[](DiffType offset) const {
        return index_ + offset * step_;
    }
};

namespace std {

template <typename IndexType, typename CounterType>
struct iterator_traits<IntegerRangeIterator<IndexType, CounterType>> {
    typedef typename IntegerRangeIterator<IndexType, CounterType>::DiffType
        difference_type;
    typedef IndexType value_type;
    typedef const IndexType *pointer;
    typedef const IndexType *reference;
    typedef std::random_access_iterator_tag iterator_category;
};

} // namespace std

template <typename IndexType, typename CounterType> class IntegerRange {
    // Required by OpenMP
    static_assert(std::random_access_iterator<
                  IntegerRangeIterator<IndexType, CounterType>>);

    IntegerRangeIterator<IndexType, CounterType> begin_, end_;

  public:
    FUNC_ATTR constexpr IntegerRange(IndexType begin, IndexType end,
                                     IndexType step, CounterType len)
        : begin_(begin, step, 0), end_(end, step, len) {}

    FUNC_ATTR constexpr const auto &begin() const { return begin_; }
    FUNC_ATTR constexpr const auto &end() const { return end_; }
};

/**
 * Range of (begin, end, step, len) that can be used in a ranged-based loop
 *
 * We use range-based loops instead of canonical loops in runtime, for the
 * following reasons:
 *
 * - It is clearly loop-invariant. On the contrary, canonical loops like `for
 * (int i = 0; i < foo; i++)` can get insufficient optimization because the
 * compiler can hardly determine whether `foo` is loop-variant. Loops like `for
 * (int i = 0, n = foo; i < n; i++)` is not recognized by OpenMP, and there is a
 * naming problem with `n`. Loops like `n = foo; for (int i = 0; i < n; i++)`
 * further introduces extra statements, and make it harder for naming and OpenMP
 * codegen.
 * - The type of the iterator can be clearly deduced. We can deduce the type of
 * `i` in `for (auto &&i : integerRange(begin, end, step, len))`. However, in
 * canonical loops like `for (auto i = 0; i < n; i++)`, the type of `i` cannot
 * be deduced from `n`.
 * - Range-based loops are supported by OpenMP.
 *
 * We keep two loop induction variables in an iterator: an "index" that counts
 * from `begin` to `end`, and a "counter" that counts from 0 to `len`. There two
 * varaibles can have different types.
 */
template <typename BeginType, typename EndType, typename StepType,
          typename LenType>
FUNC_ATTR constexpr auto integerRange(BeginType &&begin, EndType &&end,
                                      StepType &&step, LenType &&len) {
    // std::decay is called inside std::common_type
    return IntegerRange<
        std::make_signed_t<std::common_type_t<BeginType, EndType, StepType>>,
        std::make_signed_t<LenType>>{begin, end, step, len};
}

#endif // FREE_TENSOR_INTEGER_RANGE_H
