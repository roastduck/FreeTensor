#ifndef FREE_TENSOR_SHARED_LINKED_LIST_H
#define FREE_TENSOR_SHARED_LINKED_LIST_H

#include <functional>

#include <hash_combine.h>
#include <ref.h>

namespace freetensor {

/**
 * Read-only stack implemented via directed linked list
 *
 * A linked list linked from the stack top to the bottom. A `SharedLinkedList`
 * can be copied, and then all the common contents are shared, i.e., multiple
 * linked lists form a tree rooted at the stack bottom
 *
 * A `SharedLinkedList` can be hashed. Two `SharedLinkedList`s can be compared
 */
template <class T, class Hash = std::hash<T>, class Equal = std::equal_to<T>>
class SharedLinkedList {
    struct Node {
        T data_;
        size_t hash_; // Combined hash of all the nodes
        Ref<Node> prev_;
    };

    Ref<Node> tail_;

  public:
    const T &top() const {
        ASSERT(!empty());
        return tail_->data_;
    }

    bool empty() const { return !tail_.isValid(); }

    /**
     * Append a node
     *
     * This function returns a new `SharedLinkedList` object (a new tail), while
     * the original object is not changed. Use `list = list.push(...)` to update
     * a list
     */
    [[nodiscard]] SharedLinkedList push(const T &data) const {
        auto node = Ref<Node>::make();
        auto h = Hash()(data);
        if (tail_.isValid()) {
            h = hashCombine(tail_->hash_, h);
        }
        *node = {data, h, tail_};
        auto ret = *this;
        ret.tail_ = node;
        return ret;
    }

    /**
     * Drop the tail node
     *
     * This function returns a new `SharedLinkedList` object (a new tail), while
     * the original object is not changed. Use `list = list.pop()` to update a
     * list
     */
    [[nodiscard]] SharedLinkedList pop() const {
        auto ret = *this;
        ret.tail_ = tail_->prev_;
        return ret;
    }

    std::vector<T> asVector() const {
        std::vector<T> revRet;
        for (auto it = *this; !it.empty(); it = it.pop()) {
            revRet.emplace_back(it.top());
        }
        return std::vector<T>(revRet.rbegin(), revRet.rend());
    }

    size_t hash() const { return empty() ? 0 : tail_->hash_; }

    friend bool operator==(const SharedLinkedList &lhs,
                           const SharedLinkedList &rhs) {
        auto l = lhs, r = rhs;
        while (!l.empty() && !r.empty()) {
            if (l.tail_ == r.tail_) {
                // Same object, which means two shared linked list form a tree,
                // and this is their common ancestor, so no need to compare
                return true;
            }
            if (l.hash() != r.hash()) {
                return false;
            }
            if (!Equal()(l.top(), r.top())) {
                return false;
            }
            l = l.pop(), r = r.pop();
        }
        return l.empty() && r.empty();
    }
};

} // namespace freetensor

namespace std {

template <class T, class Hash, class Equal>
class hash<freetensor::SharedLinkedList<T, Hash, Equal>> {
  public:
    size_t operator()(
        const freetensor::SharedLinkedList<T, Hash, Equal> &stack) const {
        return stack.hash();
    }
};

} // namespace std

#endif // FREE_TENSOR_SHARED_LINKED_LIST_H
