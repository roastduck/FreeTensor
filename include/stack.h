#ifndef STACK_H
#define STACK_H

#include <ref.h>

namespace ir {

template <class T> class Stack {
  public:
    struct Node {
        T data_;
        Ref<Node> prev_;
    };

  private:
    Ref<Node> tail_;
    size_t size_ = 0;

  public:
    Ref<Node> top() const { return tail_; }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    void push(const T &data) {
        auto node = Ref<Node>::make();
        *node = {data, tail_};
        tail_ = node;
        size_++;
    }

    void pop() {
        tail_ = tail_->prev_;
        size_--;
    }
};

} // namespace ir

#endif // STACK_H
