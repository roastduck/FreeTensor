#ifndef SUB_TREE_H
#define SUB_TREE_H

#include <vector>

#include <ref.h>

namespace ir {

/**
 * An object that can be managed by SubTree
 *
 * Can be derived as an ASTNode or part of an ASTNode
 */
class ASTPart {
    bool isSubTree_ = false;

  public:
    ASTPart() : isSubTree_(false) {}

    // Construct a new part using another part. The new part will not initially
    // have a parent, and the other part will keep its parent
    ASTPart(ASTPart &&other) : isSubTree_(false) {}
    ASTPart(const ASTPart &other) : isSubTree_(false) {}

    // Assign the other part to the current part, but the parent of the current
    // part will still be its parent
    ASTPart &operator=(ASTPart &&) { return *this; }
    ASTPart &operator=(const ASTPart &) { return *this; }

    void setIsSubTree(bool isSubTree = true) { isSubTree_ = isSubTree; }
    bool isSubTree() const { return isSubTree_; }
};

enum NullPolicy : int { NotNull, Nullable };

/**
 * Plugging a Ref as a sub-tree in the AST
 *
 * This class ensures that each Ref of an AST node having a single parent. In
 * other words, there will not be two nodes in one AST sharing the same address.
 * If an AST node is assigned as a SubTree, but it has already been in another
 * SubTree, it will be automatically copied
 */
template <class T, NullPolicy POLICY = NullPolicy::NotNull> class SubTree {
    Ref<T> obj_;

    template <class, NullPolicy> friend class SubTree;

  public:
    SubTree() {}
    SubTree(std::nullptr_t) { ASSERT(POLICY == NullPolicy::Nullable); }

    ~SubTree() {
        if (obj_.isValid()) {
            obj_->setIsSubTree(false);
        }
    }

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTree(const Ref<U> &obj) : obj_(obj) {
        if (obj_.isValid()) {
            if (obj_->isSubTree()) {
                obj_ = deepCopy(obj).template as<T>();
            }
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
    }
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTree(Ref<U> &&obj) : obj_(obj) {
        if (obj_.isValid()) {
            if (obj_->isSubTree()) {
                obj_ = deepCopy(obj).template as<T>();
            }
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
    }

    SubTree(SubTree &&other) : obj_(std::move(other.obj_)) {
        if (!obj_.isValid()) {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
    }
    template <NullPolicy OTHER_POLICY>
    SubTree(SubTree<T, OTHER_POLICY> &&other) : obj_(std::move(other.obj_)) {
        if (!obj_.isValid()) {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
    }

    /**
     * For a `SubTree y`, `auto x = y` will result in a deep copy of the entire
     * `SubTree`. We avoid this misuse by making the copy constructor explicit.
     * Please use `Ref<T> x = y` or `auto &&x = y` instead
     */
    explicit SubTree(const SubTree &other) {
        if (other.obj_.isValid()) {
            obj_ = deepCopy(other.obj_).template as<T>();
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
            obj_ = nullptr;
        }
    }
    template <NullPolicy OTHER_POLICY>
    explicit SubTree(const SubTree<T, OTHER_POLICY> &other) {
        if (other.obj_.isValid()) {
            obj_ = deepCopy(other.obj_).template as<T>();
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
            obj_ = nullptr;
        }
    }

    SubTree &operator=(SubTree &&other) {
        if (obj_.isValid()) {
            obj_->setIsSubTree(false);
        }
        obj_ = std::move(other.obj_);
        if (!obj_.isValid()) {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
        return *this;
    }
    template <NullPolicy OTHER_POLICY>
    SubTree &operator=(SubTree<T, OTHER_POLICY> &&other) {
        if (obj_.isValid()) {
            obj_->setIsSubTree(false);
        }
        obj_ = std::move(other.obj_);
        if (!obj_.isValid()) {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
        return *this;
    }

    SubTree &operator=(const SubTree &other) {
        if (obj_.isValid()) {
            obj_->setIsSubTree(false);
        }
        if (other.obj_.isValid()) {
            obj_ = deepCopy(other.obj_).template as<T>();
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
            obj_ = nullptr;
        }
        return *this;
    }
    template <NullPolicy OTHER_POLICY>
    SubTree &operator=(const SubTree<T, OTHER_POLICY> &other) {
        if (obj_.isValid()) {
            obj_->setIsSubTree(false);
        }
        if (other.obj_.isValid()) {
            obj_ = deepCopy(other.obj_).template as<T>();
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
            obj_ = nullptr;
        }
        return *this;
    }

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<U, T>> * = nullptr>
    operator Ref<U>() const {
        return obj_;
    }

    T &operator*() const { return *obj_; }
    T *operator->() const { return obj_.get(); }

    template <class U> Ref<U> as() const { return obj_.template as<U>(); }

    bool isValid() const { return obj_.isValid(); }
};

/**
 * A list of SubTree
 *
 * This class can be used as a list of its inner AST nodes
 */
template <class T, NullPolicy POLICY = NullPolicy::NotNull> class SubTreeList {
    std::vector<SubTree<T, POLICY>> objs_;

    template <class, NullPolicy> friend class SubTree;

  public:
    SubTreeList() {}

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTreeList(const std::vector<Ref<U>> &objs)
        : objs_(objs.begin(), objs.end()) {}
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTreeList(std::vector<Ref<U>> &&objs) {
        objs_.reserve(objs.size());
        for (auto &obj : objs) {
            objs_.emplace_back(std::move(obj));
        }
    }
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTreeList(std::initializer_list<Ref<U>> objs) {
        objs_.reserve(objs.size());
        for (auto &obj : objs) {
            objs_.emplace_back(std::move(obj));
        }
    }

    SubTreeList(SubTreeList<T, POLICY> &&other)
        : objs_(std::move(other.objs_)) {}
    explicit SubTreeList(const SubTreeList<T, POLICY> &other)
        : objs_(other.objs_) {}

    template <NullPolicy OTHER_POLICY>
    SubTreeList(SubTreeList<T, OTHER_POLICY> &&other) {
        objs_.reserve(other.objs_.size());
        for (auto &obj : other.objs_) {
            objs_.emplace_back(std::move(obj));
        }
    }
    template <NullPolicy OTHER_POLICY>
    SubTreeList(const SubTreeList<T, OTHER_POLICY> &other)
        : objs_(other.objs_.begin(), other.objs_.end()) {}

    SubTreeList &operator=(SubTreeList<T, POLICY> &&other) {
        objs_ = std::move(other.objs_);
        return *this;
    }
    SubTreeList &operator=(const SubTreeList<T, POLICY> &other) {
        objs_ = other.objs_;
        return *this;
    }

    template <NullPolicy OTHER_POLICY>
    SubTreeList &operator=(SubTreeList<T, OTHER_POLICY> &&other) {
        objs_.clear();
        objs_.reserve(other.objs_.size());
        for (auto &obj : other.objs_) {
            objs_.emplace_back(std::move(obj));
        }
        return *this;
    }
    template <NullPolicy OTHER_POLICY>
    SubTreeList &operator=(const SubTreeList<T, OTHER_POLICY> &other) {
        objs_ = std::vector<SubTree<T, POLICY>>(other.objs_.begin(),
                                                other.objs_.end());
        return *this;
    }

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<U, T>> * = nullptr>
    operator std::vector<Ref<U>>() const {
        return std::vector<Ref<U>>(objs_.begin(), objs_.end());
    }

    auto begin() { return objs_.begin(); }
    auto begin() const { return objs_.begin(); }
    auto end() { return objs_.end(); }
    auto end() const { return objs_.end(); }
    auto rbegin() { return objs_.rbegin(); }
    auto rbegin() const { return objs_.rbegin(); }
    auto rend() { return objs_.rend(); }
    auto rend() const { return objs_.rend(); }
    auto size() const { return objs_.size(); }
    auto empty() const { return objs_.empty(); }
    template <class U> auto &operator[](U &&i) {
        return objs_[std::forward<U>(i)];
    }
    template <class U> const auto &operator[](U &&i) const {
        return objs_[std::forward<U>(i)];
    }
    template <class U> auto &at(U &&i) { return objs_.at(std::forward<U>(i)); }
    template <class U> const auto &at(U &&i) const {
        return objs_.at(std::forward<U>(i));
    }
    auto &front() { return objs_.front(); }
    const auto &front() const { return objs_.front(); }
    auto &back() { return objs_.back(); }
    const auto &back() const { return objs_.back(); }
    template <class U> void emplace_back(U &&x) {
        objs_.emplace_back(std::forward<U>(x));
    }
    template <class U> void push_back(U &&x) {
        objs_.push_back(std::forward<U>(x));
    }
    template <class... U> auto insert(U &&...i) {
        return objs_.insert(std::forward<U>(i)...);
    }
    template <class U> auto erase(U &&i) {
        return objs_.erase(std::forward<U>(i));
    }
    void pop_back() { objs_.pop_back(); }
};

} // namespace ir

#endif // SUB_TREE_H
