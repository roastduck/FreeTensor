#ifndef SUB_TREE_H
#define SUB_TREE_H

#include <vector>

#include <ref.h>

namespace ir {

class ASTPart;

/**
 * Explicitly mark a SubTree's parent
 *
 * A `ChildOf` can be initialized before the `Ref` of its child node being
 * initialized, so we store its raw pointer rather than a `Weak`. It is OK
 * because a `ChildOf` is only meant to be a temporary object
 */
struct ChildOf {
    ASTPart *parent_;
};

/**
 * An object that can be managed by SubTree
 *
 * Can be derived as an ASTNode or part of an ASTNode
 */
class ASTPart : public EnableSelf<ASTPart> {
    Weak<ASTPart> parent_;

  public:
    ASTPart() {}

    // Construct a new part using another part. The new part will not initially
    // have a parent, and the other part will keep its parent
    ASTPart(ASTPart &&other) {}
    ASTPart(const ASTPart &other) {}

    // Assign the other part to the current part, but the parent of the current
    // part will still be its parent
    ASTPart &operator=(ASTPart &&) { return *this; }
    ASTPart &operator=(const ASTPart &) { return *this; }

    void setParent(const Ref<ASTPart> &parent) { parent_ = parent; }
    void resetParent() { parent_ = nullptr; }
    bool isSubTree() const { return parent_.isValid(); }
};

enum NullPolicy : int { NotNull, Nullable };

/**
 * Plugging a `Ref` of `ASTPart` as a sub-tree in the AST
 *
 * This class ensures that each `Ref` of an `ASTPart` having a single parent. In
 * other words, there will not be two `ASTPart`s in one AST sharing the same
 * address. If an `ASTPart` is assigned as a `SubTree`, but it has already been
 * in another `SubTree`, it will be automatically copied
 */
template <class T, NullPolicy POLICY = NullPolicy::NotNull> class SubTree {
    Ref<T> obj_;

    /**
     * The parent of this SubTree. Using a raw pointer here is OK because the
     * parent of a SubTree will always be alive given the SubTree is alive
     */
    ASTPart *parent_ = nullptr;

    template <class, NullPolicy> friend class SubTree;

  private:
    void reset() {
        if (obj_.isValid()) {
            obj_->resetParent();
        }
        obj_ = nullptr;
    }

    void checkNull() {
        if (!obj_.isValid() && POLICY == NullPolicy::NotNull) {
            ERROR("Cannot assign a null Ref to a NotNull SubTree");
        }
    }

  public:
    SubTree(const ChildOf &c) : parent_(c.parent_) {}
    ~SubTree() { reset(); }

    SubTree(std::nullptr_t) { ASSERT(POLICY == NullPolicy::Nullable); }

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTree(const Ref<U> &obj) : obj_(obj) {
        if (obj_.isValid()) {
            if (obj_->isSubTree()) {
                obj_ = deepCopy(obj).template as<T>();
            }
            ASSERT(!obj_->isSubTree());
        }
        checkNull();
    }
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTree(Ref<U> &&obj) : obj_(obj) {
        if (obj_.isValid()) {
            if (obj_->isSubTree()) {
                obj_ = deepCopy(obj).template as<T>();
            }
            ASSERT(!obj_->isSubTree());
        }
        checkNull();
    }

    SubTree(SubTree &&other) : obj_(std::move(other.obj_)) { checkNull(); }
    template <NullPolicy OTHER_POLICY>
    SubTree(SubTree<T, OTHER_POLICY> &&other) : obj_(std::move(other.obj_)) {
        checkNull();
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
        } else {
            obj_ = nullptr;
        }
        checkNull();
    }
    template <NullPolicy OTHER_POLICY>
    explicit SubTree(const SubTree<T, OTHER_POLICY> &other) {
        if (other.obj_.isValid()) {
            obj_ = deepCopy(other.obj_).template as<T>();
            ASSERT(!obj_->isSubTree());
        } else {
            obj_ = nullptr;
        }
        checkNull();
    }

    SubTree &operator=(SubTree &&other) {
        reset();
        obj_ = std::move(other.obj_);
        checkNull();
        return *this;
    }
    template <NullPolicy OTHER_POLICY>
    SubTree &operator=(SubTree<T, OTHER_POLICY> &&other) {
        reset();
        obj_ = std::move(other.obj_);
        checkNull();
        return *this;
    }

    SubTree &operator=(const SubTree &other) {
        reset();
        if (other.obj_.isValid()) {
            obj_ = deepCopy(other.obj_).template as<T>();
            ASSERT(!obj_->isSubTree());
            if (parent_ != nullptr) {
                obj_->setParent(
                    ((EnableSelf<typename T::Self> *)parent_)->self());
            }
        } else {
            obj_ = nullptr;
        }
        checkNull();
        return *this;
    }
    template <NullPolicy OTHER_POLICY>
    SubTree &operator=(const SubTree<T, OTHER_POLICY> &other) {
        reset();
        if (other.obj_.isValid()) {
            obj_ = deepCopy(other.obj_).template as<T>();
            ASSERT(!obj_->isSubTree());
            if (parent_ != nullptr) {
                obj_->setParent(
                    ((EnableSelf<typename T::Self> *)parent_)->self());
            }
        } else {
            obj_ = nullptr;
        }
        checkNull();
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

    /**
     * The parent of this SubTree. Using a raw pointer here is OK because the
     * parent of a SubTree will always be alive given the SubTree is alive
     */
    ASTPart *parent_ = nullptr;

    template <class, NullPolicy> friend class SubTree;

  public:
    SubTreeList(const ChildOf &c) : parent_(c.parent_) {}

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTreeList(const std::vector<Ref<U>> &objs) {
        objs_.reserve(objs.size());
        for (auto &&item : objs) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = item;
            objs_.emplace_back(std::move(newItem));
        }
    }
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTreeList(std::vector<Ref<U>> &&objs) {
        objs_.reserve(objs.size());
        for (auto &&item : objs) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = std::move(item);
            objs_.emplace_back(std::move(newItem));
        }
    }
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTreeList(std::initializer_list<Ref<U>> objs) {
        objs_.reserve(objs.size());
        for (auto &&item : objs) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = std::move(item);
            objs_.emplace_back(std::move(newItem));
        }
    }

    SubTreeList(SubTreeList &&other) {
        objs_.reserve(other.objs_.size());
        for (auto &&item : other.objs_) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = std::move(item);
            objs_.emplace_back(std::move(newItem));
        }
    }
    template <NullPolicy OTHER_POLICY>
    SubTreeList(SubTreeList<T, OTHER_POLICY> &&other) {
        objs_.reserve(other.objs_.size());
        for (auto &&item : other.objs_) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = std::move(item);
            objs_.emplace_back(std::move(newItem));
        }
    }

    /**
     * For a `SubTreeList y`, `auto x = y` will result in a deep copy of the
     * entire `SubTreeList`. We avoid this misuse by making the copy constructor
     * explicit. Please use `std::vector<Ref<T>> x = y` or `auto &&x = y`
     * instead
     */
    explicit SubTreeList(const SubTreeList &other) {
        objs_.reserve(other.objs_.size());
        for (auto &&item : other.objs_) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = item;
            objs_.emplace_back(std::move(newItem));
        }
    }
    template <NullPolicy OTHER_POLICY>
    explicit SubTreeList(const SubTreeList<T, OTHER_POLICY> &other) {
        objs_.reserve(other.objs_.size());
        for (auto &&item : other.objs_) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = item;
            objs_.emplace_back(std::move(newItem));
        }
    }

    SubTreeList &operator=(SubTreeList &&other) {
        objs_.clear();
        objs_.reserve(other.objs_.size());
        for (auto &&item : other.objs_) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = std::move(item);
            objs_.emplace_back(std::move(newItem));
        }
        return *this;
    }
    SubTreeList &operator=(const SubTreeList &other) {
        objs_.clear();
        objs_.reserve(other.objs_.size());
        for (auto &&item : other.objs_) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = item;
            objs_.emplace_back(std::move(newItem));
        }
        return *this;
    }

    template <NullPolicy OTHER_POLICY>
    SubTreeList &operator=(SubTreeList<T, OTHER_POLICY> &&other) {
        objs_.clear();
        objs_.reserve(other.objs_.size());
        for (auto &&item : other.objs_) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = std::move(item);
            objs_.emplace_back(std::move(newItem));
        }
        return *this;
    }
    template <NullPolicy OTHER_POLICY>
    SubTreeList &operator=(const SubTreeList<T, OTHER_POLICY> &other) {
        objs_.clear();
        objs_.reserve(other.objs_.size());
        for (auto &&item : other.objs_) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = item;
            objs_.emplace_back(std::move(newItem));
        }
        return *this;
    }

    SubTreeList &operator=(std::initializer_list<Ref<T>> &&objs) {
        objs_.clear();
        objs_.reserve(objs.size());
        for (auto &&item : objs) {
            SubTree<T, POLICY> newItem = ChildOf{parent_};
            newItem = std::move(item);
            objs_.emplace_back(std::move(newItem));
        }
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
    template <class U> void reserve(U &&x) {
        objs_.reserve(std::forward<U>(x));
    }
    void clear() { objs_.clear(); }
};

} // namespace ir

#endif // SUB_TREE_H
