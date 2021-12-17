#ifndef PRESBURGER_H
#define PRESBURGER_H

#include <functional>

#include <math/isl.h>
#include <ref.h>

namespace ir {

/**
 * Presburger arithmetic
 *
 * Currently implemented with ISL
 *
 * Operations are evaluated in a lazy manner to perform some runtime
 * optimizations
 */

class PBMapOp;
class PBSetOp;

enum class PBOpType : int {
    None = 0,
    Complement,
    Reverse,
    LexMin,
    LexMax,
    Subtract,
    Intersect,
    Union,
    ApplyRange,
    ApplyDomain,
    Identity,
    LexGE,
    LexGT,
    Empty,
    Universe,
    Domain,
    Range,
};

class PBCtx {
    ISLCtx ctx_;

  public:
    const ISLCtx &get() const { return ctx_; }
};

struct PBMapNode {
    Ref<PBMapOp> eval_;
    ISLMap map_;
};

class PBMap {
    Ref<PBMapNode> node_;

  public:
    PBMap() {}
    PBMap(const PBCtx &ctx, const std::string &str);
    PBMap(const Ref<PBMapOp> &op);

    bool isValid() const { return node_.isValid(); }
    long useCount() const { return node_.useCount(); }

    PBOpType opType() const;
    Ref<PBMapOp> op() const;

    void exec() const;

    const ISLMap &copy() const;
    ISLMap &&move() const;

    bool empty() const { return copy().empty(); }
};

inline std::string toString(const PBMap &map) { return toString(map.copy()); }

inline std::ostream &operator<<(std::ostream &os, const PBMap &map) {
    return os << toString(map);
}

struct PBSetNode {
    Ref<PBSetOp> eval_;
    ISLSet set_;
};

class PBSet {
    Ref<PBSetNode> node_;

  public:
    PBSet() {}
    PBSet(const PBCtx &ctx, const std::string &str);
    PBSet(const Ref<PBSetOp> &op);

    bool isValid() const { return node_.isValid(); }
    long useCount() const { return node_.useCount(); }

    PBOpType opType() const;
    Ref<PBSetOp> op() const;

    void exec() const;

    const ISLSet &copy() const;
    ISLSet &&move() const;
};

inline std::string toString(const PBSet &set) { return toString(set.copy()); }

inline std::ostream &operator<<(std::ostream &os, const PBSet &set) {
    return os << toString(set);
}

struct PBSpaceNode {
    ISLSpace space_;
};

class PBSpace {
    Ref<PBSpaceNode> node_;

  public:
    PBSpace() {}

    bool isValid() const { return node_.isValid(); }
    long useCount() const { return node_.useCount(); }

    const ISLSpace &copy() const;
    ISLSpace &&move() const;

    friend PBSpace spaceAlloc(const PBCtx &ctx, unsigned nparam, unsigned nIn,
                              unsigned nOut);
    friend PBSpace spaceSetAlloc(const PBCtx &ctx, unsigned nparam,
                                 unsigned dim);

    template <class T> friend PBSpace spaceMapFromSet(T &&other) {
        PBSpace space;
        space.node_ = Ref<PBSpaceNode>::make();
        space.node_->space_ = spaceMapFromSet(other.copy());
        return space;
    }
};

inline PBSpace spaceAlloc(const PBCtx &ctx, unsigned nparam, unsigned nIn,
                          unsigned nOut) {
    PBSpace space;
    space.node_ = Ref<PBSpaceNode>::make();
    space.node_->space_ = spaceAlloc(ctx.get(), nparam, nIn, nOut);
    return space;
}

inline PBSpace spaceSetAlloc(const PBCtx &ctx, unsigned nparam, unsigned dim) {
    PBSpace space;
    space.node_ = Ref<PBSpaceNode>::make();
    space.node_->space_ = spaceSetAlloc(ctx.get(), nparam, dim);
    return space;
}

inline std::string toString(const PBSpace &space) {
    return toString(space.copy());
}

inline std::ostream &operator<<(std::ostream &os, const PBSpace &space) {
    return os << toString(space);
}

typedef ISLVal PBVal;

class PBMapOp {
  public:
    virtual ~PBMapOp() {}
    virtual PBOpType type() const = 0;
    virtual ISLMap exec() = 0;
};

class PBMapComplement : public PBMapOp {
    PBMap map_;

  public:
    template <class T> PBMapComplement(T &&map) : map_(std::forward<T>(map)) {}
    PBOpType type() const override { return PBOpType::Complement; }
    ISLMap exec() override;
};

class PBMapReverse : public PBMapOp {
    PBMap map_;

  public:
    template <class T> PBMapReverse(T &&map) : map_(std::forward<T>(map)) {}
    PBOpType type() const override { return PBOpType::Reverse; }
    ISLMap exec() override;
};

class PBMapLexMin : public PBMapOp {
    PBMap map_;

  public:
    template <class T> PBMapLexMin(T &&map) : map_(std::forward<T>(map)) {}
    PBOpType type() const override { return PBOpType::LexMin; }
    ISLMap exec() override;
};

class PBMapLexMax : public PBMapOp {
    PBMap map_;

  public:
    template <class T> PBMapLexMax(T &&map) : map_(std::forward<T>(map)) {}
    PBOpType type() const override { return PBOpType::LexMax; }
    ISLMap exec() override;
};

class PBMapSubtract : public PBMapOp {
    PBMap lhs_, rhs_;

  public:
    template <class T, class U>
    PBMapSubtract(T &&lhs, U &&rhs)
        : lhs_(std::forward<T>(lhs)), rhs_(std::forward<U>(rhs)) {}
    PBOpType type() const override { return PBOpType::Subtract; }
    ISLMap exec() override;
};

class PBMapIntersect : public PBMapOp {
    PBMap lhs_, rhs_;

  public:
    template <class T, class U>
    PBMapIntersect(T &&lhs, U &&rhs)
        : lhs_(std::forward<T>(lhs)), rhs_(std::forward<U>(rhs)) {}
    PBOpType type() const override { return PBOpType::Intersect; }
    ISLMap exec() override;
};

class PBMapUnion : public PBMapOp {
    PBMap lhs_, rhs_;

  public:
    template <class T, class U>
    PBMapUnion(T &&lhs, U &&rhs)
        : lhs_(std::forward<T>(lhs)), rhs_(std::forward<U>(rhs)) {}
    PBOpType type() const override { return PBOpType::Union; }
    ISLMap exec() override;
};

class PBMapApplyDomain : public PBMapOp {
    PBMap lhs_, rhs_;

  public:
    template <class T, class U>
    PBMapApplyDomain(T &&lhs, U &&rhs)
        : lhs_(std::forward<T>(lhs)), rhs_(std::forward<U>(rhs)) {}
    PBOpType type() const override { return PBOpType::ApplyDomain; }
    ISLMap exec() override;
};

class PBMapApplyRange : public PBMapOp {
    PBMap lhs_, rhs_;

  public:
    template <class T, class U>
    PBMapApplyRange(T &&lhs, U &&rhs)
        : lhs_(std::forward<T>(lhs)), rhs_(std::forward<U>(rhs)) {}
    PBOpType type() const override { return PBOpType::ApplyRange; }
    ISLMap exec() override;
};

class PBMapIdentity : public PBMapOp {
    PBSpace space_;

  public:
    template <class T>
    PBMapIdentity(T &&space) : space_(std::forward<T>(space)) {}
    PBOpType type() const override { return PBOpType::Identity; }
    ISLMap exec() override;
};

class PBMapLexGE : public PBMapOp {
    PBSpace space_;

  public:
    template <class T> PBMapLexGE(T &&space) : space_(std::forward<T>(space)) {}
    PBOpType type() const override { return PBOpType::LexGE; }
    ISLMap exec() override;
};

class PBMapLexGT : public PBMapOp {
    PBSpace space_;

  public:
    template <class T> PBMapLexGT(T &&space) : space_(std::forward<T>(space)) {}
    PBOpType type() const override { return PBOpType::LexGT; }
    ISLMap exec() override;
};

class PBMapEmpty : public PBMapOp {
    PBSpace space_;

  public:
    template <class T> PBMapEmpty(T &&space) : space_(std::forward<T>(space)) {}
    PBOpType type() const override { return PBOpType::Empty; }
    ISLMap exec() override;
};

class PBMapUniverse : public PBMapOp {
    PBSpace space_;

  public:
    template <class T>
    PBMapUniverse(T &&space) : space_(std::forward<T>(space)) {}
    PBOpType type() const override { return PBOpType::Universe; }
    ISLMap exec() override;
};

class PBSetOp {
  public:
    virtual ~PBSetOp() {}
    virtual PBOpType type() const = 0;
    virtual ISLSet exec() = 0;
};

class PBSetComplement : public PBSetOp {
    PBSet set_;

  public:
    template <class T> PBSetComplement(T &&set) : set_(std::forward<T>(set)) {}
    PBOpType type() const override { return PBOpType::Complement; }
    ISLSet exec() override;
};

class PBSetEmpty : public PBSetOp {
    PBSpace space_;

  public:
    template <class T> PBSetEmpty(T &&space) : space_(std::forward<T>(space)) {}
    PBOpType type() const override { return PBOpType::Empty; }
    ISLSet exec() override;
};

class PBSetUniverse : public PBSetOp {
    PBSpace space_;

  public:
    template <class T>
    PBSetUniverse(T &&space) : space_(std::forward<T>(space)) {}
    PBOpType type() const override { return PBOpType::Universe; }
    ISLSet exec() override;
};

class PBSetDomain : public PBSetOp {
    PBMap map_;

  public:
    template <class T> PBSetDomain(T &&map) : map_(std::forward<T>(map)) {}
    PBOpType type() const override { return PBOpType::Domain; }
    ISLSet exec() override;
};

class PBSetRange : public PBSetOp {
    PBMap map_;

  public:
    template <class T> PBSetRange(T &&map) : map_(std::forward<T>(map)) {}
    PBOpType type() const override { return PBOpType::Range; }
    ISLSet exec() override;
};

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSet>> * = nullptr>
PBSet complement(T &&set) {
    return PBSet(
        Ref<PBSetComplement>::make(PBSetComplement(std::forward<T>(set))));
}
template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBMap>> * = nullptr>
PBMap complement(T &&map) {
    return PBMap(
        Ref<PBMapComplement>::make(PBMapComplement(std::forward<T>(map))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBMap>> * = nullptr>
PBMap reverse(T &&map) {
    return PBMap(Ref<PBMapReverse>::make(PBMapReverse(std::forward<T>(map))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBMap>> * = nullptr>
PBMap lexmin(T &&map) {
    return PBMap(Ref<PBMapLexMin>::make(PBMapLexMin(std::forward<T>(map))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBMap>> * = nullptr>
PBMap lexmax(T &&map) {
    return PBMap(Ref<PBMapLexMax>::make(PBMapLexMax(std::forward<T>(map))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBMap>> * = nullptr>
PBSet domain(T &&map) {
    return PBSet(Ref<PBSetDomain>::make(PBSetDomain(std::forward<T>(map))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBMap>> * = nullptr>
PBSet range(T &&map) {
    return PBSet(Ref<PBSetRange>::make(PBSetRange(std::forward<T>(map))));
}

template <class T, class U,
          typename std::enable_if_t<std::is_same_v<std::decay_t<T>, PBMap>> * =
              nullptr,
          typename std::enable_if_t<std::is_same_v<std::decay_t<U>, PBMap>> * =
              nullptr>
PBMap subtract(T &&lhs, U &&rhs) {
    return PBMap(Ref<PBMapSubtract>::make(
        PBMapSubtract(std::forward<T>(lhs), std::forward<U>(rhs))));
}

template <class T, class U,
          typename std::enable_if_t<std::is_same_v<std::decay_t<T>, PBMap>> * =
              nullptr,
          typename std::enable_if_t<std::is_same_v<std::decay_t<U>, PBMap>> * =
              nullptr>
PBMap intersect(T &&lhs, U &&rhs) {
    return PBMap(Ref<PBMapIntersect>::make(
        PBMapIntersect(std::forward<T>(lhs), std::forward<U>(rhs))));
}

template <class T, class U,
          typename std::enable_if_t<std::is_same_v<std::decay_t<T>, PBMap>> * =
              nullptr,
          typename std::enable_if_t<std::is_same_v<std::decay_t<U>, PBMap>> * =
              nullptr>
PBMap uni(T &&lhs, U &&rhs) {
    return PBMap(Ref<PBMapUnion>::make(
        PBMapUnion(std::forward<T>(lhs), std::forward<U>(rhs))));
}

template <class T, class U,
          typename std::enable_if_t<std::is_same_v<std::decay_t<T>, PBMap>> * =
              nullptr,
          typename std::enable_if_t<std::is_same_v<std::decay_t<U>, PBMap>> * =
              nullptr>
PBMap applyDomain(T &&lhs, U &&rhs) {
    return PBMap(Ref<PBMapApplyDomain>::make(
        PBMapApplyDomain(std::forward<T>(lhs), std::forward<U>(rhs))));
}

template <class T, class U,
          typename std::enable_if_t<std::is_same_v<std::decay_t<T>, PBMap>> * =
              nullptr,
          typename std::enable_if_t<std::is_same_v<std::decay_t<U>, PBMap>> * =
              nullptr>
PBMap applyRange(T &&lhs, U &&rhs) {
    return PBMap(Ref<PBMapApplyRange>::make(
        PBMapApplyRange(std::forward<T>(lhs), std::forward<U>(rhs))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSpace>> * = nullptr>
PBMap identity(T &&space) {
    return PBMap(
        Ref<PBMapIdentity>::make(PBMapIdentity(std::forward<T>(space))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSpace>> * = nullptr>
PBMap lexGE(T &&space) {
    return PBMap(Ref<PBMapLexGE>::make(PBMapLexGE(std::forward<T>(space))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSpace>> * = nullptr>
PBMap lexGT(T &&space) {
    return PBMap(Ref<PBMapLexGT>::make(PBMapLexGT(std::forward<T>(space))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSpace>> * = nullptr>
PBMap emptyMap(T &&space) {
    return PBMap(Ref<PBMapEmpty>::make(PBMapEmpty(std::forward<T>(space))));
}
template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSpace>> * = nullptr>
PBSet emptySet(T &&space) {
    return PBSet(Ref<PBSetEmpty>::make(PBSetEmpty(std::forward<T>(space))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSpace>> * = nullptr>
PBMap universeMap(T &&space) {
    return PBMap(
        Ref<PBMapUniverse>::make(PBMapUniverse(std::forward<T>(space))));
}
template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSpace>> * = nullptr>
PBSet universeSet(T &&space) {
    return PBSet(
        Ref<PBSetUniverse>::make(PBSetUniverse(std::forward<T>(space))));
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSet>> * = nullptr>
PBVal dimMaxVal(T &&set, int pos) {
    return dimMaxVal(set.copy(), pos);
}

template <class T, typename std::enable_if_t<
                       std::is_same_v<std::decay_t<T>, PBSet>> * = nullptr>
PBVal dimMinVal(T &&set, int pos) {
    return dimMinVal(set.copy(), pos);
}

template <class T, class U,
          typename std::enable_if_t<std::is_same_v<std::decay_t<T>, PBSet>> * =
              nullptr,
          typename std::enable_if_t<std::is_same_v<std::decay_t<U>, PBSet>> * =
              nullptr>
inline bool operator==(T &&lhs, U &&rhs) {
    return lhs.copy() == rhs.copy();
}

template <class T, class U,
          typename std::enable_if_t<std::is_same_v<std::decay_t<T>, PBSet>> * =
              nullptr,
          typename std::enable_if_t<std::is_same_v<std::decay_t<U>, PBSet>> * =
              nullptr>
inline bool operator!=(T &&lhs, U &&rhs) {
    return lhs.copy() != rhs.copy();
}

} // namespace ir

#endif // PRESBURGER_H
