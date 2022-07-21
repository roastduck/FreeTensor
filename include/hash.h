#ifndef FREE_TENSOR_HASH_H
#define FREE_TENSOR_HASH_H

#include <unordered_map>
#include <unordered_set>

#include <expr.h>
#include <hash_combine.h>
#include <stmt.h>

namespace freetensor {

class Hasher {
    static constexpr size_t P = 2147483647; // % P
    static constexpr size_t K1 = 179424673, B1 = 275604541;
    // (node type * K1 + B1) % P
    static constexpr size_t K2 = 373587883, B2 = 472882027;
    // ((current hash + non-permutable factor) * K2 + B2) % P
    // or
    // (current hash + permutable factor) % P
    static constexpr size_t K3 = 573259391, B3 = 674506081;
    // (finally * K3 + B3) % P

  public:
    static size_t compHash(const Tensor &t);
    static size_t compHash(const Buffer &b);
    static size_t compHash(const ReductionItem &r);
    static size_t compHash(const ForProperty &p);

    // stmt
    static size_t compHash(const AnyNode &op);
    static size_t compHash(const StmtSeqNode &op);
    static size_t compHash(const VarDefNode &op);
    static size_t compHash(const StoreNode &op);
    static size_t compHash(const AllocNode &op);
    static size_t compHash(const FreeNode &op);
    static size_t compHash(const ReduceToNode &op);
    static size_t compHash(const ForNode &op);
    static size_t compHash(const IfNode &op);
    static size_t compHash(const AssertNode &op);
    static size_t compHash(const AssumeNode &op);
    static size_t compHash(const EvalNode &op);
    static size_t compHash(const MatMulNode &op);

    // expr
    static size_t compHash(const CommutativeBinaryExprNode &op);
    static size_t compHash(const NonCommutativeBinaryExprNode &op);
    static size_t compHash(const UnaryExprNode &op);
    static size_t compHash(const AnyExprNode &op);
    static size_t compHash(const VarNode &op);
    static size_t compHash(const LoadNode &op);
    static size_t compHash(const IntConstNode &op);
    static size_t compHash(const FloatConstNode &op);
    static size_t compHash(const BoolConstNode &op);
    static size_t compHash(const IfExprNode &op);
    static size_t compHash(const CastNode &op);
    static size_t compHash(const IntrinsicNode &op);

    size_t operator()(const Ref<ASTPart> &op) const {
        return op.isValid() ? op->hash() : P;
    }
};

class HashComparator {
  private:
    // stmt
    bool compare(const Any &lhs, const Any &rhs) const;
    bool compare(const StmtSeq &lhs, const StmtSeq &rhs) const;
    bool compare(const VarDef &lhs, const VarDef &rhs) const;
    bool compare(const Store &lhs, const Store &rhs) const;
    bool compare(const Alloc &lhs, const Alloc &rhs) const;
    bool compare(const Free &lhs, const Free &rhs) const;
    bool compare(const ReduceTo &lhs, const ReduceTo &rhs) const;
    bool compare(const For &lhs, const For &rhs) const;
    bool compare(const If &lhs, const If &rhs) const;
    bool compare(const Assert &lhs, const Assert &rhs) const;
    bool compare(const Assume &lhs, const Assume &rhs) const;
    bool compare(const Eval &lhs, const Eval &rhs) const;
    bool compare(const MatMul &lhs, const MatMul &rhs) const;

    // expr
    bool compare(const CommutativeBinaryExpr &lhs,
                 const CommutativeBinaryExpr &rhs) const;
    bool compare(const NonCommutativeBinaryExpr &lhs,
                 const NonCommutativeBinaryExpr &rhs) const;
    bool compare(const UnaryExpr &lhs, const UnaryExpr &rhs) const;
    bool compare(const Var &lhs, const Var &rhs) const;
    bool compare(const IntConst &lhs, const IntConst &rhs) const;
    bool compare(const FloatConst &lhs, const FloatConst &rhs) const;
    bool compare(const BoolConst &lhs, const BoolConst &rhs) const;
    bool compare(const Load &lhs, const Load &rhs) const;
    bool compare(const IfExpr &lhs, const IfExpr &rhs) const;
    bool compare(const Cast &lhs, const Cast &rhs) const;
    bool compare(const Intrinsic &lhs, const Intrinsic &rhs) const;

  public:
    bool operator()(const Ref<Tensor> &lhs, const Ref<Tensor> &rhs) const;
    bool operator()(const Ref<Buffer> &lhs, const Ref<Buffer> &rhs) const;
    bool operator()(const Ref<ReductionItem> &lhs,
                    const Ref<ReductionItem> &rhs) const;
    bool operator()(const Ref<ForProperty> &lhs,
                    const Ref<ForProperty> &rhs) const;
    bool operator()(const AST &lhs, const AST &rhs) const;
};

template <class K, class V>
using ASTHashMap = std::unordered_map<K, V, Hasher, HashComparator>;

template <class K>
using ASTHashSet = std::unordered_set<K, Hasher, HashComparator>;

} // namespace freetensor

namespace std {

template <class T, class U> class hash<std::pair<T, U>> {
    std::hash<T> hashT_;
    std::hash<U> hashU_;

  public:
    size_t operator()(const std::pair<T, U> &pair) const {
        return freetensor::hashCombine(hashT_(pair.first), hashU_(pair.second));
    }
};

template <class... Ts> class hash<std::tuple<Ts...>> {
  private:
    template <class T> static size_t hashManyImpl(const T &first) {
        return std::hash<T>()(first);
    }

    template <class T, class... Args>
    static size_t hashManyImpl(const T &first, const Args &...others) {
        return freetensor::hashCombine(std::hash<T>()(first),
                                       hashManyImpl(others...));
    }

    // std::apply needs an unoverloaded invocable
    static size_t hashMany(const Ts &...args) { return hashManyImpl(args...); }

  public:
    size_t operator()(const std::tuple<Ts...> &t) const {
        return std::apply(hashMany, t);
    }
};

template <class T> class hash<std::vector<T>> {
    std::hash<T> hash_;

  public:
    size_t operator()(const std::vector<T> &vec) const {
        // Encode the size first, to distinguish between one vector and several
        // consecutive or nested vectors
        size_t h = std::hash<size_t>()(vec.size());

        for (auto &&item : vec) {
            h = freetensor::hashCombine(h, hash_(item));
        }
        return h;
    }
};

} // namespace std

#endif // FREE_TENSOR_HASH_H
