#ifndef FREE_TENSOR_ID_H
#define FREE_TENSOR_ID_H

#include <atomic>
#include <iostream>

namespace freetensor {

/**
 * Identify an Stmt acrossing passes, so we do not need to pass pointers
 *
 * An Stmt is identified by a string-typed id_ property, which is unique to each
 * Stmt node
 */
class ID {
    friend class StmtNode;

    int64_t id_;

    static std::atomic_int64_t globalIdCnt_;
    explicit ID(int64_t id) : id_(id) {}

  public:
    ID() : id_(-1) {}

    static ID make() { return ID(globalIdCnt_++); }
    static ID make(int64_t id) { return ID(id); }

    bool isValid() const { return id_ != -1; }

    friend std::ostream &operator<<(std::ostream &os, const ID &id);
    friend bool operator==(const ID &lhs, const ID &rhs);
    friend struct ::std::hash<ID>;
};

std::ostream &operator<<(std::ostream &os, const ID &id);

} // namespace freetensor

namespace std {

template <> struct hash<freetensor::ID> {
    size_t operator()(const freetensor::ID &id) const;
};

} // namespace std

#endif // FREE_TENSOR_ID_H
