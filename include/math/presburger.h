#ifndef FREE_TENSOR_PRESBURGER_H
#define FREE_TENSOR_PRESBURGER_H

#include <iostream>
#include <string>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/ilp.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>
#include <isl/space.h>

#include <debug/profile.h>
#include <except.h>
#include <serialize/to_string.h>

namespace freetensor {

// Presburger arithmetic, currently implemented with ISL

template <class T> T *GET_ISL_PTR(T *ptr) {
    ASSERT(ptr != nullptr);
    return ptr;
}

#define COPY_ISL_PTR(ptr, type) _COPY_ISL_PTR(ptr, isl_##type##_copy)
template <class T> T *_COPY_ISL_PTR(const T *ptr, T *(copy)(T *)) {
    ASSERT(ptr != nullptr);
    return copy(const_cast<T *>(ptr));
}

template <class T> T *MOVE_ISL_PTR(T *&ptr) {
    ASSERT(ptr != nullptr);
    auto ret = ptr;
    ptr = nullptr;
    return ret;
}

class PBCtx {
    isl_ctx *ctx_ = nullptr;

  public:
    PBCtx() : ctx_(isl_ctx_alloc()) {
        isl_options_set_on_error(ctx_, ISL_ON_ERROR_ABORT);
    }
    ~PBCtx() { isl_ctx_free(ctx_); }

    PBCtx(const PBCtx &other) = delete;
    PBCtx &operator=(const PBCtx &other) = delete;

    isl_ctx *get() const { return GET_ISL_PTR(ctx_); }
};

class PBMap {
    isl_map *map_ = nullptr;

  public:
    PBMap() {}
    PBMap(isl_map *map) : map_(map) {}
    PBMap(const PBCtx &ctx, const std::string &str)
        : map_(isl_map_read_from_str(ctx.get(), str.c_str())) {
        if (map_ == nullptr) {
            ERROR("Unable to construct an PBMap from " + str);
        }
    }
    ~PBMap() {
        if (map_ != nullptr) {
            isl_map_free(map_);
        }
    }

    PBMap(const PBMap &other) : map_(other.copy()) {}
    PBMap &operator=(const PBMap &other) {
        if (map_ != nullptr) {
            isl_map_free(map_);
        }
        map_ = other.copy();
        return *this;
    }

    PBMap(PBMap &&other) : map_(other.move()) {}
    PBMap &operator=(PBMap &&other) {
        if (map_ != nullptr) {
            isl_map_free(map_);
        }
        map_ = other.move();
        return *this;
    }

    bool isValid() const { return map_ != nullptr; }

    isl_map *get() const { return GET_ISL_PTR(map_); }
    isl_map *copy() const { return COPY_ISL_PTR(map_, map); }
    isl_map *move() { return MOVE_ISL_PTR(map_); }

    bool empty() const {
        DEBUG_PROFILE("empty");
        return isl_map_is_empty(get());
    }
    bool isSingleValued() const { return isl_map_is_single_valued(get()); }
    bool isBijective() const { return isl_map_is_bijective(get()); }

    isl_size nBasic() const { return isl_map_n_basic_map(map_); }

    friend std::ostream &operator<<(std::ostream &os, const PBMap &map) {
        return os << isl_map_to_str(map.map_);
    }
};

class PBVal {
    isl_val *val_ = nullptr;

  public:
    PBVal() {}
    PBVal(isl_val *val) : val_(val) {}
    ~PBVal() {
        if (val_ != nullptr) {
            isl_val_free(val_);
        }
    }

    PBVal(const PBVal &other) : val_(other.copy()) {}
    PBVal &operator=(const PBVal &other) {
        if (val_ != nullptr) {
            isl_val_free(val_);
        }
        val_ = other.copy();
        return *this;
    }

    PBVal(PBVal &&other) : val_(other.move()) {}
    PBVal &operator=(PBVal &&other) {
        if (val_ != nullptr) {
            isl_val_free(val_);
        }
        val_ = other.move();
        return *this;
    }

    bool isValid() const { return val_ != nullptr; }

    isl_val *get() const { return GET_ISL_PTR(val_); }
    isl_val *copy() const { return COPY_ISL_PTR(val_, val); }
    isl_val *move() { return MOVE_ISL_PTR(val_); }

    bool isRat() const { return isl_val_is_rat(get()); }
    int numSi() const { return isl_val_get_num_si(get()); }
    int denSi() const { return isl_val_get_den_si(get()); }

    friend std::ostream &operator<<(std::ostream &os, const PBVal &val) {
        return os << isl_val_to_str(val.val_);
    }
};

class PBSet {
    isl_set *set_ = nullptr;

  public:
    PBSet() {}
    PBSet(isl_set *set) : set_(set) {}
    PBSet(const PBCtx &ctx, const std::string &str)
        : set_(isl_set_read_from_str(ctx.get(), str.c_str())) {
        if (set_ == nullptr) {
            ERROR("Unable to construct an PBSet from " + str);
        }
    }
    ~PBSet() {
        if (set_ != nullptr) {
            isl_set_free(set_);
        }
    }

    PBSet(const PBSet &other) : set_(other.copy()) {}
    PBSet &operator=(const PBSet &other) {
        if (set_ != nullptr) {
            isl_set_free(set_);
        }
        set_ = other.copy();
        return *this;
    }

    PBSet(PBSet &&other) : set_(other.move()) {}
    PBSet &operator=(PBSet &&other) {
        if (set_ != nullptr) {
            isl_set_free(set_);
        }
        set_ = other.move();
        return *this;
    }

    bool isValid() const { return set_ != nullptr; }

    isl_set *get() const { return GET_ISL_PTR(set_); }
    isl_set *copy() const { return COPY_ISL_PTR(set_, set); }
    isl_set *move() { return MOVE_ISL_PTR(set_); }

    bool empty() const {
        DEBUG_PROFILE("empty");
        return isl_set_is_empty(get());
    }

    isl_size nBasic() const { return isl_set_n_basic_set(set_); }

    friend std::ostream &operator<<(std::ostream &os, const PBSet &set) {
        return os << isl_set_to_str(set.set_);
    }
};

class PBSpace {
    isl_space *space_ = nullptr;

  public:
    PBSpace() {}
    PBSpace(isl_space *space) : space_(space) {}
    PBSpace(const PBSet &set) : space_(isl_set_get_space(set.get())) {}
    ~PBSpace() {
        if (space_ != nullptr) {
            isl_space_free(space_);
        }
    }

    PBSpace(const PBSpace &other) : space_(other.copy()) {}
    PBSpace &operator=(const PBSpace &other) {
        if (space_ != nullptr) {
            isl_space_free(space_);
        }
        space_ = other.copy();
        return *this;
    }

    PBSpace(PBSpace &&other) : space_(other.move()) {}
    PBSpace &operator=(PBSpace &&other) {
        if (space_ != nullptr) {
            isl_space_free(space_);
        }
        space_ = other.move();
        return *this;
    }

    bool isValid() const { return space_ != nullptr; }

    isl_space *get() const { return GET_ISL_PTR(space_); }
    isl_space *copy() const { return COPY_ISL_PTR(space_, space); }
    isl_space *move() { return MOVE_ISL_PTR(space_); }

    bool operator==(const PBSpace &other) const {
        if (space_ == nullptr || other.space_ == nullptr)
            return space_ == other.space_;
        return isl_space_is_equal(get(), other.get());
    }

    friend std::ostream &operator<<(std::ostream &os, const PBSpace &space) {
        return os << isl_space_to_str(space.space_);
    }
};

class PBFunc {
    isl_pw_multi_aff *func_ = nullptr;

  public:
    PBFunc() {}
    PBFunc(isl_pw_multi_aff *func) : func_(func) {}
    PBFunc(const PBMap &map) : func_(isl_pw_multi_aff_from_map(map.copy())) {}
    PBFunc(PBMap &&map) : func_(isl_pw_multi_aff_from_map(map.move())) {}
    ~PBFunc() {
        if (func_ != nullptr) {
            isl_pw_multi_aff_free(func_);
        }
    }

    PBFunc(const PBFunc &other) : func_(other.copy()) {}
    PBFunc &operator=(const PBFunc &other) {
        if (func_ != nullptr) {
            isl_pw_multi_aff_free(func_);
        }
        func_ = other.copy();
        return *this;
    }

    PBFunc(PBFunc &&other) : func_(other.move()) {}
    PBFunc &operator=(PBFunc &&other) {
        if (func_ != nullptr) {
            isl_pw_multi_aff_free(func_);
        }
        func_ = other.move();
        return *this;
    }

    bool isValid() const { return func_ != nullptr; }

    isl_pw_multi_aff *get() const { return GET_ISL_PTR(func_); }
    isl_pw_multi_aff *copy() const { return COPY_ISL_PTR(func_, pw_multi_aff); }
    isl_pw_multi_aff *move() { return MOVE_ISL_PTR(func_); }

    friend std::ostream &operator<<(std::ostream &os, const PBFunc &func) {
        return os << isl_pw_multi_aff_to_str(func.func_);
    }
};

template <typename T>
concept PBMapRef = std::same_as<PBMap, std::decay_t<T>>;
template <typename T>
concept PBValRef = std::same_as<PBVal, std::decay_t<T>>;
template <typename T>
concept PBSetRef = std::same_as<PBSet, std::decay_t<T>>;
template <typename T>
concept PBSpaceRef = std::same_as<PBSpace, std::decay_t<T>>;
template <typename T>
concept PBFuncRef = std::same_as<PBFunc, std::decay_t<T>>;

template <typename T> auto PBRefTake(std::remove_reference_t<T> &t) {
    return t.copy();
}
template <typename T> auto PBRefTake(std::remove_reference_t<T> &&t) {
    static_assert(!std::is_lvalue_reference_v<T>); // similar to std::forward
    return t.move();
}

template <PBSetRef T> PBSet complement(T &&set) {
    DEBUG_PROFILE("complement");
    return isl_set_complement(PBRefTake<T>(set));
}
template <PBMapRef T> PBMap complement(T &&map) {
    DEBUG_PROFILE("complement");
    return isl_map_complement(PBRefTake<T>(map));
}

template <PBMapRef T> PBMap reverse(T &&map) {
    DEBUG_PROFILE("reverse");
    return isl_map_reverse(PBRefTake<T>(map));
}

template <PBMapRef T, PBMapRef U> PBMap subtract(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_map_subtract(PBRefTake<T>(lhs), PBRefTake<U>(rhs));
}
template <PBSetRef T, PBSetRef U> PBSet subtract(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_set_subtract(PBRefTake<T>(lhs), PBRefTake<U>(rhs));
}

template <PBMapRef T, PBMapRef U> PBMap intersect(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_map_intersect(PBRefTake<T>(lhs), PBRefTake<U>(rhs));
}
template <PBSetRef T, PBSetRef U> PBSet intersect(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_set_intersect(PBRefTake<T>(lhs), PBRefTake<U>(rhs));
}

template <PBMapRef T, PBMapRef U> PBMap uni(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("uni", "nBasic=" + std::to_string(lhs.nBasic()) +
                                     "," + std::to_string(rhs.nBasic()));
    return isl_map_union(PBRefTake<T>(lhs), PBRefTake<U>(rhs));
}

template <PBSetRef T, PBMapRef U> PBSet apply(T &&lhs, U &&rhs) {
    DEBUG_PROFILE("apply");
    return isl_set_apply(PBRefTake<T>(lhs), PBRefTake<U>(rhs));
}

template <PBMapRef T, PBMapRef U> PBMap applyDomain(T &&lhs, U &&rhs) {
    DEBUG_PROFILE("applyDomain");
    return isl_map_apply_domain(PBRefTake<T>(lhs), PBRefTake<U>(rhs));
}

template <PBMapRef T, PBMapRef U> PBMap applyRange(T &&lhs, U &&rhs) {
    DEBUG_PROFILE("applyRange");
    return isl_map_apply_range(PBRefTake<T>(lhs), PBRefTake<U>(rhs));
}

template <PBMapRef T> PBMap lexmax(T &&map) {
    DEBUG_PROFILE_VERBOSE("lexmax", "nBasic=" + std::to_string(map.nBasic()));
    return isl_map_lexmax(PBRefTake<T>(map));
}

template <PBMapRef T> PBMap lexmin(T &&map) {
    DEBUG_PROFILE_VERBOSE("lexmin", "nBasic=" + std::to_string(map.nBasic()));
    return isl_map_lexmin(PBRefTake<T>(map));
}

template <PBSpaceRef T> PBMap identity(T &&space) {
    DEBUG_PROFILE("identity");
    return isl_map_identity(PBRefTake<T>(space));
}

template <PBSpaceRef T> PBMap lexGE(T &&space) {
    DEBUG_PROFILE("lexGE");
    return isl_map_lex_ge(PBRefTake<T>(space));
}

template <PBSpaceRef T> PBMap lexGT(T &&space) {
    DEBUG_PROFILE("lexGT");
    return isl_map_lex_gt(PBRefTake<T>(space));
}

template <PBSpaceRef T> PBMap lexLE(T &&space) {
    DEBUG_PROFILE("lexLE");
    return isl_map_lex_le(PBRefTake<T>(space));
}

template <PBSpaceRef T> PBMap lexLT(T &&space) {
    DEBUG_PROFILE("lexLT");
    return isl_map_lex_lt(PBRefTake<T>(space));
}

inline PBMap lexLE(PBSpace &&space) {
    DEBUG_PROFILE("lexLE");
    return isl_map_lex_le(space.move());
}
inline PBMap lexLE(const PBSpace &space) {
    DEBUG_PROFILE("lexLE");
    return isl_map_lex_le(space.copy());
}

inline PBMap lexLT(PBSpace &&space) {
    DEBUG_PROFILE("lexLT");
    return isl_map_lex_lt(space.move());
}
inline PBMap lexLT(const PBSpace &space) {
    DEBUG_PROFILE("lexLT");
    return isl_map_lex_lt(space.copy());
}

inline PBSpace spaceAlloc(const PBCtx &ctx, unsigned nparam, unsigned nIn,
                          unsigned nOut) {
    return isl_space_alloc(ctx.get(), nparam, nIn, nOut);
}

inline PBSpace spaceSetAlloc(const PBCtx &ctx, unsigned nparam, unsigned dim) {
    return isl_space_set_alloc(ctx.get(), nparam, dim);
}

template <PBSpaceRef T> PBSet emptySet(T &&space) {
    return isl_set_empty(PBRefTake<T>(space));
}

template <PBSpaceRef T> PBMap emptyMap(T &&space) {
    return isl_map_empty(PBRefTake<T>(space));
}

template <PBSpaceRef T> PBSet universeSet(T &&space) {
    return isl_set_universe(PBRefTake<T>(space));
}

template <PBSpaceRef T> PBMap universeMap(T &&space) {
    return isl_map_universe(PBRefTake<T>(space));
}

template <PBMapRef T> PBSet domain(T &&map) {
    return isl_map_domain(PBRefTake<T>(map));
}

template <PBMapRef T> PBSet range(T &&map) {
    return isl_map_range(PBRefTake<T>(map));
}

template <PBSetRef T> PBSet coalesce(T &&set) {
    DEBUG_PROFILE("coalesce");
    return isl_set_coalesce(PBRefTake<T>(set));
}

template <PBMapRef T> PBMap coalesce(T &&map) {
    DEBUG_PROFILE("coalesce");
    return isl_map_coalesce(PBRefTake<T>(map));
}

template <PBSetRef T> PBVal dimMaxVal(T &&set, int pos) {
    return isl_set_dim_max_val(PBRefTake<T>(set), pos);
}

template <PBSetRef T> PBVal dimMinVal(T &&set, int pos) {
    return isl_set_dim_min_val(PBRefTake<T>(set), pos);
}

template <PBSpaceRef T> PBSpace spaceMapFromSet(T &&space) {
    return isl_space_map_from_set(PBRefTake<T>(space));
}

inline bool operator==(const PBSet &lhs, const PBSet &rhs) {
    DEBUG_PROFILE_VERBOSE("equal", "nBasic=" + std::to_string(lhs.nBasic()) +
                                       "," + std::to_string(rhs.nBasic()));
    return isl_set_is_equal(lhs.get(), rhs.get());
}

inline bool operator==(const PBMap &lhs, const PBMap &rhs) {
    DEBUG_PROFILE_VERBOSE("equal", "nBasic=" + std::to_string(lhs.nBasic()) +
                                       "," + std::to_string(rhs.nBasic()));
    return isl_map_is_equal(lhs.get(), rhs.get());
}

} // namespace freetensor

#endif // FREE_TENSOR_PRESBURGER_H
