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

    isl_size nBasic() const { return isl_map_n_basic_map(map_); }

    friend std::string toString(const PBMap &map) {
        return isl_map_to_str(map.map_);
    }
};

inline std::ostream &operator<<(std::ostream &os, const PBMap &map) {
    return os << toString(map);
}

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

    friend std::string toString(const PBVal &val) {
        return isl_val_to_str(val.val_);
    }
};

inline std::ostream &operator<<(std::ostream &os, const PBVal &val) {
    return os << toString(val);
}

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

    friend std::string toString(const PBSet &set) {
        return isl_set_to_str(set.set_);
    }
};

inline std::ostream &operator<<(std::ostream &os, const PBSet &set) {
    return os << toString(set);
}

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

    friend std::string toString(const PBSpace &space) {
        return isl_space_to_str(space.space_);
    }
};

inline std::ostream &operator<<(std::ostream &os, const PBSpace &space) {
    return os << toString(space);
}

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

    friend std::string toString(const PBFunc &func) {
        return isl_pw_multi_aff_to_str(func.func_);
    }
};

inline std::ostream &operator<<(std::ostream &os, const PBFunc &func) {
    return os << toString(func);
}

inline PBSet complement(PBSet &&set) {
    DEBUG_PROFILE("complement");
    return isl_set_complement(set.move());
}
inline PBSet complement(const PBSet &set) {
    DEBUG_PROFILE("complement");
    return isl_set_complement(set.copy());
}
inline PBMap complement(PBMap &&map) {
    DEBUG_PROFILE("complement");
    return isl_map_complement(map.move());
}
inline PBMap complement(const PBMap &map) {
    DEBUG_PROFILE("complement");
    return isl_map_complement(map.copy());
}

inline PBMap reverse(PBMap &&map) {
    DEBUG_PROFILE("reverse");
    return isl_map_reverse(map.move());
}
inline PBMap reverse(const PBMap &map) {
    DEBUG_PROFILE("reverse");
    return isl_map_reverse(map.copy());
}

inline PBMap subtract(PBMap &&lhs, PBMap &&rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_map_subtract(lhs.move(), rhs.move());
}
inline PBMap subtract(const PBMap &lhs, PBMap &&rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_map_subtract(lhs.copy(), rhs.move());
}
inline PBMap subtract(PBMap &&lhs, const PBMap &rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_map_subtract(lhs.move(), rhs.copy());
}
inline PBMap subtract(const PBMap &lhs, const PBMap &rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_map_subtract(lhs.copy(), rhs.copy());
}
inline PBSet subtract(PBSet &&lhs, PBSet &&rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_set_subtract(lhs.move(), rhs.move());
}
inline PBSet subtract(const PBSet &lhs, PBSet &&rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_set_subtract(lhs.copy(), rhs.move());
}
inline PBSet subtract(PBSet &&lhs, const PBSet &rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_set_subtract(lhs.move(), rhs.copy());
}
inline PBSet subtract(const PBSet &lhs, const PBSet &rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return isl_set_subtract(lhs.copy(), rhs.copy());
}

inline PBSet intersect(PBSet &&lhs, PBSet &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_set_intersect(lhs.move(), rhs.move());
}
inline PBSet intersect(const PBSet &lhs, PBSet &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_set_intersect(lhs.copy(), rhs.move());
}
inline PBSet intersect(PBSet &&lhs, const PBSet &rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_set_intersect(lhs.move(), rhs.copy());
}
inline PBSet intersect(const PBSet &lhs, const PBSet &rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_set_intersect(lhs.copy(), rhs.copy());
}

inline PBMap intersect(PBMap &&lhs, PBMap &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_map_intersect(lhs.move(), rhs.move());
}
inline PBMap intersect(const PBMap &lhs, PBMap &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_map_intersect(lhs.copy(), rhs.move());
}
inline PBMap intersect(PBMap &&lhs, const PBMap &rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_map_intersect(lhs.move(), rhs.copy());
}
inline PBMap intersect(const PBMap &lhs, const PBMap &rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return isl_map_intersect(lhs.copy(), rhs.copy());
}

inline PBMap uni(PBMap &&lhs, PBMap &&rhs) {
    DEBUG_PROFILE_VERBOSE("uni", "nBasic=" + std::to_string(lhs.nBasic()) +
                                     "," + std::to_string(rhs.nBasic()));
    return isl_map_union(lhs.move(), rhs.move());
}
inline PBMap uni(const PBMap &lhs, PBMap &&rhs) {
    DEBUG_PROFILE_VERBOSE("uni", "nBasic=" + std::to_string(lhs.nBasic()) +
                                     "," + std::to_string(rhs.nBasic()));
    return isl_map_union(lhs.copy(), rhs.move());
}
inline PBMap uni(PBMap &&lhs, const PBMap &rhs) {
    DEBUG_PROFILE_VERBOSE("uni", "nBasic=" + std::to_string(lhs.nBasic()) +
                                     "," + std::to_string(rhs.nBasic()));
    return isl_map_union(lhs.move(), rhs.copy());
}
inline PBMap uni(const PBMap &lhs, const PBMap &rhs) {
    DEBUG_PROFILE_VERBOSE("uni", "nBasic=" + std::to_string(lhs.nBasic()) +
                                     "," + std::to_string(rhs.nBasic()));
    return isl_map_union(lhs.copy(), rhs.copy());
}

inline PBSet apply(PBSet &&lhs, PBMap &&rhs) {
    DEBUG_PROFILE("apply");
    return isl_set_apply(lhs.move(), rhs.move());
}
inline PBSet apply(const PBSet &lhs, PBMap &&rhs) {
    DEBUG_PROFILE("apply");
    return isl_set_apply(lhs.copy(), rhs.move());
}
inline PBSet apply(PBSet &&lhs, const PBMap &rhs) {
    DEBUG_PROFILE("apply");
    return isl_set_apply(lhs.move(), rhs.copy());
}
inline PBSet apply(const PBSet &lhs, const PBMap &rhs) {
    DEBUG_PROFILE("apply");
    return isl_set_apply(lhs.copy(), rhs.copy());
}

inline PBMap applyDomain(PBMap &&lhs, PBMap &&rhs) {
    DEBUG_PROFILE("applyDomain");
    return isl_map_apply_domain(lhs.move(), rhs.move());
}
inline PBMap applyDomain(const PBMap &lhs, PBMap &&rhs) {
    DEBUG_PROFILE("applyDomain");
    return isl_map_apply_domain(lhs.copy(), rhs.move());
}
inline PBMap applyDomain(PBMap &&lhs, const PBMap &rhs) {
    DEBUG_PROFILE("applyDomain");
    return isl_map_apply_domain(lhs.move(), rhs.copy());
}
inline PBMap applyDomain(const PBMap &lhs, const PBMap &rhs) {
    DEBUG_PROFILE("applyDomain");
    return isl_map_apply_domain(lhs.copy(), rhs.copy());
}

inline PBMap applyRange(PBMap &&lhs, PBMap &&rhs) {
    DEBUG_PROFILE("applyRange");
    return isl_map_apply_range(lhs.move(), rhs.move());
}
inline PBMap applyRange(const PBMap &lhs, PBMap &&rhs) {
    DEBUG_PROFILE("applyRange");
    return isl_map_apply_range(lhs.copy(), rhs.move());
}
inline PBMap applyRange(PBMap &&lhs, const PBMap &rhs) {
    DEBUG_PROFILE("applyRange");
    return isl_map_apply_range(lhs.move(), rhs.copy());
}
inline PBMap applyRange(const PBMap &lhs, const PBMap &rhs) {
    DEBUG_PROFILE("applyRange");
    return isl_map_apply_range(lhs.copy(), rhs.copy());
}

inline PBMap lexmax(PBMap &&map) {
    DEBUG_PROFILE_VERBOSE("lexmax", "nBasic=" + std::to_string(map.nBasic()));
    return isl_map_lexmax(map.move());
}
inline PBMap lexmax(const PBMap &map) {
    DEBUG_PROFILE_VERBOSE("lexmax", "nBasic=" + std::to_string(map.nBasic()));
    return isl_map_lexmax(map.copy());
}

inline PBMap lexmin(PBMap &&map) {
    DEBUG_PROFILE_VERBOSE("lexmin", "nBasic=" + std::to_string(map.nBasic()));
    return isl_map_lexmin(map.move());
}
inline PBMap lexmin(const PBMap &map) {
    DEBUG_PROFILE_VERBOSE("lexmin", "nBasic=" + std::to_string(map.nBasic()));
    return isl_map_lexmin(map.copy());
}

inline PBMap identity(PBSpace &&space) {
    DEBUG_PROFILE("identity");
    return isl_map_identity(space.move());
}
inline PBMap identity(const PBSpace &space) {
    DEBUG_PROFILE("identity");
    return isl_map_identity(space.copy());
}

inline PBMap lexGE(PBSpace &&space) {
    DEBUG_PROFILE("lexGE");
    return isl_map_lex_ge(space.move());
}
inline PBMap lexGE(const PBSpace &space) {
    DEBUG_PROFILE("lexGE");
    return isl_map_lex_ge(space.copy());
}

inline PBMap lexGT(PBSpace &&space) {
    DEBUG_PROFILE("lexGT");
    return isl_map_lex_gt(space.move());
}
inline PBMap lexGT(const PBSpace &space) {
    DEBUG_PROFILE("lexGT");
    return isl_map_lex_gt(space.copy());
}

inline PBSpace spaceAlloc(const PBCtx &ctx, unsigned nparam, unsigned nIn,
                          unsigned nOut) {
    return isl_space_alloc(ctx.get(), nparam, nIn, nOut);
}

inline PBSpace spaceSetAlloc(const PBCtx &ctx, unsigned nparam, unsigned dim) {
    return isl_space_set_alloc(ctx.get(), nparam, dim);
}

inline PBSet emptySet(PBSpace &&space) { return isl_set_empty(space.move()); }
inline PBSet emptySet(const PBSpace &space) {
    return isl_set_empty(space.copy());
}

inline PBMap emptyMap(PBSpace &&space) { return isl_map_empty(space.move()); }
inline PBMap emptyMap(const PBSpace &space) {
    return isl_map_empty(space.copy());
}

inline PBSet universeSet(PBSpace &&space) {
    return isl_set_universe(space.move());
}
inline PBSet universeSet(const PBSpace &space) {
    return isl_set_universe(space.copy());
}

inline PBMap universeMap(PBSpace &&space) {
    return isl_map_universe(space.move());
}
inline PBMap universeMap(const PBSpace &space) {
    return isl_map_universe(space.copy());
}

inline PBSet domain(PBMap &&map) { return isl_map_domain(map.move()); }
inline PBSet domain(const PBMap &map) { return isl_map_domain(map.copy()); }

inline PBSet range(PBMap &&map) { return isl_map_range(map.move()); }
inline PBSet range(const PBMap &map) { return isl_map_range(map.copy()); }

inline PBSet coalesce(PBSet &&set) {
    DEBUG_PROFILE("coalesce");
    return isl_set_coalesce(set.move());
}
inline PBSet coalesce(const PBSet &set) {
    DEBUG_PROFILE("coalesce");
    return isl_set_coalesce(set.copy());
}

inline PBMap coalesce(PBMap &&map) {
    DEBUG_PROFILE("coalesce");
    return isl_map_coalesce(map.move());
}
inline PBMap coalesce(const PBMap &map) {
    DEBUG_PROFILE("coalesce");
    return isl_map_coalesce(map.copy());
}

inline PBVal dimMaxVal(PBSet &&set, int pos) {
    return isl_set_dim_max_val(set.move(), pos);
}
inline PBVal dimMaxVal(const PBSet &set, int pos) {
    return isl_set_dim_max_val(set.copy(), pos);
}

inline PBVal dimMinVal(PBSet &&set, int pos) {
    return isl_set_dim_min_val(set.move(), pos);
}
inline PBVal dimMinVal(const PBSet &set, int pos) {
    return isl_set_dim_min_val(set.copy(), pos);
}

inline PBSpace spaceMapFromSet(PBSpace &&space) {
    return isl_space_map_from_set(space.move());
}
inline PBSpace spaceMapFromSet(const PBSpace &space) {
    return isl_space_map_from_set(space.copy());
}

inline bool operator==(const PBSet &lhs, const PBSet &rhs) {
    DEBUG_PROFILE_VERBOSE("equal", "nBasic=" + std::to_string(lhs.nBasic()) +
                                       "," + std::to_string(rhs.nBasic()));
    return isl_set_is_equal(lhs.get(), rhs.get());
}

inline bool operator!=(const PBSet &lhs, const PBSet &rhs) {
    DEBUG_PROFILE_VERBOSE("inequal", "nBasic=" + std::to_string(lhs.nBasic()) +
                                         "," + std::to_string(rhs.nBasic()));
    return !isl_set_is_equal(lhs.get(), rhs.get());
}

inline bool operator==(const PBMap &lhs, const PBMap &rhs) {
    DEBUG_PROFILE_VERBOSE("equal", "nBasic=" + std::to_string(lhs.nBasic()) +
                                       "," + std::to_string(rhs.nBasic()));
    return isl_map_is_equal(lhs.get(), rhs.get());
}

inline bool operator!=(const PBMap &lhs, const PBMap &rhs) {
    DEBUG_PROFILE_VERBOSE("inequal", "nBasic=" + std::to_string(lhs.nBasic()) +
                                         "," + std::to_string(rhs.nBasic()));
    return !isl_map_is_equal(lhs.get(), rhs.get());
}

} // namespace freetensor

#endif // FREE_TENSOR_PRESBURGER_H
