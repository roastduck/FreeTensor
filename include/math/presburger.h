#ifndef FREE_TENSOR_PRESBURGER_H
#define FREE_TENSOR_PRESBURGER_H

#include <cstddef>
#include <iostream>
#include <isl/id_type.h>
#include <string>
#include <unordered_set>
#include <vector>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/ilp.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/space_type.h>
#include <isl/val.h>

#include <debug.h>
#include <debug/profile.h>
#include <except.h>
#include <ref.h>
#include <serialize/to_string.h>
#include <timeout.h>

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

/**
 * Context for presburger operation
 *
 * - All operands of a presburger operation should be on the same context.
 * - Operations in the same context is NOT thread-safe. Explitly transfer to a
 * different context if you want to use in multiple threads.
 */
class PBCtx {
    isl_ctx *ctx_ = nullptr;

  public:
    PBCtx() : ctx_(isl_ctx_alloc()) {
        isl_options_set_on_error(ctx_, ISL_ON_ERROR_ABORT);
    }
    ~PBCtx() { isl_ctx_free(ctx_); }

    PBCtx(const PBCtx &other) = delete;
    PBCtx &operator=(const PBCtx &other) = delete;
    PBCtx(PBCtx &&other) = delete;
    PBCtx &operator=(PBCtx &&other) = delete;

    isl_ctx *get() const { return GET_ISL_PTR(ctx_); }
};

class PBMap {
    Ref<PBCtx> ctx_;
    isl_map *map_ = nullptr;

  public:
    PBMap() {}
    PBMap(const Ref<PBCtx> &ctx, isl_map *map) : ctx_(ctx), map_(map) {}
    PBMap(const Ref<PBCtx> &ctx, const std::string &str)
        : ctx_(ctx), map_(isl_map_read_from_str(ctx->get(), str.c_str())) {
        if (map_ == nullptr) {
            ERROR("Unable to construct an PBMap from " + str);
        }
    }
    ~PBMap() {
        if (map_ != nullptr) {
            isl_map_free(map_);
        }
    }

    PBMap(const PBMap &other) : ctx_(other.ctx_), map_(other.copy()) {}
    PBMap &operator=(const PBMap &other) {
        ctx_ = other.ctx_;
        if (map_ != nullptr) {
            isl_map_free(map_);
        }
        map_ = other.copy();
        return *this;
    }

    PBMap(PBMap &&other) : ctx_(std::move(other.ctx_)), map_(other.move()) {}
    PBMap &operator=(PBMap &&other) {
        ctx_ = std::move(other.ctx_);
        if (map_ != nullptr) {
            isl_map_free(map_);
        }
        map_ = other.move();
        return *this;
    }

    bool isValid() const { return map_ != nullptr; }

    const auto &ctx() const { return ctx_; }
    auto &ctx() { return ctx_; }

    isl_map *get() const { return GET_ISL_PTR(map_); }
    isl_map *copy() const { return COPY_ISL_PTR(map_, map); }
    isl_map *move() { return MOVE_ISL_PTR(map_); }

    class Serialized {
        std::string data_;

      public:
        Serialized() {}
        Serialized(const std::string &data) : data_(data) {}
        PBMap to(const Ref<PBCtx> &ctx) const { return {ctx, data_}; }
        bool isValid() const { return !data_.empty(); }
        const auto &data() const { return data_; }
        friend std::ostream &operator<<(std::ostream &os, const Serialized &s) {
            return os << s.data_;
        }
    };
    Serialized toSerialized() const { return {isl_map_to_str(get())}; }
    PBMap to(const Ref<PBCtx> &ctx) const { return toSerialized().to(ctx); }

    bool empty() const {
        DEBUG_PROFILE("empty");
        return isl_map_is_empty(get());
    }
    bool isSingleValued() const { return isl_map_is_single_valued(get()); }
    bool isBijective() const { return isl_map_is_bijective(get()); }

    isl_size nBasic() const { return isl_map_n_basic_map(get()); }

    isl_size nInDims() const { return isl_map_dim(get(), isl_dim_in); }
    isl_size nOutDims() const { return isl_map_dim(get(), isl_dim_out); }
    isl_size nParamDims() const { return isl_map_dim(get(), isl_dim_param); }

    const char *nameInDim(unsigned i) const {
        return isl_map_get_dim_name(get(), isl_dim_in, i);
    }
    const char *nameOutDim(unsigned i) const {
        return isl_map_get_dim_name(get(), isl_dim_out, i);
    }
    const char *nameParamDim(unsigned i) const {
        return isl_map_get_dim_name(get(), isl_dim_param, i);
    }

    friend std::ostream &operator<<(std::ostream &os, const PBMap &map) {
        return os << isl_map_to_str(map.map_);
    }
};

class PBVal {
    Ref<PBCtx> ctx_;
    isl_val *val_ = nullptr;

  public:
    PBVal() {}
    PBVal(const Ref<PBCtx> &ctx, isl_val *val) : ctx_(ctx), val_(val) {}
    ~PBVal() {
        if (val_ != nullptr) {
            isl_val_free(val_);
        }
    }

    PBVal(const PBVal &other) : ctx_(other.ctx_), val_(other.copy()) {}
    PBVal &operator=(const PBVal &other) {
        ctx_ = other.ctx_;
        if (val_ != nullptr) {
            isl_val_free(val_);
        }
        val_ = other.copy();
        return *this;
    }

    PBVal(PBVal &&other) : ctx_(std::move(other.ctx_)), val_(other.move()) {}
    PBVal &operator=(PBVal &&other) {
        ctx_ = std::move(other.ctx_);
        if (val_ != nullptr) {
            isl_val_free(val_);
        }
        val_ = other.move();
        return *this;
    }

    bool isValid() const { return val_ != nullptr; }

    const auto &ctx() const { return ctx_; }
    auto &ctx() { return ctx_; }

    isl_val *get() const { return GET_ISL_PTR(val_); }
    isl_val *copy() const { return COPY_ISL_PTR(val_, val); }
    isl_val *move() { return MOVE_ISL_PTR(val_); }

    bool isNaN() const { return isl_val_is_nan(get()); }
    bool isRat() const { return isl_val_is_rat(get()); }
    bool isInt() const { return isl_val_is_int(get()); }
    bool isInf() const { return isl_val_is_infty(get()); }
    bool isNegInf() const { return isl_val_is_neginfty(get()); }

    int numSi() const { return isl_val_get_num_si(get()); }
    int denSi() const { return isl_val_get_den_si(get()); }

    friend std::ostream &operator<<(std::ostream &os, const PBVal &val) {
        return os << isl_val_to_str(val.val_);
    }
};

class PBSet {
    Ref<PBCtx> ctx_;
    isl_set *set_ = nullptr;

  public:
    PBSet() {}
    PBSet(const Ref<PBCtx> &ctx, isl_set *set) : ctx_(ctx), set_(set) {}
    PBSet(const Ref<PBCtx> &ctx, const std::string &str)
        : ctx_(ctx), set_(isl_set_read_from_str(ctx->get(), str.c_str())) {
        if (set_ == nullptr) {
            ERROR("Unable to construct an PBSet from " + str);
        }
    }
    ~PBSet() {
        if (set_ != nullptr) {
            isl_set_free(set_);
        }
    }

    PBSet(const PBSet &other) : ctx_(other.ctx_), set_(other.copy()) {}
    PBSet &operator=(const PBSet &other) {
        ctx_ = other.ctx_;
        if (set_ != nullptr) {
            isl_set_free(set_);
        }
        set_ = other.copy();
        return *this;
    }

    PBSet(PBSet &&other) : ctx_(std::move(other.ctx_)), set_(other.move()) {}
    PBSet &operator=(PBSet &&other) {
        ctx_ = std::move(other.ctx_);
        if (set_ != nullptr) {
            isl_set_free(set_);
        }
        set_ = other.move();
        return *this;
    }

    bool isValid() const { return set_ != nullptr; }

    const auto &ctx() const { return ctx_; }
    auto &ctx() { return ctx_; }

    isl_set *get() const { return GET_ISL_PTR(set_); }
    isl_set *copy() const { return COPY_ISL_PTR(set_, set); }
    isl_set *move() { return MOVE_ISL_PTR(set_); }

    class Serialized {
        std::string data_;

      public:
        Serialized() {}
        Serialized(const std::string &data) : data_(data) {}
        PBSet to(const Ref<PBCtx> &ctx) const { return {ctx, data_}; }
        bool isValid() const { return !data_.empty(); }
        const auto &data() const { return data_; }
        friend std::ostream &operator<<(std::ostream &os, const Serialized &s) {
            return os << s.data_;
        }
    };
    Serialized toSerialized() const { return {isl_set_to_str(get())}; }
    PBSet to(const Ref<PBCtx> &ctx) const { return toSerialized().to(ctx); }

    bool empty() const {
        DEBUG_PROFILE("empty");
        return isl_set_is_empty(get());
    }

    bool isSingleValued() const { return isl_set_is_singleton(get()); }

    isl_size nBasic() const { return isl_set_n_basic_set(set_); }

    isl_size nDims() const { return isl_set_dim(get(), isl_dim_set); }
    isl_size nParamDims() const { return isl_set_dim(get(), isl_dim_param); }

    const char *nameDim(unsigned i) const {
        return isl_set_get_dim_name(get(), isl_dim_set, i);
    }
    const char *nameParamDim(unsigned i) const {
        return isl_set_get_dim_name(get(), isl_dim_param, i);
    }

    bool hasLowerBound(unsigned i) const {
        return isl_set_dim_has_lower_bound(get(), isl_dim_set, i);
    }
    bool hasUpperBound(unsigned i) const {
        return isl_set_dim_has_upper_bound(get(), isl_dim_set, i);
    }

    friend std::ostream &operator<<(std::ostream &os, const PBSet &set) {
        return os << isl_set_to_str(set.set_);
    }
};

class PBSpace {
    Ref<PBCtx> ctx_;
    isl_space *space_ = nullptr;

  public:
    PBSpace() {}
    PBSpace(const Ref<PBCtx> &ctx, isl_space *space)
        : ctx_(ctx), space_(space) {}
    PBSpace(const PBSet &set)
        : ctx_(set.ctx()), space_(isl_set_get_space(set.get())) {}
    PBSpace(const PBMap &map)
        : ctx_(map.ctx()), space_(isl_map_get_space(map.get())) {}
    ~PBSpace() {
        if (space_ != nullptr) {
            isl_space_free(space_);
        }
    }

    PBSpace(const PBSpace &other) : ctx_(other.ctx_), space_(other.copy()) {}
    PBSpace &operator=(const PBSpace &other) {
        ctx_ = other.ctx_;
        if (space_ != nullptr) {
            isl_space_free(space_);
        }
        space_ = other.copy();
        return *this;
    }

    PBSpace(PBSpace &&other)
        : ctx_(std::move(other.ctx_)), space_(other.move()) {}
    PBSpace &operator=(PBSpace &&other) {
        ctx_ = std::move(other.ctx_);
        if (space_ != nullptr) {
            isl_space_free(space_);
        }
        space_ = other.move();
        return *this;
    }

    bool isValid() const { return space_ != nullptr; }

    const auto &ctx() const { return ctx_; }
    auto &ctx() { return ctx_; }

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

class PBSingleFunc {
    Ref<PBCtx> ctx_;
    isl_pw_aff *func_ = nullptr;

  public:
    PBSingleFunc() {}
    PBSingleFunc(const Ref<PBCtx> &ctx, isl_pw_aff *func)
        : ctx_(ctx), func_(func) {}
    PBSingleFunc(const Ref<PBCtx> &ctx, const std::string &str)
        : ctx_(ctx), func_(isl_pw_aff_read_from_str(ctx->get(), str.c_str())) {
        if (func_ == nullptr) {
            ERROR("Unable to construct an PBSingleFunc from " + str);
        }
    }
    explicit PBSingleFunc(const Ref<PBCtx> &ctx, isl_aff *func)
        : ctx_(ctx), func_(isl_pw_aff_from_aff(func)) {}

    ~PBSingleFunc() {
        if (func_ != nullptr) {
            isl_pw_aff_free(func_);
        }
    }

    PBSingleFunc(const PBSingleFunc &other)
        : ctx_(other.ctx_), func_(other.copy()) {}
    PBSingleFunc &operator=(const PBSingleFunc &other) {
        ctx_ = other.ctx_;
        if (func_ != nullptr) {
            isl_pw_aff_free(func_);
        }
        func_ = other.copy();
        return *this;
    }

    PBSingleFunc(PBSingleFunc &&other)
        : ctx_(std::move(other.ctx_)), func_(other.move()) {}
    PBSingleFunc &operator=(PBSingleFunc &&other) {
        ctx_ = std::move(other.ctx_);
        if (func_ != nullptr) {
            isl_pw_aff_free(func_);
        }
        func_ = other.move();
        return *this;
    }

    bool isValid() const { return func_ != nullptr; }

    const auto &ctx() const { return ctx_; }
    auto &ctx() { return ctx_; }

    isl_pw_aff *get() const { return GET_ISL_PTR(func_); }
    isl_pw_aff *copy() const { return COPY_ISL_PTR(func_, pw_aff); }
    isl_pw_aff *move() { return MOVE_ISL_PTR(func_); }

    class Serialized {
        std::string data_;

      public:
        Serialized() {}
        Serialized(const std::string &data) : data_(data) {}
        PBSingleFunc to(const Ref<PBCtx> &ctx) const { return {ctx, data_}; }
        bool isValid() const { return !data_.empty(); }
        const auto &data() const { return data_; }
        friend std::ostream &operator<<(std::ostream &os, const Serialized &s) {
            return os << s.data_;
        }
    };
    Serialized toSerialized() const { return {isl_pw_aff_to_str(get())}; }
    PBSingleFunc to(const Ref<PBCtx> &ctx) const {
        return toSerialized().to(ctx);
    }

    isl_size nInDims() const { return isl_pw_aff_dim(get(), isl_dim_in); }

    std::vector<std::pair<PBSet, PBSingleFunc>> pieces() const {
        typedef std::vector<std::pair<PBSet, PBSingleFunc>> Result;
        std::pair<Result, Ref<PBCtx>> userData;
        isl_pw_aff_foreach_piece(
            get(),
            [](isl_set *set, isl_aff *piece, void *userDataRaw) {
                auto &[result, ctx] =
                    *(std::pair<Result, Ref<PBCtx>> *)userDataRaw;
                result.emplace_back(
                    PBSet(ctx, set),
                    PBSingleFunc(ctx, isl_pw_aff_from_aff(piece)));
                return isl_stat_ok;
            },
            &userData);
        return userData.first;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const PBSingleFunc &func) {
        return os << isl_pw_aff_to_str(func.func_);
    }
};

class PBFunc {
    Ref<PBCtx> ctx_;
    isl_pw_multi_aff *func_ = nullptr;

  public:
    PBFunc() {}
    PBFunc(const Ref<PBCtx> &ctx, isl_pw_multi_aff *func)
        : ctx_(ctx), func_(func) {}
    PBFunc(const Ref<PBCtx> &ctx, const std::string &str)
        : ctx_(ctx),
          func_(isl_pw_multi_aff_read_from_str(ctx->get(), str.c_str())) {
        if (func_ == nullptr) {
            ERROR("Unable to construct an PBFunc from " + str);
        }
    }

    PBFunc(const PBSingleFunc &singleFunc)
        : ctx_(singleFunc.ctx()),
          func_(isl_pw_multi_aff_from_pw_aff(singleFunc.copy())) {}
    PBFunc(PBSingleFunc &&singleFunc)
        : ctx_(std::move(singleFunc.ctx())),
          func_(isl_pw_multi_aff_from_pw_aff(singleFunc.move())) {}

    PBFunc(const PBMap &map)
        : ctx_(map.ctx()), func_(isl_pw_multi_aff_from_map(map.copy())) {}
    PBFunc(PBMap &&map)
        : ctx_(std::move(map.ctx())),
          func_(isl_pw_multi_aff_from_map(map.move())) {}

    PBFunc(const PBSet &set)
        : ctx_(set.ctx()), func_(isl_pw_multi_aff_from_set(set.copy())) {}
    PBFunc(PBSet &&set)
        : ctx_(std::move(set.ctx())),
          func_(isl_pw_multi_aff_from_set(set.move())) {}

    ~PBFunc() {
        if (func_ != nullptr) {
            isl_pw_multi_aff_free(func_);
        }
    }

    PBFunc(const PBFunc &other) : ctx_(other.ctx_), func_(other.copy()) {}
    PBFunc &operator=(const PBFunc &other) {
        ctx_ = other.ctx_;
        if (func_ != nullptr) {
            isl_pw_multi_aff_free(func_);
        }
        func_ = other.copy();
        return *this;
    }

    PBFunc(PBFunc &&other) : ctx_(std::move(other.ctx_)), func_(other.move()) {}
    PBFunc &operator=(PBFunc &&other) {
        ctx_ = std::move(other.ctx_);
        if (func_ != nullptr) {
            isl_pw_multi_aff_free(func_);
        }
        func_ = other.move();
        return *this;
    }

    bool isValid() const { return func_ != nullptr; }

    const auto &ctx() const { return ctx_; }
    auto &ctx() { return ctx_; }

    isl_pw_multi_aff *get() const { return GET_ISL_PTR(func_); }
    isl_pw_multi_aff *copy() const { return COPY_ISL_PTR(func_, pw_multi_aff); }
    isl_pw_multi_aff *move() { return MOVE_ISL_PTR(func_); }

    class Serialized {
        std::string data_;

      public:
        Serialized() {}
        Serialized(const std::string &data) : data_(data) {}
        PBFunc to(const Ref<PBCtx> &ctx) const { return {ctx, data_}; }
        bool isValid() const { return !data_.empty(); }
        const auto &data() const { return data_; }
        friend std::ostream &operator<<(std::ostream &os, const Serialized &s) {
            return os << s.data_;
        }
    };
    Serialized toSerialized() const { return {isl_pw_multi_aff_to_str(get())}; }
    PBFunc to(const Ref<PBCtx> &ctx) const { return toSerialized().to(ctx); }

    isl_size nInDims() const { return isl_pw_multi_aff_dim(get(), isl_dim_in); }
    isl_size nOutDims() const {
        return isl_pw_multi_aff_dim(get(), isl_dim_out);
    }

    PBSingleFunc operator[](isl_size i) const {
        return {ctx_, isl_pw_multi_aff_get_pw_aff(get(), 0)};
    }

    std::vector<std::pair<PBSet, PBFunc>> pieces() const {
        typedef std::vector<std::pair<PBSet, PBFunc>> Result;
        std::pair<Result, Ref<PBCtx>> userData;
        isl_pw_multi_aff_foreach_piece(
            get(),
            [](isl_set *set, isl_multi_aff *piece, void *userDataRaw) {
                auto &[result, ctx] =
                    *(std::pair<Result, Ref<PBCtx>> *)userDataRaw;
                result.emplace_back(
                    PBSet(ctx, set),
                    PBFunc(ctx, isl_pw_multi_aff_from_multi_aff(piece)));
                return isl_stat_ok;
            },
            &userData);
        return userData.first;
    }

    friend std::ostream &operator<<(std::ostream &os, const PBFunc &func) {
        return os << isl_pw_multi_aff_to_str(func.func_);
    }
};

class PBPoint {
    Ref<PBCtx> ctx_;
    isl_point *point_ = nullptr;

  public:
    PBPoint() {}
    PBPoint(const Ref<PBCtx> &ctx, isl_point *point)
        : ctx_(ctx), point_(point) {}

    ~PBPoint() {
        if (point_ != nullptr) {
            isl_point_free(point_);
        }
    }

    PBPoint(const PBPoint &other) : ctx_(other.ctx_), point_(other.copy()) {}
    PBPoint &operator=(const PBPoint &other) {
        ctx_ = other.ctx_;
        if (point_ != nullptr) {
            isl_point_free(point_);
        }
        point_ = other.copy();
        return *this;
    }

    PBPoint(PBPoint &&other)
        : ctx_(std::move(other.ctx_)), point_(other.move()) {}
    PBPoint &operator=(PBPoint &&other) {
        ctx_ = std::move(other.ctx_);
        if (point_ != nullptr) {
            isl_point_free(point_);
        }
        point_ = other.move();
        return *this;
    }

    bool isValid() const { return point_ != nullptr; }

    isl_point *get() const { return GET_ISL_PTR(point_); }
    isl_point *copy() const { return COPY_ISL_PTR(point_, point); }
    isl_point *move() { return MOVE_ISL_PTR(point_); }

    bool isVoid() const { return isl_point_is_void(point_); }

    const auto &ctx() const { return ctx_; }
    auto &ctx() { return ctx_; }

    std::vector<PBVal> coordinates() const {
        ASSERT(!isVoid());
        std::vector<PBVal> result;
        isl_size nCoord = isl_space_dim(
            PBSpace(ctx_, isl_point_get_space(point_)).get(), isl_dim_set);
        for (isl_size i = 0; i < nCoord; ++i)
            result.emplace_back(
                ctx_, isl_point_get_coordinate_val(point_, isl_dim_set, i));
        return result;
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
template <typename T>
concept PBSingleFuncRef = std::same_as<PBSingleFunc, std::decay_t<T>>;

template <typename T> auto PBRefTake(std::remove_reference_t<T> &t) {
    return t.copy();
}
template <typename T> auto PBRefTake(std::remove_reference_t<T> &&t) {
    static_assert(!std::is_lvalue_reference_v<T>); // similar to std::forward
    return t.move();
}

Ref<PBCtx> commonCtx(const auto &lhs, const auto &rhs) {
    if (lhs.ctx()->get() != rhs.ctx()->get()) {
        ERROR(
            "Operands of a Presburger operation should be on the same context");
    }
    return lhs.ctx();
}

template <PBSetRef T> PBSet projectOutAllParams(T &&set) {
    return {set.ctx(), isl_set_project_out_all_params(PBRefTake<T>(set))};
}
template <PBMapRef T> PBMap projectOutAllParams(T &&map) {
    return {map.ctx(), isl_map_project_out_all_params(PBRefTake<T>(map))};
}

template <PBSetRef T>
PBSet projectOutParamById(T &&set, const std::string &name) {
    return {set.ctx(),
            isl_set_project_out_param_id(
                PBRefTake<T>(set),
                isl_id_alloc(set.ctx()->get(), name.c_str(), nullptr))};
}
template <PBSetRef T>
PBSet projectOutParamDims(T &&set, unsigned first, unsigned n) {
    return {set.ctx(),
            isl_set_project_out(PBRefTake<T>(set), isl_dim_param, first, n)};
}
template <PBSetRef T>
PBSet projectOutDims(T &&set, unsigned first, unsigned n) {
    return {set.ctx(),
            isl_set_project_out(PBRefTake<T>(set), isl_dim_set, first, n)};
}

template <PBMapRef T>
PBMap projectOutParamById(T &&map, const std::string &name) {
    return {map.ctx(),
            isl_map_project_out_param_id(
                PBRefTake<T>(map),
                isl_id_alloc(map.ctx()->get(), name.c_str(), nullptr))};
}
template <PBMapRef T>
PBMap projectOutParamDims(T &&map, unsigned first, unsigned n) {
    return {map.ctx(),
            isl_map_project_out(PBRefTake<T>(map), isl_dim_param, first, n)};
}
template <PBMapRef T>
PBMap projectOutInputDims(T &&map, unsigned first, unsigned n) {
    return {map.ctx(),
            isl_map_project_out(PBRefTake<T>(map), isl_dim_in, first, n)};
}
template <PBMapRef T>
PBMap projectOutOutputDims(T &&map, unsigned first, unsigned n) {
    return {map.ctx(),
            isl_map_project_out(PBRefTake<T>(map), isl_dim_out, first, n)};
}

template <PBSetRef T> PBSet insertDims(T &&set, unsigned first, unsigned n) {
    return {set.ctx(),
            isl_set_insert_dims(PBRefTake<T>(set), isl_dim_set, first, n)};
}
template <PBMapRef T>
PBMap insertInputDims(T &&map, unsigned first, unsigned n) {
    return {map.ctx(),
            isl_map_insert_dims(PBRefTake<T>(map), isl_dim_in, first, n)};
}
template <PBMapRef T>
PBMap insertOutputDims(T &&map, unsigned first, unsigned n) {
    return {map.ctx(),
            isl_map_insert_dims(PBRefTake<T>(map), isl_dim_out, first, n)};
}

template <PBSetRef T> PBSet fixDim(T &&set, unsigned pos, int x) {
    return {set.ctx(), isl_set_fix_si(PBRefTake<T>(set), isl_dim_set, pos, x)};
}
template <PBMapRef T> PBMap fixInputDim(T &&map, unsigned pos, int x) {
    return {map.set(), isl_map_fix_si(PBRefTake<T>(map), isl_dim_in, pos, x)};
}
template <PBMapRef T> PBMap fixOutputDim(T &&map, unsigned pos, int x) {
    return {map.ctx(), isl_map_fix_si(PBRefTake<T>(map), isl_dim_out, pos, x)};
}

template <PBSetRef T> PBSet lowerBoundDim(T &&set, unsigned pos, int x) {
    return {set.ctx(),
            isl_set_lower_bound_si(PBRefTake<T>(set), isl_dim_set, pos, x)};
}
template <PBMapRef T> PBMap lowerBoundInputDim(T &&map, unsigned pos, int x) {
    return {map.ctx(),
            isl_map_lower_bound_si(PBRefTake<T>(map), isl_dim_in, pos, x)};
}
template <PBMapRef T> PBMap lowerBoundOutputDim(T &&map, unsigned pos, int x) {
    return {map.ctx(),
            isl_map_lower_bound_si(PBRefTake<T>(map), isl_dim_out, pos, x)};
}

template <PBSetRef T> PBSet upperBoundDim(T &&set, unsigned pos, int x) {
    return {set.ctx(),
            isl_set_upper_bound_si(PBRefTake<T>(set), isl_dim_set, pos, x)};
}
template <PBMapRef T> PBMap upperBoundInputDim(T &&map, unsigned pos, int x) {
    return {map.ctx(),
            isl_map_upper_bound_si(PBRefTake<T>(map), isl_dim_in, pos, x)};
}
template <PBMapRef T> PBMap upperBoundOutputDim(T &&map, unsigned pos, int x) {
    return {map.ctx(),
            isl_map_upper_bound_si(PBRefTake<T>(map), isl_dim_out, pos, x)};
}

template <PBSetRef T> PBMap newDomainOnlyMap(T &&set) {
    return {set.ctx(), isl_map_from_domain(PBRefTake(set))};
}
template <PBSetRef T> PBMap newRangeOnlyMap(T &&set) {
    return {set.ctx(), isl_map_from_range(PBRefTake(set))};
}

template <PBMapRef T>
PBMap moveDimsInputToOutput(T &&map, unsigned first, unsigned n,
                            unsigned target) {
    return {map.ctx(), isl_map_move_dims(PBRefTake<T>(map), isl_dim_out, target,
                                         isl_dim_in, first, n)};
}
template <PBMapRef T>
PBMap moveDimsOutputToInput(T &&map, unsigned first, unsigned n,
                            unsigned target) {
    return {map.ctx(), isl_map_move_dims(PBRefTake<T>(map), isl_dim_in, target,
                                         isl_dim_out, first, n)};
}
/**
 * Move other dimensions to be parameters.
 *
 * NOTE: This function can only be applied on named dimensions, which typically
 * mean dimensions previously converted from parameters. For unnamed dimensions,
 * currently you need to apply a moving mapping by yourself.
 *
 * @{
 */
template <PBMapRef T>
PBMap moveDimsInputToParam(T &&map, unsigned first, unsigned n,
                           unsigned target) {
    return {map.ctx(), isl_map_move_dims(PBRefTake<T>(map), isl_dim_param,
                                         target, isl_dim_in, first, n)};
}
template <PBMapRef T>
PBMap moveDimsOutputToParam(T &&map, unsigned first, unsigned n,
                            unsigned target) {
    return {map.ctx(), isl_map_move_dims(PBRefTake<T>(map), isl_dim_param,
                                         target, isl_dim_out, first, n)};
}
/** @} */
template <PBMapRef T>
PBMap moveDimsParamToInput(T &&map, unsigned first, unsigned n,
                           unsigned target) {
    return {map.ctx(), isl_map_move_dims(PBRefTake<T>(map), isl_dim_in, target,
                                         isl_dim_param, first, n)};
}
template <PBMapRef T>
PBMap moveDimsParamToOutput(T &&map, unsigned first, unsigned n,
                            unsigned target) {
    return {map.ctx(), isl_map_move_dims(PBRefTake<T>(map), isl_dim_out, target,
                                         isl_dim_param, first, n)};
}

template <PBSetRef T>
PBSet moveDimsSetToParam(T &&set, unsigned first, unsigned n, unsigned target) {
    return {set.ctx(), isl_set_move_dims(PBRefTake<T>(set), isl_dim_param,
                                         target, isl_dim_set, first, n)};
}
template <PBSetRef T>
PBSet moveDimsParamToSet(T &&set, unsigned first, unsigned n, unsigned target) {
    return {set.ctx(), isl_set_move_dims(PBRefTake<T>(set), isl_dim_set, target,
                                         isl_dim_param, first, n)};
}

template <PBSetRef T, PBSetRef U>
std::pair<PBSet, PBSet> padToSameDims(T &&lhs, U &&rhs) {
    auto n = std::max(lhs.nDims(), rhs.nDims());
    return std::make_pair(
        insertDims(std::forward<T>(lhs), lhs.nDims(), n - lhs.nDims()),
        insertDims(std::forward<T>(rhs), rhs.nDims(), n - rhs.nDims()));
}

template <PBSetRef T> PBSet complement(T &&set) {
    DEBUG_PROFILE("complement");
    return {set.ctx(), isl_set_complement(PBRefTake<T>(set))};
}
template <PBMapRef T> PBMap complement(T &&map) {
    DEBUG_PROFILE("complement");
    return {map.ctx(), isl_map_complement(PBRefTake<T>(map))};
}

template <PBMapRef T> PBMap reverse(T &&map) {
    DEBUG_PROFILE("reverse");
    return {map.ctx(), isl_map_reverse(PBRefTake<T>(map))};
}

template <PBMapRef T, PBMapRef U> PBMap subtract(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return {commonCtx(lhs, rhs),
            isl_map_subtract(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}
template <PBSetRef T, PBSetRef U> PBSet subtract(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("subtract", "nBasic=" + std::to_string(lhs.nBasic()) +
                                          "," + std::to_string(rhs.nBasic()));
    return {commonCtx(lhs, rhs),
            isl_set_subtract(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBMapRef T, PBMapRef U> PBMap intersect(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return {commonCtx(lhs, rhs),
            isl_map_intersect(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}
template <PBSetRef T, PBSetRef U> PBSet intersect(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersect",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return {commonCtx(lhs, rhs),
            isl_set_intersect(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBMapRef T, PBSetRef U> PBMap intersectDomain(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersectDomain",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return {commonCtx(lhs, rhs),
            isl_map_intersect_domain(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}
template <PBMapRef T, PBSetRef U> PBMap intersectRange(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("intersectRange",
                          "nBasic=" + std::to_string(lhs.nBasic()) + "," +
                              std::to_string(rhs.nBasic()));
    return {commonCtx(lhs, rhs),
            isl_map_intersect_range(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBSingleFuncRef T, PBSetRef U>
PBSingleFunc intersectDomain(T &&lhs, U &&rhs) {
    return {commonCtx(lhs, rhs),
            isl_pw_aff_intersect_domain(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}
template <PBFuncRef T, PBSetRef U> PBFunc intersectDomain(T &&lhs, U &&rhs) {
    return {commonCtx(lhs, rhs), isl_multi_pw_aff_intersect_domain(
                                     PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBSetRef T, PBSetRef U> PBSet intersectParams(T &&lhs, U &&rhs) {
    return {commonCtx(lhs, rhs),
            isl_set_intersect_params(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}
template <PBMapRef T, PBSetRef U> PBMap intersectParams(T &&lhs, U &&rhs) {
    return {commonCtx(lhs, rhs),
            isl_map_intersect_params(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBMapRef T, PBMapRef U> PBMap uni(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("uni", "nBasic=" + std::to_string(lhs.nBasic()) +
                                     "," + std::to_string(rhs.nBasic()));
    return {commonCtx(lhs, rhs),
            isl_map_union(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBSetRef T, PBSetRef U> PBSet uni(T &&lhs, U &&rhs) {
    DEBUG_PROFILE_VERBOSE("uni", "nBasic=" + std::to_string(lhs.nBasic()) +
                                     "," + std::to_string(rhs.nBasic()));
    return {commonCtx(lhs, rhs),
            isl_set_union(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBSetRef T, PBMapRef U> PBSet apply(T &&lhs, U &&rhs) {
    DEBUG_PROFILE("apply");
    return {commonCtx(lhs, rhs),
            isl_set_apply(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBMapRef T, PBMapRef U> PBMap applyDomain(T &&lhs, U &&rhs) {
    DEBUG_PROFILE("applyDomain");
    return {commonCtx(lhs, rhs),
            isl_map_apply_domain(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBMapRef T, PBMapRef U> PBMap applyRange(T &&lhs, U &&rhs) {
    DEBUG_PROFILE("applyRange");
    return {commonCtx(lhs, rhs),
            isl_map_apply_range(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBMapRef T, PBMapRef U> PBMap sum(T &&lhs, U &&rhs) {
    DEBUG_PROFILE("sum");
    return {commonCtx(lhs, rhs),
            isl_map_sum(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}
template <PBSetRef T, PBSetRef U> PBSet sum(T &&lhs, U &&rhs) {
    DEBUG_PROFILE("sum");
    return {commonCtx(lhs, rhs),
            isl_set_sum(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBMapRef T> PBMap neg(T &&map) {
    DEBUG_PROFILE("neg");
    return {map.ctx(), isl_map_neg(PBRefTake<T>(map))};
}
template <PBSetRef T> PBSet neg(T &&set) {
    DEBUG_PROFILE("neg");
    return {set.ctx(), isl_set_neg(PBRefTake<T>(set))};
}

template <PBMapRef T> PBMap lexmax(T &&map) {
    DEBUG_PROFILE_VERBOSE("lexmax", "nBasic=" + std::to_string(map.nBasic()));
    return {map.ctx(), isl_map_lexmax(PBRefTake<T>(map))};
}

template <PBMapRef T> PBMap lexmin(T &&map) {
    DEBUG_PROFILE_VERBOSE("lexmin", "nBasic=" + std::to_string(map.nBasic()));
    return {map.ctx(), isl_map_lexmin(PBRefTake<T>(map))};
}

template <PBSetRef T> PBSet lexmax(T &&set) {
    DEBUG_PROFILE_VERBOSE("lexmax", "nBasic=" + std::to_string(set.nBasic()));
    return {set.ctx(), isl_set_lexmax(PBRefTake<T>(set))};
}

template <PBSetRef T> PBSet lexmin(T &&set) {
    DEBUG_PROFILE_VERBOSE("lexmin", "nBasic=" + std::to_string(set.nBasic()));
    return {set.ctx(), isl_set_lexmin(PBRefTake<T>(set))};
}

template <PBSpaceRef T> PBMap identity(T &&space) {
    DEBUG_PROFILE("identity");
    return {space.ctx(), isl_map_identity(PBRefTake<T>(space))};
}

template <PBSpaceRef T> PBMap lexGE(T &&space) {
    DEBUG_PROFILE("lexGE");
    return {space.ctx(), isl_map_lex_ge(PBRefTake<T>(space))};
}

template <PBSpaceRef T> PBMap lexGT(T &&space) {
    DEBUG_PROFILE("lexGT");
    return {space.ctx(), isl_map_lex_gt(PBRefTake<T>(space))};
}

template <PBSpaceRef T> PBMap lexLE(T &&space) {
    DEBUG_PROFILE("lexLE");
    return {space.ctx(), isl_map_lex_le(PBRefTake<T>(space))};
}

template <PBSpaceRef T> PBMap lexLT(T &&space) {
    DEBUG_PROFILE("lexLT");
    return {space.ctx(), isl_map_lex_lt(PBRefTake<T>(space))};
}

inline PBSpace spaceAlloc(const Ref<PBCtx> &ctx, unsigned nparam, unsigned nIn,
                          unsigned nOut) {
    return {ctx, isl_space_alloc(ctx->get(), nparam, nIn, nOut)};
}

inline PBSpace spaceSetAlloc(const Ref<PBCtx> &ctx, unsigned nparam,
                             unsigned dim) {
    return {ctx, isl_space_set_alloc(ctx->get(), nparam, dim)};
}

template <PBSpaceRef T> PBSet emptySet(T &&space) {
    return {space.ctx(), isl_set_empty(PBRefTake<T>(space))};
}

template <PBSpaceRef T> PBMap emptyMap(T &&space) {
    return {space.ctx(), isl_map_empty(PBRefTake<T>(space))};
}

template <PBSpaceRef T> PBSet universeSet(T &&space) {
    return {space.ctx(), isl_set_universe(PBRefTake<T>(space))};
}

template <PBSpaceRef T> PBMap universeMap(T &&space) {
    return {space.ctx(), isl_map_universe(PBRefTake<T>(space))};
}

template <PBMapRef T> PBSet domain(T &&map) {
    return {map.ctx(), isl_map_domain(PBRefTake<T>(map))};
}

template <PBMapRef T> PBSet range(T &&map) {
    return {map.ctx(), isl_map_range(PBRefTake<T>(map))};
}

template <PBSingleFuncRef T> PBSet domain(T &&func) {
    return {func.ctx(), isl_pw_aff_domain(PBRefTake<T>(func))};
}
template <PBFuncRef T> PBSet domain(T &&func) {
    return {func.ctx(), isl_multi_pw_aff_domain(PBRefTake<T>(func))};
}

template <PBSetRef T> PBSet params(T &&set) {
    return {set.ctx(), isl_set_params(PBRefTake<T>(set))};
}
template <PBMapRef T> PBSet params(T &&map) {
    return {map.ctx(), isl_map_params(PBRefTake<T>(map))};
}

template <PBSetRef T> PBSet coalesce(T &&set) {
    DEBUG_PROFILE("coalesce");
    return {set.ctx(), isl_set_coalesce(PBRefTake<T>(set))};
}

template <PBMapRef T> PBMap coalesce(T &&map) {
    DEBUG_PROFILE("coalesce");
    return {map.ctx(), isl_map_coalesce(PBRefTake<T>(map))};
}

template <PBSetRef T, PBSetRef U> PBSet cartesianProduct(T &&lhs, U &&rhs) {
    return {commonCtx(lhs, rhs),
            isl_set_flat_product(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBSetRef T> PBVal dimMaxVal(T &&set, int pos) {
    return {set.ctx(), isl_set_dim_max_val(PBRefTake<T>(set), pos)};
}

template <PBSetRef T> PBVal dimMinVal(T &&set, int pos) {
    return {set.ctx(), isl_set_dim_min_val(PBRefTake<T>(set), pos)};
}

inline PBVal dimFixVal(const PBSet &set, int pos) {
    return {set.ctx(),
            isl_set_plain_get_val_if_fixed(set.get(), isl_dim_set, pos)};
}

template <PBSpaceRef T> PBSpace spaceMapFromSet(T &&space) {
    return {space.ctx(), isl_space_map_from_set(PBRefTake<T>(space))};
}

template <PBMapRef T> PBSet wrap(T &&map) {
    return {map.ctx(), isl_map_wrap(PBRefTake<T>(map))};
}

template <PBSetRef T> PBMap unwrap(T &&set) {
    return {set.ctx(), isl_set_unwrap(PBRefTake<T>(set))};
}

template <PBSetRef T> PBSet flatten(T &&set) {
    return {set.ctx(), isl_set_flatten(PBRefTake<T>(set))};
}
template <PBMapRef T> PBMap flattenDomain(T &&map) {
    return {map.ctx(), isl_map_flatten_domain(PBRefTake<T>(map))};
}
template <PBMapRef T> PBMap flattenRange(T &&map) {
    return {map.ctx(), isl_map_flatten_range(PBRefTake<T>(map))};
}

template <PBMapRef T> PBSet flattenMapToSet(T &&map) {
    return flatten(wrap(std::forward<T>(map)));
}

template <PBSetRef T> PBPoint sample(T &&set) {
    return {set.ctx(), isl_set_sample_point(PBRefTake<T>(set))};
}

template <PBSingleFuncRef T, PBSingleFuncRef U>
PBSingleFunc min(T &&lhs, U &&rhs) {
    return {commonCtx(lhs, rhs),
            isl_pw_aff_min(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

template <PBSingleFuncRef T, PBSingleFuncRef U>
PBSingleFunc max(T &&lhs, U &&rhs) {
    return {commonCtx(lhs, rhs),
            isl_pw_aff_max(PBRefTake<T>(lhs), PBRefTake<U>(rhs))};
}

/**
 * @brief Compute the set of coefficients corresponded to the given set
 *
 * The output coefficient set will be of the same dimension as the given set
 *
 * Let the variables of the input set $X$ be $x_i$, and that of the output
 * coefficient set $C$ be $c_i$, then this procedure guarantees:
 * $\{c_i\} \in C \Leftrightarrow \forall \{x_i\} \in X, \Sigma c_i x_i \geq c$
 *
 * @param set the input set
 * @param c the RHS constant
 * @return PBSet set of valid coefficients for the input set
 */
template <PBSetRef T> PBSet coefficients(T &&set, int64_t c = 0) {
    // ISL coefficients applies Farkas lemma which disallows intermediate
    // variables introduced by division/modulo constraints. The best we can do
    // is to remove such constraints and provide a slightly relaxed bound for
    // the coefficients, so remove_divs first.
    auto coefficientsMap = isl_map_from_basic_map(isl_basic_set_unwrap(
        isl_set_coefficients(isl_set_remove_divs(PBRefTake<T>(set)))));
    auto ctx = isl_map_get_ctx(coefficientsMap);
    auto paramsSpace = isl_space_domain(isl_map_get_space(coefficientsMap));
    auto nParams = isl_space_dim(paramsSpace, isl_dim_set);
    auto cPoint = isl_point_zero(paramsSpace);
    isl_point_set_coordinate_val(cPoint, isl_dim_set, nParams - 1,
                                 isl_val_int_from_si(ctx, -c));
    return apply(PBSet(set.ctx(), isl_set_from_point(cPoint)),
                 PBMap(set.ctx(), coefficientsMap));
}

inline bool isSubset(const PBSet &small, const PBSet &big) {
    return isl_set_is_subset(small.get(), big.get());
}
inline bool isSubset(const PBMap &small, const PBMap &big) {
    return isl_map_is_subset(small.get(), big.get());
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

inline bool operator==(const PBSingleFunc &lhs, const PBSingleFunc &rhs) {
    return isl_pw_aff_is_equal(lhs.get(), rhs.get());
}

inline bool operator==(const PBFunc &lhs, const PBFunc &rhs) {
    return isl_pw_multi_aff_is_equal(lhs.get(), rhs.get());
}

class PBBuildExpr {
    std::string expr_;
    explicit PBBuildExpr(const std::string &expr) : expr_(expr) {}
    friend class PBBuilder;

  public:
    PBBuildExpr() = default;
    PBBuildExpr(const PBBuildExpr &) = default;
    PBBuildExpr(PBBuildExpr &&) = default;
    PBBuildExpr &operator=(const PBBuildExpr &) = default;
    PBBuildExpr &operator=(PBBuildExpr &&) = default;

    PBBuildExpr(bool b) : expr_(b ? "true" : "false") {}
    PBBuildExpr(std::integral auto i) : expr_(toString(i)) {}

    friend PBBuildExpr operator+(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " + " + b.expr_ + ")");
    }
    PBBuildExpr &operator+=(const PBBuildExpr &other) {
        return *this = *this + other;
    }

    PBBuildExpr operator-() const { return PBBuildExpr("(-" + expr_ + ")"); }
    friend PBBuildExpr operator-(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " - " + b.expr_ + ")");
    }
    PBBuildExpr &operator-=(const PBBuildExpr &other) {
        return *this = *this - other;
    }

    friend PBBuildExpr operator*(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " * " + b.expr_ + ")");
    }
    PBBuildExpr &operator*=(const PBBuildExpr &other) {
        return *this = *this * other;
    }

    friend PBBuildExpr operator/(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " / " + b.expr_ + ")");
    }
    PBBuildExpr &operator/=(const PBBuildExpr &other) {
        return *this = *this / other;
    }

    friend PBBuildExpr ceilDiv(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("ceil(" + a.expr_ + " / " + b.expr_ + ")");
    }
    friend PBBuildExpr floorDiv(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("floor(" + a.expr_ + " / " + b.expr_ + ")");
    }

    friend PBBuildExpr operator%(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " % " + b.expr_ + ")");
    }
    PBBuildExpr &operator%=(const PBBuildExpr &other) {
        return *this = *this % other;
    }

    friend PBBuildExpr operator<(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " < " + b.expr_ + ")");
    }

    friend PBBuildExpr operator<=(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " <= " + b.expr_ + ")");
    }

    friend PBBuildExpr operator>(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " > " + b.expr_ + ")");
    }

    friend PBBuildExpr operator>=(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " >= " + b.expr_ + ")");
    }

    friend PBBuildExpr operator==(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " = " + b.expr_ + ")");
    }

    friend PBBuildExpr operator!=(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " != " + b.expr_ + ")");
    }

    friend PBBuildExpr operator&&(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " and " + b.expr_ + ")");
    }

    friend PBBuildExpr operator||(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("(" + a.expr_ + " or " + b.expr_ + ")");
    }

    friend PBBuildExpr max(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("max(" + a.expr_ + ", " + b.expr_ + ")");
    }

    friend PBBuildExpr min(const PBBuildExpr &a, const PBBuildExpr &b) {
        return PBBuildExpr("min(" + a.expr_ + ", " + b.expr_ + ")");
    }

    friend std::ostream &operator<<(std::ostream &os, const PBBuildExpr &e);
};

class PBBuilder {
    int anonVarNum_ = 0;
    std::unordered_set<std::string> namedVars;

    std::vector<PBBuildExpr> constraints_;

  protected:
    PBBuildExpr newVar(const std::string &name = "");
    std::vector<PBBuildExpr> newVars(int n, const std::string &prefix = "");

    std::string getConstraintsStr() const;

  public:
    PBBuilder() = default;
    PBBuilder(const PBBuilder &) = default;
    PBBuilder(PBBuilder &&) = default;
    PBBuilder &operator=(const PBBuilder &) = default;
    PBBuilder &operator=(PBBuilder &&) = default;

    void addConstraint(const PBBuildExpr &constraint);
    void addConstraint(PBBuildExpr &&constraint);
    template <typename T = std::initializer_list<PBBuildExpr>>
    void addConstraints(T &&constraints) {
        for (auto &&c : constraints)
            addConstraint(c);
    }
    const std::vector<PBBuildExpr> &constraints() const { return constraints_; }
    void clearConstraints() { constraints_.clear(); };
};

class PBMapBuilder : public PBBuilder {
    std::vector<PBBuildExpr> inputs_;
    std::vector<PBBuildExpr> outputs_;

  public:
    PBMapBuilder() = default;
    PBMapBuilder(const PBMapBuilder &) = default;
    PBMapBuilder(PBMapBuilder &&) = default;
    PBMapBuilder &operator=(const PBMapBuilder &) = default;
    PBMapBuilder &operator=(PBMapBuilder &&) = default;

    void addInput(const PBBuildExpr &expr);
    void addInputs(auto &&exprs) {
        for (auto &&e : exprs)
            addInput(e);
    }
    PBBuildExpr newInput(const std::string &name);
    std::vector<PBBuildExpr> newInputs(int n, const std::string &prefix = "");
    const std::vector<PBBuildExpr> &inputs() const { return inputs_; }
    void clearInputs() { inputs_.clear(); }

    void addOutput(const PBBuildExpr &expr);
    void addOutputs(auto &&exprs) {
        for (auto &&e : exprs)
            addOutput(e);
    }
    PBBuildExpr newOutput(const std::string &name);
    std::vector<PBBuildExpr> newOutputs(int n, const std::string &prefix = "");
    const std::vector<PBBuildExpr> &outputs() const { return outputs_; }
    void clearOutputs() { outputs_.clear(); }

    PBMap build(const Ref<PBCtx> &ctx) const;
};

class PBSetBuilder : public PBBuilder {
    std::vector<PBBuildExpr> vars_;

  public:
    PBSetBuilder() = default;
    PBSetBuilder(const PBSetBuilder &) = default;
    PBSetBuilder(PBSetBuilder &&) = default;
    PBSetBuilder &operator=(const PBSetBuilder &) = default;
    PBSetBuilder &operator=(PBSetBuilder &&) = default;

    void addVar(const PBBuildExpr &expr);
    void addVars(auto &&exprs) {
        for (auto &&e : exprs)
            addVar(e);
    }
    PBBuildExpr newVar(const std::string &name);
    std::vector<PBBuildExpr> newVars(int n, const std::string &prefix = "");
    const std::vector<PBBuildExpr> &vars() const { return vars_; }
    void clearVars() { vars_.clear(); }

    PBSet build(const Ref<PBCtx> &ctx) const;
};

auto pbFuncWithTimeout(const auto &func, int seconds, const auto &...args)
    -> std::optional<decltype(func(args...))> {
    decltype(func(args...)) ret;
    if (timeout([&]() { ret = func(args...); }, seconds)) {
        return ret;
    } else {
        return std::nullopt;
    }
}

} // namespace freetensor

#endif // FREE_TENSOR_PRESBURGER_H
