#include <algorithm>

#include <math/presburger.h>

namespace ir {

PBMap::PBMap(const PBCtx &ctx, const std::string &str) {
    node_ = Ref<PBMapNode>::make();
    node_->map_ = ISLMap(ctx.get(), str);
}

PBMap::PBMap(const Ref<PBMapOp> &op) {
    node_ = Ref<PBMapNode>::make();
    node_->eval_ = op;
}

PBOpType PBMap::opType() const {
    return node_->map_.isValid() ? PBOpType::None : node_->eval_->type();
}

Ref<PBMapOp> PBMap::op() const {
    ASSERT(node_->eval_.isValid());
    return node_->eval_;
}

void PBMap::exec() const {
    if (!node_->map_.isValid()) {
        node_->map_ = node_->eval_->exec();
    }
}

const ISLMap &PBMap::copy() const {
    ASSERT(isValid());
    exec();
    return node_->map_;
}

ISLMap &&PBMap::move() const {
    ASSERT(useCount() == 1);
    exec();
    return std::move(node_->map_);
}

PBSet::PBSet(const PBCtx &ctx, const std::string &str) {
    node_ = Ref<PBSetNode>::make();
    node_->set_ = ISLSet(ctx.get(), str);
}

PBSet::PBSet(const Ref<PBSetOp> &op) {
    node_ = Ref<PBSetNode>::make();
    node_->eval_ = op;
}

PBOpType PBSet::opType() const {
    return node_->set_.isValid() ? PBOpType::None : node_->eval_->type();
}

Ref<PBSetOp> PBSet::op() const {
    ASSERT(node_->eval_.isValid());
    return node_->eval_;
}

void PBSet::exec() const {
    if (!node_->set_.isValid()) {
        node_->set_ = node_->eval_->exec();
    }
}

const ISLSet &PBSet::copy() const {
    ASSERT(isValid());
    exec();
    return node_->set_;
}

ISLSet &&PBSet::move() const {
    ASSERT(useCount() == 1);
    exec();
    return std::move(node_->set_);
}

const ISLSpace &PBSpace::copy() const {
    ASSERT(isValid());
    return node_->space_;
}

ISLSpace &&PBSpace::move() const {
    ASSERT(useCount() == 1);
    return std::move(node_->space_);
}

ISLMap PBMapComplement::exec() {
    if (map_.useCount() == 1) {
        return complement(map_.move());
    } else {
        return complement(map_.copy());
    }
}

ISLMap PBMapReverse::exec() {
    if (map_.useCount() == 1) {
        return reverse(map_.move());
    } else {
        return reverse(map_.copy());
    }
}

ISLMap PBMapLexMin::exec() {
    if (map_.useCount() == 1) {
        return lexmin(map_.move());
    } else {
        return lexmin(map_.copy());
    }
}

ISLMap PBMapLexMax::exec() {
    if (map_.useCount() == 1) {
        return lexmax(map_.move());
    } else {
        return lexmax(map_.copy());
    }
}

ISLMap PBMapSubtract::exec() {
    if (lhs_.useCount() == 1 && rhs_.useCount() == 1) {
        return subtract(lhs_.move(), rhs_.move());
    } else if (lhs_.useCount() == 1) {
        return subtract(lhs_.move(), rhs_.copy());
    } else if (rhs_.useCount() == 1) {
        return subtract(lhs_.copy(), rhs_.move());
    } else {
        return subtract(lhs_.copy(), rhs_.copy());
    }
}

ISLMap PBMapIntersect::exec() {
    std::vector<ISLMap> args;
    std::function<void(const PBMap &x)> f = [&](const PBMap &x) {
        if (x.opType() == PBOpType::Intersect && x.useCount() == 1) {
            f(x.op().as<PBMapIntersect>()->lhs_);
            f(x.op().as<PBMapIntersect>()->rhs_);
        } else {
            if (x.useCount() == 1) {
                args.emplace_back(x.move());
            } else {
                args.emplace_back(x.copy());
            }
        }
    };
    f(lhs_);
    f(rhs_);

    auto cmp = [](const ISLMap &lhs, const ISLMap &rhs) {
        return lhs.nBasic() > rhs.nBasic();
    };
    std::make_heap(args.begin(), args.end(), cmp);
    while (args.size() > 1) {
        std::pop_heap(args.begin(), args.end(), cmp);
        auto lhs = std::move(args.back());
        args.pop_back();
        std::pop_heap(args.begin(), args.end(), cmp);
        auto rhs = std::move(args.back());
        args.pop_back();
        auto res = intersect(std::move(lhs), std::move(rhs));
        args.emplace_back(std::move(res));
        std::push_heap(args.begin(), args.end(), cmp);
    }
    auto ret = std::move(args.front());
    return ret;
}

ISLMap PBMapUnion::exec() {
    if (lhs_.useCount() == 1 && rhs_.useCount() == 1) {
        return uni(lhs_.move(), rhs_.move());
    } else if (lhs_.useCount() == 1) {
        return uni(lhs_.move(), rhs_.copy());
    } else if (rhs_.useCount() == 1) {
        return uni(lhs_.copy(), rhs_.move());
    } else {
        return uni(lhs_.copy(), rhs_.copy());
    }
}

ISLMap PBMapApplyDomain::exec() {
    if (lhs_.useCount() == 1 && rhs_.useCount() == 1) {
        return applyDomain(lhs_.move(), rhs_.move());
    } else if (lhs_.useCount() == 1) {
        return applyDomain(lhs_.move(), rhs_.copy());
    } else if (rhs_.useCount() == 1) {
        return applyDomain(lhs_.copy(), rhs_.move());
    } else {
        return applyDomain(lhs_.copy(), rhs_.copy());
    }
}

ISLMap PBMapApplyRange::exec() {
    if (lhs_.useCount() == 1 && rhs_.useCount() == 1) {
        return applyRange(lhs_.move(), rhs_.move());
    } else if (lhs_.useCount() == 1) {
        return applyRange(lhs_.move(), rhs_.copy());
    } else if (rhs_.useCount() == 1) {
        return applyRange(lhs_.copy(), rhs_.move());
    } else {
        return applyRange(lhs_.copy(), rhs_.copy());
    }
}

ISLMap PBMapIdentity::exec() {
    if (space_.useCount() == 1) {
        return identity(space_.move());
    } else {
        return identity(space_.copy());
    }
}

ISLMap PBMapLexGE::exec() {
    if (space_.useCount() == 1) {
        return lexGE(space_.move());
    } else {
        return lexGE(space_.copy());
    }
}

ISLMap PBMapLexGT::exec() {
    if (space_.useCount() == 1) {
        return lexGT(space_.move());
    } else {
        return lexGT(space_.copy());
    }
}

ISLMap PBMapEmpty::exec() {
    if (space_.useCount() == 1) {
        return emptyMap(space_.move());
    } else {
        return emptyMap(space_.copy());
    }
}

ISLMap PBMapUniverse::exec() {
    if (space_.useCount() == 1) {
        return universeMap(space_.move());
    } else {
        return universeMap(space_.copy());
    }
}

ISLSet PBSetComplement::exec() {
    if (set_.useCount() == 1) {
        return complement(set_.move());
    } else {
        return complement(set_.copy());
    }
}

ISLSet PBSetEmpty::exec() {
    if (space_.useCount() == 1) {
        return emptySet(space_.move());
    } else {
        return emptySet(space_.copy());
    }
}

ISLSet PBSetUniverse::exec() {
    if (space_.useCount() == 1) {
        return universeSet(space_.move());
    } else {
        return universeSet(space_.copy());
    }
}

ISLSet PBSetDomain::exec() {
    if (map_.useCount() == 1) {
        return domain(map_.move());
    } else {
        return domain(map_.copy());
    }
}

ISLSet PBSetRange::exec() {
    if (map_.useCount() == 1) {
        return range(map_.move());
    } else {
        return range(map_.copy());
    }
}

} // namespace ir

