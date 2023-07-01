#ifndef FREE_TENSOR_DISJOINT_SET_H
#define FREE_TENSOR_DISJOINT_SET_H

#include "stmt.h"
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace freetensor {

template <typename T> class DisjointSet {
    struct Node {
        int parent, rank;
        T key;
    };

    std::unordered_map<T, int> idMap_;
    std::vector<Node> tree_;

    int find(int id) {
        int root = id;
        // lookup the root
        while (tree_[root].parent != root)
            root = tree_[root].parent;
        // path compression
        while (tree_[id].parent != root) {
            auto &p = tree_[id].parent;
            id = p;
            p = root;
        }
        return root;
    }

    void uni(int id0, int id1) {
        int root0 = find(id0);
        int root1 = find(id1);
        if (root0 != root1) {
            if (tree_[root0].rank < tree_[root1].rank)
                tree_[root0].parent = root1;
            else {
                tree_[root1].parent = root0;
                if (tree_[root0].rank == tree_[root1].rank)
                    tree_[root1].rank++;
            }
        }
    }

    void add(const T &key) {
        if (idMap_.contains(key))
            return;
        auto prev_size = idMap_.size();
        idMap_.emplace(key, prev_size);
        tree_.emplace_back(prev_size, 0, key);
    }

  public:
    const T &find(const T &key) {
        add(key);
        return tree_[find(idMap_[key])].key;
    }

    void uni(const T &key0, const T &key1) {
        add(key0);
        add(key1);
        uni(idMap_[key0], idMap_[key1]);
    }
};

} // namespace freetensor

#endif
