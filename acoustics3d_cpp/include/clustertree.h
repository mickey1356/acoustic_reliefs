#pragma once

#include <utility>
#include <vector>

#include "constants.h"

namespace bem3d {

class ClusterTreeNode {
public:
    ClusterTreeNode(vec3 center, double radius, int start, int n_elems);
    ~ClusterTreeNode();

    void print(int depth);
    bool check(const mat3 &Cs);

    pii indices;
    bool is_leaf = true;
    ClusterTreeNode *left = nullptr, *right = nullptr;
    vec3 center;
    double radius;
};

class ClusterTree {
public:
    ClusterTree(int max_pts, const mat3 &Cs);
    ~ClusterTree();

    ClusterTreeNode *build_tree(int start, const Eigen::ArrayXi &indices, const mat3 &Cs);
    void get_direct_and_approx_blocks(double factor, std::vector<std::pair<pii, pii>> &direct, std::vector<std::pair<pii, pii>> &approx) const;

    void print() {
        root->print(0);
    }

    bool check(const mat3 &Cs) {
        return root->check(Cs);
    }

    Eigen::ArrayXi reordering;

private:
    void dual_tree_traversal(ClusterTreeNode *src, ClusterTreeNode *tgt,
                             std::vector<std::pair<pii, pii>> &direct, std::vector<std::pair<pii, pii>> &approx, double factor) const;

    ClusterTreeNode *root;
    int max_pts;
};

} // namespace bem3d