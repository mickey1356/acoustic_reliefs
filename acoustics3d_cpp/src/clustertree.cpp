#include "clustertree.h"

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace bem3d {

ClusterTreeNode::ClusterTreeNode(vec3 center, double radius, int start, int n_elems) : center(center), radius(radius) {
    indices = std::make_pair(start, start + n_elems);
}

ClusterTreeNode::~ClusterTreeNode() {
    if (left) {
        delete left;
    }
    if (right) {
        delete right;
    }
}

void ClusterTreeNode::print(int depth) {
    if (left) {
        left->print(depth + 1);
    }
    std::cout << std::string(depth, '\t') + " depth: " << depth << " (" << indices.first << ", " << indices.second << ")" << std::endl;
    if (right) {
        right->print(depth + 1);
    }
}

bool ClusterTreeNode::check(const mat3 &Cs) {
    if (is_leaf) {
        bool ok = true;
        for (int i = indices.first; i < indices.second; i++) {
            ok = ok && ((Cs.row(i) - center.transpose()).norm() <= radius + EPSILON);
        }
        return ok;
    } else {
        bool ok = true;
        for (int i = indices.first; i < indices.second; i++) {
            ok = ok && ((Cs.row(i) - center.transpose()).norm() <= radius + EPSILON);
        }
        return ok && left->check(Cs) && right->check(Cs);
    }
}

ClusterTree::ClusterTree(int max_pts, const mat3 &Cs) : max_pts(max_pts) {
    reordering.setZero(Cs.rows());
    Eigen::ArrayXi indices(Cs.rows());
    std::iota(indices.begin(), indices.end(), 0);
    root = build_tree(0, indices, Cs);
}

ClusterTree::~ClusterTree() {
    delete root;
}

ClusterTreeNode *ClusterTree::build_tree(int start, const Eigen::ArrayXi &indices, const mat3 &Cs) {
    if (indices.size() == 0) {
        return nullptr;
    }

    auto pts_to_split = Cs(indices, all);
    Eigen::RowVector3d center = pts_to_split.colwise().mean();
    double radius = (pts_to_split.rowwise() - center).rowwise().norm().maxCoeff();
    ClusterTreeNode *node = new ClusterTreeNode(center, radius, start, static_cast<int>(indices.size()));

    if (indices.size() <= max_pts) {
        // number of elements in this node is smaller than the maximum allowed number, so we are done
        // update the reordering (only leaf nodes)
        reordering(Eigen::seqN(start, indices.size())) = indices;
        return node;
    }
    // node will have children
    node->is_leaf = false;

    // split the elements in pts_to_split
    // first find the largest dimension to split
    auto bb_lengths = pts_to_split.colwise().maxCoeff() - pts_to_split.colwise().minCoeff();
    int dim_to_split;
    bb_lengths.maxCoeff(&dim_to_split);
    auto gamma = center[dim_to_split];

    std::vector<int> left_vec, right_vec;
    for (auto i : indices) {
        if (Cs(i, dim_to_split) < gamma) {
            left_vec.push_back(i);
        } else {
            right_vec.push_back(i);
        }
    }
    auto left_indices = Eigen::ArrayXi::Map(left_vec.data(), left_vec.size());
    auto right_indices = Eigen::ArrayXi::Map(right_vec.data(), right_vec.size());

    node->left = build_tree(start, left_indices, Cs);
    node->right = build_tree(start + static_cast<int>(left_indices.size()), right_indices, Cs);

    return node;
}

void ClusterTree::get_direct_and_approx_blocks(double factor,
                                               std::vector<std::pair<pii, pii>> &direct, std::vector<std::pair<pii, pii>> &approx) const {
    
    dual_tree_traversal(root, root, direct, approx, factor);
}

// direct / approx is a list of (src_indices, tgt_indices) pairs where src_indices are rows and tgt_indices are cols
void ClusterTree::dual_tree_traversal(ClusterTreeNode *src, ClusterTreeNode *tgt,
                                      std::vector<std::pair<pii, pii>> &direct, std::vector<std::pair<pii, pii>> &approx, double factor) const {

    double dist = (src->center - tgt->center).norm();

    // check for far field(distance > factor * radpii)
    if (dist > factor * (src->radius + tgt->radius)) {
        auto pair = std::make_pair(src->indices, tgt->indices);
        approx.push_back(pair);
    } else {
        // if we have only leaf nodes, then we have to directly compute the matrix
        if (src->is_leaf && tgt->is_leaf) {
            auto pair = std::make_pair(src->indices, tgt->indices);
            direct.push_back(pair);
        } else {
            // recursively traverse down the nodes
            bool split_src = tgt->is_leaf || ((src->indices.second - src->indices.first) > (tgt->indices.second - tgt->indices.first));

            if (split_src) {
                dual_tree_traversal(src->left, tgt, direct, approx, factor);
                dual_tree_traversal(src->right, tgt, direct, approx, factor);
            } else {
                dual_tree_traversal(src, tgt->left, direct, approx, factor);
                dual_tree_traversal(src, tgt->right, direct, approx, factor);
            }
        }
    }
}

} // namespace bem3d