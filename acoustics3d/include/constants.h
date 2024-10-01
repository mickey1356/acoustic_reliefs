#pragma once

// since we're using openmp, disable eigen's multithreading
#define EIGEN_DONT_PARALLELIZE

#include <complex>

#include <Eigen/Dense>

#define pii std::pair<int, int>

#define PI 3.14159265358979323846
#define SPEED_OF_SOUND 344.0
#define EPSILON 1e-6

#define APPROX_ACA_MAX_K 300
#define Q_ACA_MAX_K 700

namespace bem3d {
using namespace std::complex_literals;
using Eigen::placeholders::all;

typedef Eigen::VectorXd vec;
typedef Eigen::VectorXi ivec;
typedef Eigen::Vector3d vec3;
typedef Eigen::Vector3i ivec3;
typedef Eigen::VectorXcd cvec;

typedef Eigen::MatrixXd mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> mat3;
typedef Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> imat3;
typedef Eigen::MatrixXcd cmat;

const Eigen::RowVector3d SRC_PT(0, 100, 0);
// const Eigen::RowVector3d SRC_PT(0, 0, 100);
// constexpr double LISTENER_D_ELEV = 5, LISTENER_D_AZIM = 5;
constexpr double LISTENER_RADIUS = 50;

constexpr double QUAD_WEIGHTS_AND_POINTS[6][3] = {
    {0.223381589678011, 0.445948490915965, 0.108103018168070},
    {0.223381589678011, 0.445948490915965, 0.445948490915965},
    {0.223381589678011, 0.108103018168070, 0.445948490915965},
    {0.109951743655322, 0.091576213509771, 0.816847572980459},
    {0.109951743655322, 0.091576213509771, 0.091576213509771},
    {0.109951743655322, 0.816847572980459, 0.091576213509771},
};
}; // namespace bem3d
