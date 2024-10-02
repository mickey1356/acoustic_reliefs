#include <fstream>
#include <iostream>

#include "read_obj.h"

namespace bem3d {

int read_obj(const std::string &fname, mat3 &Ps, imat3 &Es)  {

    std::ifstream fs(fname);
    std::string line;
    std::vector<vec3> verts;
    std::vector<ivec3> tris;

    while (std::getline(fs, line)) {
        char c;
        if (line[0] == 'v') {
            double x, y, z;
            std::istringstream iss(line);
            if (!(iss >> c >> x >> y >> z)) {
                std::cerr << "Invalid vertex specification" << std::endl;
                return -1;
            }
            vec3 v(3);
            v << x, y, z;
            verts.push_back(v);
        } else if (line[0] == 'f') {
            int v1, v2, v3;
            std::istringstream iss(line);
            if (!(iss >> c >> v1 >> v2 >> v3)) {
                std::cerr << "Invalid face specification (only triangles supported)" << std::endl;
                return -1;
            }
            ivec3 t(3);
            t << v1 - 1, v2 - 1, v3 - 1;
            tris.push_back(t);
        }
    }
    Ps.setZero(verts.size(), 3);
    for (int i = 0; i < verts.size(); i++) {
        Ps.row(i) = verts[i];
    }

    Es.setZero(tris.size(), 3);
    for (int i = 0; i < tris.size(); i++) {
        Es.row(i) = tris[i];
    }

    return static_cast<int>(tris.size());
}

void write_obj(const std::string &fname, const mat3 &Ps, const imat3 &Es) {
    Eigen::IOFormat fmt(Eigen::FullPrecision, Eigen::DontAlignCols);

    std::ofstream out(fname);

    for (int i = 0; i < Ps.rows(); i++) {
        out << "v " << Ps.row(i).format(fmt) << std::endl;
    }

    for (int i = 0; i < Es.rows(); i++) {
        out << "f " << (1 + Es.row(i).array()).format(fmt) << std::endl;
    }
}

} // namespace bem3d