import gmsh
import numpy as np
from . import helpers as H

# pts = [(0,0,0), (2,0,0), (2,2,0), (0,2,0), (0,0,12), (2,0,12), (2,2,12), (0,2,12)]
# faces = [[0,1,2,3], [4,5,6,7], [0,4,5,1], [1,5,6,2], [2,6,7,3], [3,7,4,0]]

def mesh_surface(verts, faces, e_size):
    gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.model.add("t1")

    for i, (x, y, z) in enumerate(verts):
        gmsh.model.geo.add_point(x, y, z, e_size, i+1)

    edges = {}
    for vis in faces:
        for i in range(len(vis)):
            a = 1 + vis[i]
            b = 1 + vis[(i + 1) % len(vis)]
            edge = (min(a, b), max(a, b))
            if edge not in edges:
                edges[edge] = 1 + len(edges)

    for (s, e) in edges:
        gmsh.model.geo.add_line(s, e, edges[(s, e)])


    for f, vis in enumerate(faces):
        loop = []
        for i in range(len(vis)):
            a = 1 + vis[i]
            b = 1 + vis[(i + 1) % len(vis)]
            edge = (a, b)
            if edge in edges:
                loop.append(edges[edge])
            else:
                loop.append(-edges[(b, a)])
        gmsh.model.geo.add_curve_loop(loop, f + 1)
        gmsh.model.geo.add_plane_surface([f + 1], f + 1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    _, Ps, _ = gmsh.model.mesh.getNodes()
    Ps = Ps.reshape(-1, 3)
    Es = gmsh.model.mesh.getElementFaceNodes(2, 3).reshape(-1, 3) - 1

    return Ps, Es


def box(w=0.6, b=0.6, h=0.15):
    verts = [
        [-w / 2, h, -b / 2],
        [ w / 2, h, -b / 2],
        [ w / 2, h,  b / 2],
        [-w / 2, h,  b / 2],
        [-w / 2, 0, -b / 2],
        [ w / 2, 0, -b / 2],
        [ w / 2, 0,  b / 2],
        [-w / 2, 0,  b / 2],
    ]
    faces = [[3, 2, 1, 0], [2, 3, 7, 6], [1, 2, 6, 5], [0, 1, 5, 4], [3, 0, 4, 7], [4, 5, 6, 7]]
    return verts, faces


def box_mesher(esize=0.01, w=0.6, b=0.6, h=0.15):
    verts, faces = box(w, b, h)
    return mesh_surface(verts, faces, esize)


# subdivides specific faces, and triangulates the non-triangular resultant faces by joining them to the other vertex
def face_subdivision(Ps_, Es_, faces_, n_subdivisions=1):
    Ps = Ps_.copy()
    Es = Es_.copy()
    faces = faces_.copy()

    for _ in range(n_subdivisions):
        # first collect all the edges that would be subdivided
        subdivided_edges = {}
        for (i1, i2, i3) in Es[faces]:
            edges = [(min(i1, i2), max(i1, i2)), (min(i2, i3), max(i2, i3)), (min(i1, i3), max(i1, i3))]
            for edge in edges:
                if edge not in subdivided_edges:
                    subdivided_edges[edge] = True

        new_Ps = []
        # for each of the above edges, we subdivide them, and append these new vertices
        for edge in subdivided_edges:
            va, vb = Ps[edge[0]], Ps[edge[1]]
            subdivided_edges[edge] = len(Ps) + len(new_Ps)
            new_Ps.append((va + vb) / 2)
        
        nPs = np.vstack((Ps, new_Ps))
        nEs = []
        nfaces = []
        to_subdivide = np.zeros(len(Es))
        to_subdivide[faces] = 1
        # now we can generate the new faces
        # for each existing face, we check if its a face to be subdivided, if so, do it
        # if not, check if any of its edges was subdivided, and if so, we have to generate new triangles
        # otherwise, just add it as is
        for sub, (i1, i2, i3) in zip(to_subdivide, Es):
            edges = [(min(i1, i2), max(i1, i2)), (min(i2, i3), max(i2, i3)), (min(i1, i3), max(i1, i3))]
            i4, i5, i6 = [subdivided_edges.get(e, -1) for e in edges]

            # we're going to add 4 new faces, which formed the original face
            # keep track of which faces can be further subdivided
            if sub == 1:
                nfaces.extend([len(nEs), len(nEs) + 1, len(nEs) + 2, len(nEs) + 3])

            # only 1 edge is subdivided
            if i4 >= 0 and i5 == i6 == -1:
                nEs.append([i4, i2, i3])
                nEs.append([i1, i4, i3])
            elif i5 >= 0 and i4 == i6 == -1:
                nEs.append([i1, i2, i5])
                nEs.append([i1, i5, i3])
            elif i6 >= 0 and i4 == i5 == -1:
                nEs.append([i1, i2, i6])
                nEs.append([i2, i3, i6])
            # 2 edges are subdivided
            elif i4 == -1 and i5 >= 0 and i6 >= 0:
                nEs.append([i1, i2, i6])
                nEs.append([i2, i5, i6])
                nEs.append([i5, i3, i6])
            elif i5 == -1 and i4 >= 0 and i6 >= 0:
                nEs.append([i1, i4, i6])
                nEs.append([i2, i6, i4])
                nEs.append([i2, i3, i6])
            elif i6 == -1 and i4 >= 0 and i5 >= 0:
                nEs.append([i1, i4, i3])
                nEs.append([i3, i4, i5])
                nEs.append([i4, i2, i5])
            elif i4 >= 0 and i5 >= 0 and i6 >= 0:
                nEs.append([i1, i4, i6])
                nEs.append([i4, i2, i5])
                nEs.append([i6, i5, i3])
                nEs.append([i4, i5, i6])
            else:
                nEs.append([i1, i2, i3])
        nEs = np.array(nEs, dtype=int)

        Ps = nPs.copy()
        Es = nEs.copy()
        faces = nfaces.copy()
    return Ps, Es
