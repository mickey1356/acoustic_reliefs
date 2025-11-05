import pyoptim.mesher as mesher

V, F = mesher.cylinder_mesher()
# write to obj
with open("cylinder.obj", "w") as f:
    for v in V:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in F:
        f.write("f " + " ".join([str(idx + 1) for idx in face]) + "\n")