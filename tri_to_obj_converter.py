from awpy.visibility import VisibilityChecker
from awpy.data import TRIS_DIR

tris = VisibilityChecker.read_tri_file(TRIS_DIR / "de_dust2.tri")

with open("de_dust2.obj", "w") as f:
    vertex_count = 0
    for tri in tris:
        for p in (tri.p1, tri.p2, tri.p3):
            f.write(f"v {p.x} {p.y} {p.z}\n")
        f.write(f"f {vertex_count+1} {vertex_count+2} {vertex_count+3}\n")
        vertex_count += 3