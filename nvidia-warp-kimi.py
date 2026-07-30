"""
vision_coloring_fixed.py

Liest eine CS2 .dem Datei, extrahiert die Position/Blickrichtung eines Spielers
für ein Zeitfenster und prüft für JEDES Dreieck der Map, ob der Spieler es
gesehen hat. Das Ergebnis wird als .glb exportiert (gesehen = rot, ungesehen = grau).

NEU: Prüft nicht nur die Dreiecksmitte, sondern auch die 3 Eckpunkte.
     So werden keine großen Flächen mehr ausgelassen, nur weil ihre Mitte
     hinter einem kleinen Hindernis liegt.

Usage:
    python vision_coloring_fixed.py demo.dem --player "donk" --start 1:00 --end 1:20 --round 1

Requires:
    pip install awpy trimesh numpy polars
    awpy get tris   (einmalig, um die .tri Dateien zu laden)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import trimesh

from awpy import Demo
from awpy.data import TRIS_DIR
from awpy.visibility import VisibilityChecker

try:
    import warp as wp
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. HILFSFUNKTIONEN
# ---------------------------------------------------------------------------

def parse_time_to_seconds(t: str) -> float:
    """Wandelt 'MM:SS' oder reine Sekunden in Float-Sekunden um."""
    if ":" in t:
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    return float(t)


def tri_file_to_trimesh(tri_path: Path) -> trimesh.Trimesh:
    """
    Lädt eine Awpy .tri Datei direkt als trimesh-Objekt.
    Eine .tri Datei enthält die reine Geometrie der Map als Dreiecksliste.
    """
    tris = VisibilityChecker.read_tri_file(tri_path)

    vertices = []
    faces = []
    for i, tri in enumerate(tris):
        # Jedes Dreieck hat 3 Punkte (p1, p2, p3)
        for p in (tri.p1, tri.p2, tri.p3):
            vertices.append([p.x, p.y, p.z])
        # trimesh erwartt Indices: [0,1,2] für erstes Dreieck, [3,4,5] für zweites, ...
        faces.append([i * 3, i * 3 + 1, i * 3 + 2])

    return trimesh.Trimesh(
        vertices=np.array(vertices),
        faces=np.array(faces),
        process=False  # Wir wollen die Geometrie nicht verändern/vereinfachen
    )


def compute_forward_vector(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    Rechnet CS2-Yaw/Pitch in einen 3D-Richtungsvektor um.
    In CS2: yaw=0 schaut entlang der Y-Achse, positiver pitch = nach unten.
    """
    yr = np.radians(yaw_deg)
    pr = np.radians(pitch_deg)
    return np.array([
        np.cos(pr) * np.cos(yr),
        np.cos(pr) * np.sin(yr),
        -np.sin(pr),          # Negativ, weil positive pitch nach unten zeigt
    ])


def precompute_sample_points(mesh: trimesh.Trimesh, samples_per_tri: int):
    """
    Berechnet die Punkte, die wir pro Dreieck testen wollen.
    
    samples_per_tri = 1  -> nur der Schwerpunkt (schnell, kann Lücken haben)
    samples_per_tri = 4  -> Schwerpunkt + 3 Eckpunkte (langsamer, aber lückenlos)
    
    Gibt zurück:
        all_pts:     Array der Form (N * S, 3)  -> alle Punkte hintereinander
        all_tri_ids: Array der Länge N * S      -> zu welchem Dreieck jeder Punkt gehört
    """
    num_triangles = len(mesh.faces)
    centers = mesh.triangles_center  # (N, 3)

    if samples_per_tri == 1:
        pts = centers[:, np.newaxis, :]          # (N, 1, 3)
    elif samples_per_tri == 4:
        v0 = mesh.vertices[mesh.faces[:, 0]]     # Erste Ecke jedes Dreiecks
        v1 = mesh.vertices[mesh.faces[:, 1]]     # Zweite Ecke
        v2 = mesh.vertices[mesh.faces[:, 2]]     # Dritte Ecke
        pts = np.stack([centers, v0, v1, v2], axis=1)  # (N, 4, 3)
    else:
        raise ValueError("samples_per_tri muss 1 oder 4 sein")

    # Wir "plätten" das Array, damit wir alles vektorisiert berechnen können:
    # Statt (N, S, 3) -> (N*S, 3)
    all_pts = pts.reshape(-1, 3)
    all_tri_ids = np.repeat(np.arange(num_triangles), samples_per_tri)
    return all_pts, all_tri_ids


# ---------------------------------------------------------------------------
# 2. GPU-RAYCASTING (optional, mit NVIDIA Warp)
# ---------------------------------------------------------------------------

if WARP_AVAILABLE:
    @wp.kernel
    def raycast_kernel(
        mesh_id: wp.uint64,
        origins: wp.array(dtype=wp.vec3),
        directions: wp.array(dtype=wp.vec3),
        candidate_indices: wp.array(dtype=wp.int32),
        max_dist: float,
        visible_out: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        query = wp.mesh_query_ray(
            mesh_id, origins[tid], directions[tid], max_dist
        )
        if query.result and query.face == candidate_indices[tid]:
            visible_out[tid] = 1
        else:
            visible_out[tid] = 0


def build_warp_mesh(mesh: trimesh.Trimesh):
    """Lädt das Mesh einmalig auf die GPU hoch."""
    wp.init()
    points = wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device="cuda")
    indices = wp.array(mesh.faces.flatten().astype(np.int32), dtype=wp.int32, device="cuda")
    return wp.Mesh(points=points, indices=indices)


def gpu_visible_triangles(wp_mesh, eye_pos, directions_unit, candidate_tri_ids, max_dist):
    """
    Führt die Raycasts auf der GPU parallel aus.
    Gibt die IDs der Dreiecke zurück, die wirklich sichtbar sind.
    """
    n = len(candidate_tri_ids)
    if n == 0:
        return np.array([], dtype=np.int64)

    origins_np = np.tile(eye_pos.astype(np.float32), (n, 1))
    origins = wp.array(origins_np, dtype=wp.vec3, device="cuda")
    directions = wp.array(directions_unit.astype(np.float32), dtype=wp.vec3, device="cuda")
    candidate_wp = wp.array(candidate_tri_ids.astype(np.int32), dtype=wp.int32, device="cuda")
    visible_out = wp.zeros(n, dtype=wp.int32, device="cuda")

    wp.launch(
        kernel=raycast_kernel,
        dim=n,
        inputs=[wp_mesh.id, origins, directions, candidate_wp, float(max_dist)],
        outputs=[visible_out],
    )
    wp.synchronize()

    visible_mask = visible_out.numpy().astype(bool)
    return np.unique(candidate_tri_ids[visible_mask])


# ---------------------------------------------------------------------------
# 3. HAUPTFUNKTION
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Färbt Map-Geometrie basierend auf dem Sichtfeld eines Spielers ein."
    )
    parser.add_argument("demo", type=str, help="Pfad zur .dem Datei")
    parser.add_argument("--player", type=str, required=True, help="Spielername (wie in der Demo)")
    parser.add_argument("--start", type=str, required=True, help="Startzeit, z.B. '1:00' oder Sekunden")
    parser.add_argument("--end", type=str, required=True, help="Endzeit, z.B. '1:20' oder Sekunden")
    parser.add_argument("--round", type=int, default=1, help="Rundennummer (Default: 1)")
    parser.add_argument("--map", type=str, default=None, help="Map-Name, z.B. de_dust2 (Default: auto)")
    parser.add_argument("--tri", type=str, default=None, help="Pfad zur .tri Datei (Default: awpy intern)")
    parser.add_argument("--fov", type=float, default=90.0, help="Sichtfeld in Grad (Default: 90)")
    parser.add_argument("--max-distance", type=float, default=4000.0, help="Max. Distanz in Map-Einheiten")
    parser.add_argument("--tick-step", type=int, default=4, help="Jeden N-ten Tick verarbeiten (Default: 4)")
    parser.add_argument("--eye-height", type=float, default=64.0, help="Augenhöhe über dem Boden (Default: 64)")
    parser.add_argument("--samples-per-tri", type=int, default=4, choices=[1, 4],
                        help="1=nur Mitte (schnell, lückenhaft), 4=Mitte+Ecken (genau, empfohlen)")
    parser.add_argument("--no-gpu", action="store_true", help="GPU deaktivieren und CPU erzwingen")
    parser.add_argument("--out", type=str, default="seen_result.glb", help="Ausgabepfad (.glb)")
    args = parser.parse_args()

    demo_path = Path(args.demo)
    if not demo_path.exists():
        sys.exit(f"Demo nicht gefunden: {demo_path}")

    # -------------------------------------------------------------
    # 3a. Demo laden & Map erkennen
    # -------------------------------------------------------------
    print(f"Parse Demo: {demo_path} ...")
    dem = Demo(str(demo_path), verbose=False)
    dem.parse(
        player_props=["X", "Y", "Z", "pitch", "yaw"],
        events=["round_start", "round_freeze_end", "round_officially_ended"]
    )

    map_name = args.map or dem.header.get("map_name") or dem.header.get("map")
    if map_name is None:
        sys.exit("Map-Name konnte nicht automatisch erkannt werden. Bitte --map angeben.")
    print(f"Erkannte Map: {map_name}")

    # -------------------------------------------------------------
    # 3b. Kollisions-Mesh laden
    # -------------------------------------------------------------
    tri_path = Path(args.tri) if args.tri else TRIS_DIR / f"{map_name}.tri"
    if not tri_path.exists():
        sys.exit(f".tri Datei nicht gefunden: {tri_path}. Führe 'awpy get tris' aus.")

    print(f"Lade Map-Geometrie: {tri_path} ...")
    mesh = tri_file_to_trimesh(tri_path)

    # -------------------------------------------------------------
    # 3c. GPU initialisieren (falls gewünscht & verfügbar)
    # -------------------------------------------------------------
    use_gpu = False
    wp_mesh = None
    if not args.no_gpu and WARP_AVAILABLE:
        try:
            if wp.get_cuda_device_count() > 0:
                wp_mesh = build_warp_mesh(mesh)
                use_gpu = True
                print("GPU-Raycasting aktiviert (NVIDIA Warp).")
            else:
                print("Keine CUDA-GPU gefunden, nutze CPU.")
        except Exception as e:
            print(f"GPU-Init fehlgeschlagen ({e}), nutze CPU.")
    elif not args.no_gpu and not WARP_AVAILABLE:
        print("nvidia-warp nicht installiert. CPU-Modus. (pip install warp-lang für GPU-Speedup)")

    # -------------------------------------------------------------
    # 3d. Zeitfenster in Ticks umrechnen
    # -------------------------------------------------------------
    rounds_df = dem.rounds
    round_row = rounds_df.filter(pl.col("round_num") == args.round)
    if round_row.is_empty():
        sys.exit(f"Runde {args.round} nicht in der Demo gefunden.")

    # Freeze-Ende ist der Moment, in dem die Runde wirklich losgeht (Waffen können benutzt werden)
    round_start_tick = round_row["freeze_end"][0] if "freeze_end" in round_row.columns else round_row["start"][0]

    # WICHTIGER FIX: playback_ticks ist die GESAMTZAHL der Ticks, nicht die Rate!
    tick_rate = dem.header.get("tick_rate")
    if not isinstance(tick_rate, (int, float)) or tick_rate <= 0:
        tick_rate = 64  # CS2 Matchmaking Standard. Für FaceIT/Pro: 128
        print("Warnung: Tickrate nicht in Demo-Header gefunden, nehme 64 an (bei FaceIT/Pro --tick-rate 128 setzen).")

    start_sec = parse_time_to_seconds(args.start)
    end_sec = parse_time_to_seconds(args.end)
    start_tick = round_start_tick + int(start_sec * tick_rate)
    end_tick = round_start_tick + int(end_sec * tick_rate)

    # -------------------------------------------------------------
    # 3e. Spieler-Ticks filtern
    # -------------------------------------------------------------
    ticks_df = dem.ticks
    player_ticks = ticks_df.filter(
        (pl.col("name") == args.player)
        & (pl.col("round_num") == args.round)
        & (pl.col("tick") >= start_tick)
        & (pl.col("tick") <= end_tick)
    ).sort("tick")

    if player_ticks.is_empty():
        available = ticks_df.filter(pl.col("round_num") == args.round)["name"].unique().to_list()
        sys.exit(
            f"Keine Daten für Spieler '{args.player}' in Runde {args.round} gefunden.\n"
            f"Verfügbare Spieler in dieser Runde: {available}"
        )

    print(f"Gefunden: {len(player_ticks)} Ticks für {args.player}. Verarbeite jeden {args.tick_step}. Tick.")

    # -------------------------------------------------------------
    # 3f. Sample-Punkte vorberechnen (Schwerpunkt + Ecken)
    # -------------------------------------------------------------
    all_sample_pts, all_tri_ids = precompute_sample_points(mesh, args.samples_per_tri)
    num_triangles = len(mesh.faces)
    cos_half_fov = np.cos(np.radians(args.fov / 2.0))
    seen_triangle_ids = set()

    rows = player_ticks.to_pandas().iloc[::args.tick_step]
    total_ticks = len(rows)

    # -------------------------------------------------------------
    # 3g. HAUPTSCHLEIFE: Für jeden Tick prüfen, was der Spieler sieht
    # -------------------------------------------------------------
    for i, row in enumerate(rows.itertuples(index=False)):
        # Spieler-Augenposition = Bodenposition + Augenhöhe
        eye_pos = np.array([row.X, row.Y, row.Z + args.eye_height])
        forward = compute_forward_vector(row.yaw, row.pitch)
        forward = forward / (np.linalg.norm(forward) + 1e-12)  # Sicher normalisieren

        # --- FOV- & Distanz-Filter für ALLE Sample-Punkte auf einmal ---
        to_pts = all_sample_pts - eye_pos          # Vektor Auge -> Punkt
        dists = np.linalg.norm(to_pts, axis=1)     # Entfernung
        # Division durch 0 vermeiden (falls ein Punkt exakt auf dem Auge liegt)
        dists_safe = np.where(dists[:, None] == 0, 1e-6, dists[:, None])
        dirs = to_pts / dists_safe                 # Einheitsrichtungen

        cos_angle = dirs @ forward                 # Winkel zur Blickrichtung
        in_fov = cos_angle >= cos_half_fov         # Liegt im Sichtkegel?
        in_range = dists <= args.max_distance      # Nicht zu weit weg?
        candidate_mask = in_fov & in_range

        if not np.any(candidate_mask):
            continue

        # Nur die Punkte weiterverfolgen, die im FOV & in Reichweite liegen
        cand_dirs = dirs[candidate_mask]
        cand_tri_ids = all_tri_ids[candidate_mask]

        # --- RAYCAST: Ist der Weg vom Auge zum Punkt frei? ---
        if use_gpu:
            visible = gpu_visible_triangles(
                wp_mesh, eye_pos, cand_dirs, cand_tri_ids, args.max_distance
            )
        else:
            origins = np.tile(eye_pos, (len(cand_dirs), 1))
            # intersects_first gibt das ERSTE getroffene Dreieck zurück
            hit_tri_ids = mesh.ray.intersects_first(origins, cand_dirs)

            # Ein Punkt ist sichtbar, wenn das erste getroffene Dreieck
            # genau das Dreieck ist, zu dem der Punkt gehört (nichts blockt)
            visible_mask = hit_tri_ids == cand_tri_ids
            visible = np.unique(cand_tri_ids[visible_mask])

        seen_triangle_ids.update(visible.tolist())

        print(
            f"  Tick {i + 1}/{total_ticks}: "
            f"{len(cand_tri_ids)} Kandidaten, "
            f"{len(visible)} bestätigt sichtbar, "
            f"{len(seen_triangle_ids)} einzigartige Dreiecke bisher"
        )

    # -------------------------------------------------------------
    # 3h. Ergebnis einfärben & exportieren
    # -------------------------------------------------------------
    print(f"\nFertig! {len(seen_triangle_ids)} / {num_triangles} Dreiecke wurden gesehen.")

    # Grau = ungesehen, Rot = gesehen (RGBA)
    colors = np.tile([160, 160, 160, 255], (num_triangles, 1)).astype(np.uint8)
    if seen_triangle_ids:
        colors[list(seen_triangle_ids)] = [255, 0, 0, 255]

    mesh.visual.face_colors = colors
    out_path = Path(args.out)
    mesh.export(out_path)
    print(f"Exportiert nach: {out_path.resolve()}")
    print("Öffne die Datei in Blender: File -> Import -> glTF 2.0")


if __name__ == "__main__":
    main()