"""Generate the tiny stylized GLB overlays used by the local viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "viewer" / "assets"

SKIN = (224, 174, 125, 255)
SHIRT = (48, 116, 166, 255)
PANTS = (35, 43, 54, 255)
BOOT = (24, 27, 32, 255)
WEAPON = (42, 47, 52, 255)
GRENADE = (83, 103, 66, 255)
METAL = (116, 124, 130, 255)


def colored(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh.visual.face_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.faces), 1))
    return mesh


def box(extents: tuple[float, float, float], center: tuple[float, float, float], color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return colored(mesh, color)


def cylinder_between(
    start: tuple[float, float, float], end: tuple[float, float, float], radius: float,
    color: tuple[int, int, int, int], sections: int = 8,
) -> trimesh.Trimesh:
    start_value = np.asarray(start, dtype=float)
    end_value = np.asarray(end, dtype=float)
    direction = end_value - start_value
    mesh = trimesh.creation.cylinder(radius=radius, height=float(np.linalg.norm(direction)), sections=sections)
    transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
    mesh.apply_transform(transform)
    mesh.apply_translation((start_value + end_value) / 2)
    return colored(mesh, color)


def sphere(radius: float, center: tuple[float, float, float], color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    mesh.apply_translation(center)
    return colored(mesh, color)


def add(scene: trimesh.Scene, name: str, mesh: trimesh.Trimesh) -> None:
    scene.add_geometry(mesh, node_name=name, geom_name=name)


def base_player() -> trimesh.Scene:
    scene = trimesh.Scene()
    add(scene, "body_left_boot", box((8, 7, 5), (0, -5, 2.5), BOOT))
    add(scene, "body_right_boot", box((8, 7, 5), (0, 5, 2.5), BOOT))
    add(scene, "body_left_leg", cylinder_between((0, -5, 5), (0, -5, 29), 3.8, PANTS))
    add(scene, "body_right_leg", cylinder_between((0, 5, 5), (0, 5, 29), 3.8, PANTS))
    add(scene, "body_torso", box((17, 15, 25), (0, 0, 41.5), SHIRT))
    add(scene, "look_head", sphere(7, (0, 0, 60.5), SKIN))
    add(scene, "look_face", box((4, 8, 3), (6.2, 0, 60), (196, 136, 98, 255)))
    return scene


def aiming_player() -> trimesh.Scene:
    scene = base_player()
    add(scene, "look_left_arm", cylinder_between((3, -8, 50), (17, -4, 49), 3, SKIN))
    add(scene, "look_right_arm", cylinder_between((3, 8, 50), (17, 4, 49), 3, SKIN))
    add(scene, "look_weapon", box((31, 4, 4), (25, 0, 50), WEAPON))
    add(scene, "look_weapon_sight", box((5, 2, 3), (25, 0, 53.3), METAL))
    return scene


def crouching_aiming_player() -> trimesh.Scene:
    scene = trimesh.Scene()
    add(scene, "body_left_boot", box((9, 7, 5), (2, -5, 2.5), BOOT))
    add(scene, "body_right_boot", box((9, 7, 5), (2, 5, 2.5), BOOT))
    add(scene, "body_left_shin", cylinder_between((0, -5, 5), (-6, -5, 17), 3.8, PANTS))
    add(scene, "body_right_shin", cylinder_between((0, 5, 5), (-6, 5, 17), 3.8, PANTS))
    add(scene, "body_left_thigh", cylinder_between((-6, -5, 17), (2, -5, 25), 4.2, PANTS))
    add(scene, "body_right_thigh", cylinder_between((-6, 5, 17), (2, 5, 25), 4.2, PANTS))
    add(scene, "body_torso", box((17, 15, 21), (2, 0, 33.5), SHIRT))
    add(scene, "look_head", sphere(7, (2, 0, 49), SKIN))
    add(scene, "look_face", box((4, 8, 3), (8.2, 0, 48.5), (196, 136, 98, 255)))
    add(scene, "look_left_arm", cylinder_between((5, -8, 41), (19, -4, 40), 3, SKIN))
    add(scene, "look_right_arm", cylinder_between((5, 8, 41), (19, 4, 40), 3, SKIN))
    add(scene, "look_weapon", box((31, 4, 4), (27, 0, 41), WEAPON))
    add(scene, "look_weapon_sight", box((5, 2, 3), (27, 0, 44.3), METAL))
    return scene


def holding_player() -> trimesh.Scene:
    scene = base_player()
    add(scene, "look_left_arm", cylinder_between((2, -8, 50), (12, -3, 46), 3, SKIN))
    add(scene, "look_right_arm", cylinder_between((2, 8, 50), (12, 3, 46), 3, SKIN))
    add(scene, "look_grenade", sphere(4.2, (15, 0, 46), GRENADE))
    add(scene, "look_pin", cylinder_between((15, 0, 50), (15, 0, 54), 1.1, METAL, 6))
    return scene


def throwing_player() -> trimesh.Scene:
    scene = base_player()
    add(scene, "look_throw_arm", cylinder_between((1, -8, 51), (22, -4, 63), 3, SKIN))
    add(scene, "look_throw_hand", sphere(3.4, (24, -4, 64), SKIN))
    add(scene, "look_balance_arm", cylinder_between((1, 8, 50), (-11, 12, 43), 3, SKIN))
    add(scene, "look_released_grenade", sphere(3.5, (31, -4, 68), GRENADE))
    return scene


def flashbang() -> trimesh.Scene:
    scene = trimesh.Scene()
    body = trimesh.creation.cylinder(radius=4.2, height=10, sections=12)
    body.apply_translation((0, 0, 5))
    add(scene, "flash_body", colored(body, (178, 182, 177, 255)))
    add(scene, "flash_band_top", cylinder_between((0, 0, 8.1), (0, 0, 9.0), 4.35, (66, 73, 67, 255), 12))
    add(scene, "flash_band_bottom", cylinder_between((0, 0, 1.0), (0, 0, 1.9), 4.35, (66, 73, 67, 255), 12))
    add(scene, "flash_fuse", box((4.5, 4.5, 2.2), (0, 0, 11.1), METAL))
    add(scene, "flash_lever", box((8, 2.2, 1.4), (1.8, 0, 13), (85, 90, 92, 255)))
    for index in range(12):
        angle = index * np.pi * 2 / 12
        next_angle = (index + 1) * np.pi * 2 / 12
        start = (5.8 + np.cos(angle) * 2.3, np.sin(angle) * 2.3, 12.0)
        end = (5.8 + np.cos(next_angle) * 2.3, np.sin(next_angle) * 2.3, 12.0)
        add(scene, f"flash_pin_{index}", cylinder_between(start, end, 0.45, METAL, 5))
    return scene


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = {
        "player-aim.glb": aiming_player(),
        "player-crouch-aim.glb": crouching_aiming_player(),
        "player-hold.glb": holding_player(),
        "player-throw.glb": throwing_player(),
        "flashbang.glb": flashbang(),
    }
    for filename, scene in assets.items():
        (OUTPUT / filename).write_bytes(trimesh.exchange.gltf.export_glb(scene))
        print(f"Wrote {OUTPUT / filename}")


if __name__ == "__main__":
    main()
