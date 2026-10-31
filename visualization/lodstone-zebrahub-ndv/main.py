# /// script
# title = "Zebrahub in ndv with Lodstone"
# description = "Stream one Zebrahub timepoint into ndv with a camera shared by every lodstone-zebrahub entry"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["lodstone", "ndv", "ome-zarr", "multiscale", "zebrahub"]
# repository = "https://github.com/kephale/lodstone"
# requires-python = ">=3.12"
# dependencies = [
#     "lodstone[ome-zarr]==0.1.0a2",
#     "ndv[pyqt,vispy,pygfx] @ git+https://github.com/kephale/ndv.git@8d1f6e76c27d405ababd471d83616f0e1935d6b4",
#     "numpy",
# ]
# ///
"""Stream one Zebrahub timepoint into ndv with the shared camera."""

import argparse
import hashlib
import json

import numpy as np

# >>> lodstone-zebrahub shared scene (identical in every entry)
SCENE = {
    "url": (
        "https://public.czbiohub.org/royerlab/zebrahub/imaging/"
        "single-objective/ZSNS001.ome.zarr"
    ),
    "time": 400,
    "channel": 0,
    "clims": [0, 400],
    "colormap": "gray",
    "rendering": "mip",
    "camera": {
        "projection": "orthographic",
        "center_zyx_um": [277.0, 477.0, 532.0],
        "azimuth_deg": 20.0,
        "elevation_deg": 30.0,
        "height_um": 1300.0,
        "depth_um": 3000.0,
        "viewport": [900, 900],
    },
}
SCENE_ID = hashlib.sha256(json.dumps(SCENE, sort_keys=True).encode()).hexdigest()[:12]


def camera_axes():
    """Return screen right, screen up, and view direction in scene XYZ.

    At azimuth 0 and elevation 0 the camera looks along +Z with +X to the
    right and -Y up, which matches a 2-D image display.
    """
    azimuth = np.radians(SCENE["camera"]["azimuth_deg"])
    elevation = np.radians(SCENE["camera"]["elevation_deg"])
    around_up = np.array(
        [
            [np.cos(azimuth), 0, np.sin(azimuth)],
            [0, 1, 0],
            [-np.sin(azimuth), 0, np.cos(azimuth)],
        ]
    )
    around_right = np.array(
        [
            [1, 0, 0],
            [0, np.cos(elevation), -np.sin(elevation)],
            [0, np.sin(elevation), np.cos(elevation)],
        ]
    )
    rotation = around_right @ around_up
    right, up, forward = (
        rotation @ (1.0, 0.0, 0.0),
        rotation @ (0.0, -1.0, 0.0),
        rotation @ (0.0, 0.0, 1.0),
    )
    return right, up, forward


def world_to_clip():
    """Return the shared camera as a matrix over ZYX world coordinates."""
    camera = SCENE["camera"]
    right, up, forward = (axis[::-1] for axis in camera_axes())
    width, height = camera["viewport"]
    half_h = camera["height_um"] / 2
    half_w = half_h * width / height
    center = np.asarray(camera["center_zyx_um"], dtype=float)
    matrix = np.eye(4)
    matrix[0, :3] = right / half_w
    matrix[1, :3] = up / half_h
    matrix[2, :3] = forward / (camera["depth_um"] / 2)
    matrix[:3, 3] = -matrix[:3, :3] @ center
    return matrix


def write_report(path, host, backend, live_matrix, viewport, plan, performance):
    """Save what this run actually used, for comparison across viewers."""
    if path is None:
        return
    target = world_to_clip()
    live = np.asarray(live_matrix, dtype=float)
    report = {
        "scene_id": SCENE_ID,
        "host": host,
        "backend": backend,
        "viewport": list(viewport),
        "live_world_to_clip": live.tolist(),
        "max_screen_error": float(np.abs(live[:2] - target[:2]).max()),
        "target_level": None if plan is None else plan.target_level,
        "wanted_tiles": None if plan is None else len(plan.wanted),
        "performance": performance,
    }
    with open(path, "w") as stream:
        json.dump(report, stream, indent=2, default=str)
# <<< lodstone-zebrahub shared scene


def rotation_quaternion(matrix):
    """Return the (w, x, y, z) unit quaternion of a rotation matrix."""
    trace = np.trace(matrix)
    values = np.array(
        [
            1 + trace,
            1 + 2 * matrix[0, 0] - trace,
            1 + 2 * matrix[1, 1] - trace,
            1 + 2 * matrix[2, 2] - trace,
        ]
    )
    pivot = int(np.argmax(values))
    pairs = (
        (matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0],
         matrix[1, 0] - matrix[0, 1]),
        (matrix[2, 1] - matrix[1, 2], matrix[0, 1] + matrix[1, 0],
         matrix[0, 2] + matrix[2, 0]),
        (matrix[0, 2] - matrix[2, 0], matrix[0, 1] + matrix[1, 0],
         matrix[1, 2] + matrix[2, 1]),
        (matrix[1, 0] - matrix[0, 1], matrix[0, 2] + matrix[2, 0],
         matrix[1, 2] + matrix[2, 1]),
    )
    root = np.sqrt(values[pivot])
    rest = np.asarray(pairs[pivot]) / root
    quaternion = np.insert(rest, pivot, root) / 2
    return quaternion


def apply_vispy_camera(canvas) -> None:
    from vispy.util.quaternion import Quaternion

    camera = SCENE["camera"]
    right, up, forward = camera_axes()
    live = canvas._camera
    # ArcballCamera rotates a Z-up frame and reads its axis as (x, z, y).
    rotation = np.column_stack((right, forward, up))
    w, x, y, z = rotation_quaternion(rotation)
    live._quaternion = Quaternion(w, x, z, y)
    live.center = tuple(np.asarray(camera["center_zyx_um"], dtype=float)[::-1])
    live.scale_factor = camera["height_um"]
    live.view_changed()


def apply_pygfx_camera(canvas) -> None:
    camera = SCENE["camera"]
    right, up, forward = camera_axes()
    center = np.asarray(camera["center_zyx_um"], dtype=float)[::-1]
    live = canvas._camera
    live.world.reference_up = tuple(up)
    live.local.position = tuple(center - forward * camera["depth_um"] / 4)
    live.look_at(tuple(center))
    live.height = camera["height_um"]
    live.width = camera["height_um"]
    live.zoom = 1
    canvas.refresh()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pygfx", "vispy"), default="pygfx")
    parser.add_argument("--memory-mib", type=int, default=256)
    parser.add_argument("--screenshot", help="save the window here, then exit")
    parser.add_argument("--report", help="save a JSON run report here")
    parser.add_argument("--seconds", type=float, default=60.0)
    arguments = parser.parse_args()

    import ndv
    from lodstone import PerformanceRecorder, View
    from lodstone.adapters.ndv import NDVController
    from lodstone.sources import OMEZarrSource
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    ndv.set_canvas_backend(arguments.backend)
    width, height = SCENE["camera"]["viewport"]
    print(f"scene {SCENE_ID}: ndv + {arguments.backend}")
    source = OMEZarrSource.open(
        SCENE["url"], fixed_index={"t": SCENE["time"], "c": SCENE["channel"]}
    )
    viewer = ndv.ArrayViewer()
    window = viewer.widget()
    surface = viewer.canvas.frontend_widget()
    apply_camera = (
        apply_vispy_camera if arguments.backend == "vispy" else apply_pygfx_camera
    )

    def on_presented(publication) -> None:
        handle = controller.target.handles[publication.level]
        handle.set_clims(tuple(SCENE["clims"]))
        if not posed:
            posed.append(True)
            apply_camera(viewer.canvas)
            # Lodstone ignores camera changes during presentation, and a window
            # show can already have replanned from the default camera. Posing
            # again afterwards makes the shared camera the last one it sees.
            QTimer.singleShot(300, lambda: apply_camera(viewer.canvas))

    posed: list[bool] = []
    controller = NDVController(
        viewer,
        source,
        memory_limit=arguments.memory_mib * 1024**2,
        on_presented=on_presented,
    )
    recorder = PerformanceRecorder(
        controller.stream, host="ndv", backend=arguments.backend
    )
    plan = controller.update(
        View(
            displayed_axes=(0, 1, 2),
            index=(None, None, None),
            viewport=(width, height),
            world_to_clip=world_to_clip(),
        )
    )
    viewer.show()
    window.resize(
        window.width() + width - surface.width(),
        window.height() + height - surface.height(),
    )
    QApplication.processEvents()
    window.resize(
        window.width() + width - surface.width(),
        window.height() + height - surface.height(),
    )

    if arguments.screenshot or arguments.report:

        def finish() -> None:
            viewport, live = viewer.canvas.camera_state()
            write_report(
                arguments.report,
                "ndv",
                arguments.backend,
                live,
                viewport,
                plan,
                recorder.records()[-1:],
            )
            if arguments.screenshot:
                app = QApplication.instance()
                corner = surface.mapToGlobal(surface.rect().topLeft())
                app.primaryScreen().grabWindow(
                    0, corner.x(), corner.y(), surface.width(), surface.height()
                ).save(arguments.screenshot)
            QApplication.instance().quit()

        QTimer.singleShot(round(arguments.seconds * 1000), finish)

    try:
        ndv.run_app()
    finally:
        recorder.close()
        controller.close()


if __name__ == "__main__":
    main()
