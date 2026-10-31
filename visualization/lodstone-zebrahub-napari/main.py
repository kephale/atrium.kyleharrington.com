# /// script
# title = "Zebrahub in napari with Lodstone"
# description = "Stream one Zebrahub timepoint into napari with a camera shared by every lodstone-zebrahub entry"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["lodstone", "napari", "ome-zarr", "multiscale", "zebrahub"]
# repository = "https://github.com/kephale/lodstone"
# requires-python = ">=3.12"
# dependencies = [
#     "lodstone[ome-zarr]==0.1.0a2",
#     "napari[pyqt6] @ git+https://github.com/kephale/napari.git@1ee4499845d2afd20bd1cee76dd629a7408d2c4f",
#     "numpy",
# ]
# ///
"""Stream one Zebrahub timepoint into napari with the shared camera."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-mib", type=int, default=64)
    parser.add_argument("--interval-mib", type=int, default=512)
    parser.add_argument("--screenshot", help="save the canvas here, then exit")
    parser.add_argument("--report", help="save a JSON run report here")
    parser.add_argument("--seconds", type=float, default=60.0)
    arguments = parser.parse_args()

    import napari
    from lodstone import PerformanceRecorder
    from lodstone.adapters.napari import NapariController
    from lodstone.sources import OMEZarrSource
    from napari.experimental._lodstone_loading import _camera_view
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    MIB = 1024**2
    camera = SCENE["camera"]
    width, height = camera["viewport"]
    print(f"scene {SCENE_ID}: napari")
    source = OMEZarrSource.open(SCENE["url"])
    viewer = napari.Viewer(title="Zebrahub in napari with Lodstone")
    controller = NapariController(
        viewer,
        source,
        # ZSNS001 axes are (t, c, z, y, x).
        fixed_index={0: SCENE["time"], 1: SCENE["channel"]},
        tile_max_bytes_3d=arguments.tile_mib * MIB,
        interval_max_bytes=arguments.interval_mib * MIB,
        name=f"ZSNS001 t={SCENE['time']}",
        contrast_limits=tuple(SCENE["clims"]),
        colormap=SCENE["colormap"],
        rendering=SCENE["rendering"],
    )
    viewer.dims.ndisplay = 3
    qt_viewer = viewer.window._qt_viewer
    qt_viewer.dockLayerList.setVisible(False)
    qt_viewer.dockLayerControls.setVisible(False)

    def live_world_to_clip():
        """Read the matrix napari is drawing with, in ZYX world coordinates."""
        canvas = qt_viewer.canvas
        canvas_height, canvas_width = canvas.size
        points = np.zeros((4, 4))
        points[:, 3] = 1.0
        points[1:, :3] = np.eye(3)[::-1]
        mapped = np.asarray(canvas.view.scene.transform.map(points), dtype=float)
        framebuffer = np.eye(4)
        framebuffer[:, 3] = mapped[0]
        framebuffer[:, :3] = (mapped[1:] - mapped[0]).T
        to_clip = np.eye(4)
        to_clip[0] = (2.0 / canvas_width, 0.0, 0.0, -1.0)
        to_clip[1] = (0.0, -2.0 / canvas_height, 0.0, 1.0)
        return to_clip @ framebuffer

    def fit_window() -> None:
        window = viewer.window._qt_window
        canvas_height, canvas_width = qt_viewer.canvas.size
        window.resize(
            window.width() + width - canvas_width,
            window.height() + height - canvas_height,
        )

    def apply_camera() -> None:
        right, up, forward = (axis[::-1] for axis in camera_axes())
        # Progressive loading slices asynchronously, so this runs from the
        # event loop. napari's camera uses ZYX, like the shared world.
        viewer.reset_view()
        viewer.camera.perspective = 0
        # napari's default handedness mirrors screen X for the shared forward
        # direction. Looking from the opposite side gives the same orthographic
        # MIP without the mirror.
        viewer.camera.set_view_direction(tuple(-forward), tuple(up))
        viewer.camera.center = tuple(camera["center_zyx_um"])
        canvas_height = qt_viewer.canvas.size[0]
        viewer.camera.zoom = canvas_height / camera["height_um"]

    QTimer.singleShot(100, fit_window)
    QTimer.singleShot(400, apply_camera)

    stream = controller.loader._lodstone_stream
    recorder = PerformanceRecorder(stream, host="napari", backend="vispy")

    if arguments.screenshot or arguments.report:

        def finish() -> None:
            viewport = tuple(qt_viewer.canvas.size[::-1])
            write_report(
                arguments.report,
                "napari",
                "vispy",
                live_world_to_clip(),
                viewport,
                controller.loader._submitted_plan,
                recorder.records()[-1:],
            )
            if arguments.screenshot:
                viewer.screenshot(arguments.screenshot, canvas_only=True)
            QApplication.instance().quit()

        QTimer.singleShot(round(arguments.seconds * 1000), finish)

    try:
        napari.run()
    finally:
        recorder.close()
        controller.close()


if __name__ == "__main__":
    main()
