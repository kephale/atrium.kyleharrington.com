# /// script
# title = "Zebrahub in SceneX with Lodstone"
# description = "Stream one Zebrahub timepoint into SceneX with a camera shared by every lodstone-zebrahub entry"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.1.1"
# keywords = ["lodstone", "scenex", "ome-zarr", "multiscale", "zebrahub"]
# repository = "https://github.com/kephale/lodstone"
# requires-python = ">=3.12"
# dependencies = [
#     "lodstone[ome-zarr]==0.1.0a2",
#     "scenex @ git+https://github.com/kephale/scenex.git@0355e0cbe25ac02b0759b8226727c65a3499a279",
#     "numpy",
#     "pygfx>=0.13",
#     "rendercanvas>=2.5.1",
#     "vispy>=0.15",
#     "pyopengl>=3.1",
#     "pyqt6",
#     "qtpy",
# ]
# ///
"""Stream one Zebrahub timepoint into SceneX with the shared camera."""

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


def scene_camera_matrix():
    """Return the camera-to-world matrix in SceneX's XYZ coordinates."""
    camera = SCENE["camera"]
    right, up, forward = camera_axes()
    center = np.asarray(camera["center_zyx_um"], dtype=float)[::-1]
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = center - forward * camera["depth_um"] / 4
    return pose


def draw_on_demand(canvas):
    """Stop SceneX's PyGFX canvas from redrawing continuously.

    Its draw loop schedules another frame after every frame, which keeps a
    volume scene rendering at full cost while nothing changes.
    """
    from scenex.adaptors._pygfx._adaptor_registry import get_adaptor

    adaptor = get_adaptor(canvas)
    adaptor._renderer.request_draw(lambda: adaptor._draw(adaptor._renderer))
    return adaptor._wgpu_canvas.request_draw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pygfx", "vispy"), default="pygfx")
    parser.add_argument("--memory-mib", type=int, default=256)
    parser.add_argument("--screenshot", help="save the window here, then exit")
    parser.add_argument("--report", help="save a JSON run report here")
    parser.add_argument("--seconds", type=float, default=60.0)
    arguments = parser.parse_args()

    import scenex as snx
    from lodstone import PerformanceRecorder, View
    from lodstone.adapters import SceneXController
    from lodstone.sources import OMEZarrSource
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication
    from scenex.utils.projections import orthographic

    snx.use(arguments.backend)
    camera = SCENE["camera"]
    width, height = camera["viewport"]
    source = OMEZarrSource.open(
        SCENE["url"], fixed_index={"t": SCENE["time"], "c": SCENE["channel"]}
    )
    print(f"scene {SCENE_ID}: SceneX + {arguments.backend}")

    view = snx.View(
        scene=snx.Scene(children=[]),
        camera=snx.Camera(interactive=True),
    )
    canvas = snx.Canvas(width=width, height=height, views=[view])
    snx.show(canvas)
    request_draw = draw_on_demand(canvas) if arguments.backend == "pygfx" else None

    def apply_camera() -> None:
        view.camera.transform = snx.Transform(scene_camera_matrix().T)
        aspect = width / height
        view.camera.projection = orthographic(
            camera["height_um"] * aspect, camera["height_um"], camera["depth_um"]
        )

    def on_presented(publication) -> None:
        handle = controller.target.handles[publication.level]
        handle.set_clims(tuple(SCENE["clims"]))
        if not posed:
            # The first publication fits the view to the data; undo that once
            # so the shared camera stands and the mouse owns it afterwards.
            posed.append(True)
            apply_camera()

    def on_wheel(event) -> bool:
        # Orbit zooms by moving the camera, which an orthographic view ignores.
        if not isinstance(event, snx.app.events.WheelEvent):
            return False
        step = 2 ** (event.angle_delta[1] * 0.001)
        view.camera.projection = view.camera.projection.scaled((step, step, 1.0))
        return True

    posed: list[bool] = []
    apply_camera()
    view.camera.controller = snx.Orbit(
        center=tuple(float(value) for value in camera["center_zyx_um"][::-1])
    )
    view.set_event_filter(on_wheel)
    if request_draw is not None:
        view.camera.events.transform.connect(lambda *_: request_draw())
        view.camera.events.projection.connect(lambda *_: request_draw())
    controller = SceneXController(
        view,
        source,
        memory_limit=arguments.memory_mib * 1024**2,
        on_presented=on_presented,
    )
    if request_draw is not None:
        controller.canvas.refresh = request_draw
        # The final mask lands after the last refresh, so draw on status changes.
        seen: list[object] = [None]

        def draw_on_status_change() -> None:
            status = controller.stream.performance.status
            if (status.generation, status.state) != seen[0]:
                seen[0] = (status.generation, status.state)
                request_draw()

        repaint = QTimer()
        repaint.timeout.connect(draw_on_status_change)
        repaint.start(250)
    recorder = PerformanceRecorder(
        controller.stream, host="scenex", backend=arguments.backend
    )
    plan = controller.update(
        View(
            displayed_axes=(0, 1, 2),
            index=(None, None, None),
            viewport=(width, height),
            world_to_clip=world_to_clip(),
        )
    )

    if arguments.screenshot or arguments.report:

        def finish() -> None:
            snapshot = controller.snapshot()
            write_report(
                arguments.report,
                "scenex",
                arguments.backend,
                snapshot.world_to_clip,
                snapshot.viewport,
                plan,
                recorder.records()[-1:],
            )
            if arguments.screenshot:
                app = QApplication.instance()
                window = app.topLevelWidgets()[0]
                app.primaryScreen().grabWindow(int(window.winId())).save(
                    arguments.screenshot
                )
            QApplication.instance().quit()

        QTimer.singleShot(round(arguments.seconds * 1000), finish)

    try:
        snx.run()
    finally:
        recorder.close()
        controller.close()


if __name__ == "__main__":
    main()
