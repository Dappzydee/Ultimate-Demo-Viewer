"""Smoke tests for the dependency-free local viewer server."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from viewer import ViewerServer


class ViewerServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_directory.name) / "example.glb"
        self.model_path.write_bytes(b"glTF-test-payload")
        self.server = ViewerServer(("127.0.0.1", 0), self.model_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp_directory.cleanup()

    def test_configuration_identifies_initial_model(self) -> None:
        with urlopen(f"{self.base_url}/viewer-config.json") as response:
            self.assertEqual(
                json.load(response),
                {"modelUrl": "/model.glb", "modelName": "example.glb"},
            )

    def test_serves_application_and_model(self) -> None:
        with urlopen(f"{self.base_url}/") as response:
            application = response.read()
            self.assertIn(b"CS2 Demo Analyzer", application)
            self.assertIn(b'<canvas id="canvas" tabindex="0"', application)
            self.assertIn(b'role="tablist"', application)
            self.assertIn(b'aria-controls="vision-controls"', application)
            self.assertIn(b'aria-controls="flash-controls"', application)
            self.assertIn(b'aria-controls="lineup-controls"', application)
            self.assertIn(b'id="flash-controls" class="control-stack" role="tabpanel"', application)
            self.assertIn(b'id="lineup-controls" class="control-stack" role="tabpanel"', application)
            self.assertIn(b'id="lineup-list"', application)
            self.assertIn(b'id="lineup-round-filter"', application)
            self.assertIn(b'id="lineup-grenade-filter"', application)
            self.assertIn(b'id="lineup-player-filter"', application)
            self.assertIn(b'id="lineup-reference-filter"', application)
            self.assertIn(b'id="lineup-camera-button"', application)
            self.assertIn(b'id="lineup-aim-distance"', application)
            self.assertIn(b'id="lineup-camera-fov"', application)
            self.assertIn(b'id="lineup-crosshair"', application)
            self.assertIn(b'id="copy-lineup-commands-button"', application)
            self.assertIn(b'id="export-lineups-json-button"', application)
            self.assertIn(b'id="export-lineups-cfg-button"', application)
            self.assertIn(b'id="flash-event-list"', application)
            self.assertIn(b'id="focus-flash-toggle"', application)
            self.assertIn(b'id="flash-camera-button"', application)
            self.assertIn(b'id="move-flash-button"', application)
            self.assertIn(b'id="move-flash-status"', application)
            self.assertIn(b'id="placed-flash-origin"', application)
            self.assertIn(b'id="copy-flash-position-button"', application)
            self.assertNotIn(b'id="flash-x"', application)
            self.assertIn(b'id="player-marker-toggle"', application)
            self.assertIn(b'id="start-time" class="clock-output"', application)
            self.assertIn(b'id="vision-life-status"', application)
            self.assertIn(b'id="history-list"', application)
            self.assertIn(b'id="discarded-history-list"', application)
            self.assertIn(b'id="discarded-history-count"', application)
            self.assertIn(b'id="history-warning"', application)
            self.assertIn(b'id="analyze-vision-button"', application)
            self.assertIn(b'id="analyze-flash-button"', application)
        with urlopen(f"{self.base_url}/styles.css") as response:
            stylesheet = response.read()
            self.assertIn(b"[hidden] { display: none !important; }", stylesheet)
            self.assertIn(b"#viewport.flash-camera-active::before", stylesheet)
            self.assertIn(b"#viewport.move-flash-active::before", stylesheet)
            self.assertIn(b"#viewport.lineup-camera-active::before", stylesheet)
            self.assertIn(b"#viewport.lineup-camera-active #lineup-crosshair", stylesheet)
            self.assertIn(b".lineup-row.selected", stylesheet)
        with urlopen(f"{self.base_url}/renderer.js") as response:
            renderer = response.read()
            self.assertIn(b"focusPoint(position", renderer)
            self.assertIn(b"setFlashCamera(position", renderer)
            self.assertIn(b"this.flashCameraPosition", renderer)
            self.assertIn(b"setPlayerMarker(position", renderer)
            self.assertIn(b"setInteractionLocked(locked", renderer)
            self.assertIn(b"getMovementBasis()", renderer)
            self.assertIn(b"translateCamera(offset", renderer)
            self.assertIn(b"setLineupVisualization(value)", renderer)
            self.assertIn(b"setLineupCamera(pose)", renderer)
            self.assertIn(b"setLineupCameraFov(degrees)", renderer)
            self.assertIn(b"setOverlayAssets(models)", renderer)
            self.assertIn(b"setVisionPreview(value)", renderer)
            self.assertIn(b'"lineup-pin-pull"', renderer)
            self.assertIn(b'"lineup-detonation"', renderer)
            self.assertIn(b"frameLineup()", renderer)
        with urlopen(f"{self.base_url}/app.js") as response:
            application_script = response.read()
            self.assertIn(b"copyDemoFlashPosition(index)", application_script)
            self.assertIn(b"prepareFlashMoverForAnalysis()", application_script)
            self.assertIn(b"setResultDiscarded(resultId, discarded)", application_script)
            self.assertIn(b"configureLineupFilters()", application_script)
            self.assertIn(b"populateLineups()", application_script)
            self.assertIn(b"copySelectedLineupCommands()", application_script)
            self.assertIn(b"exportFilteredLineups(format)", application_script)
            self.assertIn(b'apiJson("/api/preview/player"', application_script)
            self.assertIn(b"loadOverlayAssets()", application_script)
            self.assertNotIn(b"sessionResultIds", application_script)
            self.assertNotIn(b"history-save", application_script)
            self.assertNotIn(b"finalizeFlashMovement", application_script)
        with urlopen(f"{self.base_url}/model.glb") as response:
            self.assertEqual(response.headers["Content-Type"], "model/gltf-binary")
            self.assertEqual(response.read(), b"glTF-test-payload")
        for asset in ("player-aim.glb", "player-hold.glb", "player-throw.glb", "flashbang.glb"):
            with urlopen(f"{self.base_url}/assets/{asset}") as response:
                self.assertEqual(response.read(4), b"glTF")

    def test_does_not_serve_files_outside_viewer_assets(self) -> None:
        with self.assertRaises(HTTPError) as error:
            urlopen(f"{self.base_url}/../README.md")
        self.assertEqual(error.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
