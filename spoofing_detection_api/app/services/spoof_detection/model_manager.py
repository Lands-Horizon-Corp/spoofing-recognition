from __future__ import annotations

import threading
import time

from app.core.utils import MediaPipeUtils
from app.core.utils import mp_utils
from app.services.spoof_detection.spoof_model import spoof_detector
from app.services.spoof_detection.spoof_model import SpoofDetector


class ModelManager:
    def __init__(self, spoof_detector: SpoofDetector, mp_utils: MediaPipeUtils, ttl_seconds: int = 300, cleanup_interval: int = 60):  # noqa: E501

        if mp_utils is None:
            raise ValueError('mp_utils cannot be None')
        if spoof_detector is None:
            raise ValueError('spoof_detector cannot be None')

        self.mp_utils = mp_utils
        self.spoof_detector = spoof_detector
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval

        # Track the last time the model was used
        self.last_accessed_time = time.time()
        self.lock = threading.Lock()

        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True)
        self.monitor_thread.start()

    def get_resources(self):
        with self.lock:
            # Update timestamp every time the model is requested
            self.last_accessed_time = time.time()

            if self.mp_utils is None:
                raise ValueError('mp_utils cannot be None')
            if self.spoof_detector is None:
                raise ValueError('spoof_detector cannot be None')

            if self.mp_utils.face_mesh_detector is None:
                self.mp_utils.load_model()
                # Logic to reload if it was previously unloaded
            if self.spoof_detector.model is None:
                self.spoof_detector.load_model()

            return self.spoof_detector, self.mp_utils

    def unload_resources(self):
        """Internal method to free memory."""
        with self.lock:
            if self.spoof_detector.model is not None:
                print('TTL expired: Unloading detector.')
                self.spoof_detector.unload_model()

            if self.mp_utils is not None:
                print('TTL expired: Closing MediaPipe Utils.')
                self.mp_utils.close()

    def _cleanup_loop(self):
        while not self.stop_event.is_set():
            time.sleep(self.cleanup_interval)

            # Check if the idle time exceeds TTL
            idle_time = time.time() - self.last_accessed_time
            if idle_time >= self.ttl_seconds:
                self.unload_resources()

    def stop(self):
        self.stop_event.set()
        self.monitor_thread.join()


model_manager = ModelManager(spoof_detector=spoof_detector, mp_utils=mp_utils)
