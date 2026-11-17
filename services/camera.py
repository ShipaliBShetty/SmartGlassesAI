import cv2
import threading
import time
from config import CAMERA_SOURCE


class Camera:
    def __init__(self):
        self.camera = None
        self.frame = None
        self.success = False

        self.running = True
        self.lock = threading.Lock()

        print("Connecting to ESP32-CAM...")

        self._connect()

        self.thread = threading.Thread(
            target=self._update,
            daemon=True
        )

        self.thread.start()

        print("ESP32-CAM camera service ready.")


    def _connect(self):
        """
        Connect or reconnect to the ESP32-CAM stream.
        """

        if not self.running:
            return False

        # Release previous connection if it exists
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass

        self.camera = cv2.VideoCapture(CAMERA_SOURCE)

        # Try to reduce video buffering
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.camera.isOpened():
            print("ESP32-CAM connected.")
            return True

        print("ESP32-CAM connection failed.")

        return False


    def _update(self):
        """
        Background thread that continuously reads frames
        from ESP32-CAM.

        Only the newest frame is stored.
        """

        while self.running:

            # Camera not connected
            if self.camera is None or not self.camera.isOpened():

                if not self.running:
                    break

                print("Reconnecting to ESP32-CAM...")

                time.sleep(1)

                if self.running:
                    self._connect()

                continue


            # Read frame
            success, frame = self.camera.read()


            # Program is shutting down
            if not self.running:
                break


            # Valid frame received
            if success and frame is not None:

                with self.lock:

                    self.frame = frame
                    self.success = True


            # Stream failure
            else:

                if not self.running:
                    break

                print("ESP32-CAM stream lost. Reconnecting...")

                with self.lock:
                    self.success = False

                try:
                    self.camera.release()
                except Exception:
                    pass

                time.sleep(1)

                if self.running:
                    self._connect()


    def read(self):
        """
        Called by main.py.

        Returns the latest available frame.
        """

        with self.lock:

            if self.frame is None:
                return False, None

            return True, self.frame.copy()


    def release(self):
        """
        Stop camera thread and release ESP32 stream.
        """

        print("Stopping ESP32-CAM...")

        self.running = False

        # Release stream BEFORE waiting for thread.
        # This helps unblock camera.read().
        if self.camera is not None:

            try:
                self.camera.release()
            except Exception:
                pass


        # Wait briefly for background thread
        if hasattr(self, "thread") and self.thread.is_alive():

            self.thread.join(timeout=1)


        with self.lock:
            self.frame = None
            self.success = False


        print("ESP32-CAM stopped.")