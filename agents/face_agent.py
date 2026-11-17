import cv2
import mediapipe as mp


class FaceAgent:

    def __init__(self):

        self.mp_face_detection = mp.solutions.face_detection

        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def detect(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.detector.process(rgb_frame)

        faces = []

        if results.detections:

            height, width, _ = frame.shape

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)

                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Keep coordinates inside image
                x = max(0, x)
                y = max(0, y)

                w = min(w, width - x)
                h = min(h, height - y)

                faces.append({
                    "box": (x, y, w, h),
                    "confidence": detection.score[0]
                })

        return faces

    def close(self):

        self.detector.close()