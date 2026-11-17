import cv2
import numpy as np

from services.camera import Camera
from services.speech import SpeechService
from agents.voice_agent import VoiceAgent
from agents.face_recognition_agent import FaceRecognitionAgent
from database.face_database import FaceDatabase


class EnrollmentService:

    def __init__(self):
        self.camera = Camera()
        self.speech = SpeechService()
        self.voice = VoiceAgent()
        self.recognizer = FaceRecognitionAgent()
        self.database = FaceDatabase()

    def enroll_by_voice(self):

        self.speech.speak("What is the person's name?")

        name = self.voice.listen()

        if not name:
            self.speech.speak("I could not understand the name.")
            return

        name = name.title()

        print(f"Name detected: {name}")

        self.speech.speak(
            f"Please keep {name} in front of the camera."
        )

        embeddings = []

        print("\nCapturing face...")
        print("Keep exactly one face visible.")
        print("SPACE = capture samples")
        print("Q = cancel\n")

        while True:

            success, frame = self.camera.read()

            if not success:
                print("Camera error.")
                break

            faces = self.recognizer.get_faces(frame)

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Samples: {len(embeddings)}/5",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "SPACE = Capture | Q = Cancel",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Voice Enrollment",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                self.speech.speak(
                    "Enrollment cancelled."
                )
                break

            if key == 32:

                if len(faces) != 1:

                    print("Exactly one face must be visible.")

                    self.speech.speak(
                        "Please keep only one face in front of the camera."
                    )

                    continue

                embedding = self.recognizer.get_embedding(
                    faces[0]
                )

                embeddings.append(embedding)

                print(
                    f"Captured sample {len(embeddings)}"
                )

                if len(embeddings) >= 5:

                    final_embedding = np.mean(
                        embeddings,
                        axis=0
                    )

                    final_embedding = (
                        final_embedding
                        / np.linalg.norm(final_embedding)
                    )

                    self.database.add_person(
                        name,
                        final_embedding
                    )

                    print(
                        f"{name} added successfully."
                    )

                    self.speech.speak(
                        f"{name} has been added."
                    )

                    break

        self.camera.release()
        self.database.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    enrollment = EnrollmentService()

    enrollment.enroll_by_voice()