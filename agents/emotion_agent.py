
import tensorflow.keras
from deepface import DeepFace

from deepface import DeepFace


class EmotionAgent:

    def __init__(self):

        print("Emotion agent ready.")

        self.valid_emotions = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral"
        ]

        # Minimum confidence required
        # before accepting an emotion
        self.confidence_threshold = 45.0

    # ==================================================
    # DETECT EMOTION
    # ==================================================

    def detect_emotion(self, face_image):

        try:

            result = DeepFace.analyze(
                img_path=face_image,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip"
            )

            if isinstance(result, list):
                result = result[0]

            emotion_scores = result.get(
                "emotion",
                {}
            )

            if not emotion_scores:

                print(
                    "No emotion scores returned."
                )

                return "unknown"

            # ==========================================
            # PRINT ALL 7 PROBABILITIES
            # ==========================================

            print("\nEmotion probabilities:")

            for emotion_name in self.valid_emotions:

                score = emotion_scores.get(
                    emotion_name,
                    0
                )

                print(
                    f"{emotion_name:10} : "
                    f"{score:.2f}%"
                )

            # ==========================================
            # FIND HIGHEST PROBABILITY
            # ==========================================

            dominant_emotion = max(
                emotion_scores,
                key=emotion_scores.get
            )

            confidence = emotion_scores[
                dominant_emotion
            ]

            dominant_emotion = (
                dominant_emotion
                .lower()
                .strip()
            )

            print(
                f"\nTop emotion: "
                f"{dominant_emotion}"
            )

            print(
                f"Confidence: "
                f"{confidence:.2f}%"
            )

            # ==========================================
            # VALIDATE EMOTION
            # ==========================================

            if (
                dominant_emotion
                not in self.valid_emotions
            ):

                return "unknown"

            # ==========================================
            # CONFIDENCE FILTER
            # ==========================================

            if (
                confidence
                < self.confidence_threshold
            ):

                print(
                    "Emotion confidence too low."
                )

                return "unknown"

            return dominant_emotion

        except Exception as error:

            print(
                "Emotion detection error:",
                error
            )

            return "unknown"