import cv2
import time
import threading
import numpy as np
from services.camera import Camera
from services.speech import SpeechService
from agents.voice_agent import VoiceAgent
from agents.wake_word_agent import WakeWordAgent
from agents.face_recognition_agent import FaceRecognitionAgent
from agents.recognition_stabilizer import RecognitionStabilizer
from agents.emotion_agent import EmotionAgent
from agents.emotion_stabilizer import EmotionStabilizer
from agents.gesture_agent import GestureAgent
from database.face_database import FaceDatabase
from config import (
    FACE_MATCH_THRESHOLD,
    EMOTION_INTERVAL,
    EMOTION_HISTORY_SIZE
)
class SmartGlassesAI:
    def __init__(self):
        print("\n====================================")
        print("SMART GLASSES AI")
        print("====================================")
        # ==================================================
        # SERVICES
        # ==================================================
        self.camera = Camera()
        self.speech = SpeechService()
        self.voice = VoiceAgent()
        self.wake_word_agent = WakeWordAgent(
            wake_phrase="hey assistant"
        )
        # ==================================================
        # FACE RECOGNITION
        # ==================================================
        self.face_recognizer = FaceRecognitionAgent()
        self.recognition_stabilizer = RecognitionStabilizer(
            history_size=5,
            minimum_votes=3
        )
        # ==================================================
        # EMOTION
        # ==================================================
        self.emotion_agent = EmotionAgent()
        self.emotion_stabilizer = EmotionStabilizer(
            history_size=EMOTION_HISTORY_SIZE,
            minimum_samples=3
        )
        # ==================================================
        # GESTURE
        # ==================================================
        self.gesture_agent = GestureAgent()
        self.last_gesture = {}
        self.last_gesture_time = {}
        self.gesture_cooldown = 5.0
        # Wave display
        self.wave_display_name = None
        self.wave_display_time = 0
        self.wave_display_duration = 1.5
        # ==================================================
        # DATABASE
        # ==================================================
        self.database = FaceDatabase()
        self.known_people = []
        self.reload_database()
        # ==================================================
        # VOICE
        # ==================================================
        self.pending_command = None
        self.listening = False
        # ==================================================
        # PERSON PRESENCE
        # ==================================================
        self.last_seen = {}
        self.present_announced = set()
        self.leave_timeout = 3.0
        # ==================================================
        # EMOTION STATE
        # ==================================================
        self.cached_emotions = {}
        self.last_emotion_time = {}
        self.emotion_processing = set()
        self.emotion_announced = {}
        self.emotion_lock = threading.Lock()
        # ==================================================
        # FACE TRACKING
        # ==================================================
        self.tracks = {}
        self.next_track_id = 0
        self.track_distance_threshold = 140
        self.track_timeout = 2.0
        # ==================================================
        # SYSTEM
        # ==================================================
        self.running = True
        print("\nSystem ready.")
        print("V = Voice Command")
        print("Q = Quit")
        print("====================================\n")
    # ======================================================
    # DATABASE
    # ======================================================
    def reload_database(self):
        self.known_people = self.database.get_all_people()
        print(
            f"\nLoaded {len(self.known_people)} "
            f"known person(s)."
        )
        for person in self.known_people:
            print("-", person["name"])
    # ======================================================
    # FACE RECOGNITION
    # ======================================================
    def recognize_face(self, face):
        embedding = self.face_recognizer.get_embedding(face)
        best_name = "Unknown"
        best_similarity = -1.0
        for person in self.known_people:
            similarity = self.face_recognizer.compare(
                embedding,
                person["embedding"]
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = person["name"]
        if best_similarity >= FACE_MATCH_THRESHOLD:
            return best_name, best_similarity
        return "Unknown", best_similarity
    # ======================================================
    # FACE CENTER
    # ======================================================
    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return center_x, center_y
    # ======================================================
    # FACE TRACKING
    # ======================================================
    def assign_tracks(self, detections):
        current_time = time.time()
        expired_tracks = []
        for track_id, data in self.tracks.items():
            if (
                current_time - data["last_seen"]
                > self.track_timeout
            ):
                expired_tracks.append(track_id)
        for track_id in expired_tracks:
            del self.tracks[track_id]
            self.recognition_stabilizer.clear(
                track_id
            )
        assigned_tracks = set()
        results = []
        for detection in detections:
            bbox = detection["bbox"]
            center = self.get_center(
                bbox
            )
            best_track = None
            best_distance = float("inf")
            for track_id, data in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                old_center = data["center"]
                distance = (
                    (
                        center[0] - old_center[0]
                    ) ** 2
                    +
                    (
                        center[1] - old_center[1]
                    ) ** 2
                ) ** 0.5
                if (
                    distance < best_distance
                    and
                    distance < self.track_distance_threshold
                ):
                    best_distance = distance
                    best_track = track_id
            if best_track is None:
                best_track = self.next_track_id
                self.next_track_id += 1
            self.tracks[best_track] = {
                "center": center,
                "bbox": bbox,
                "last_seen": current_time
            }
            assigned_tracks.add(
                best_track
            )
            detection["track_id"] = best_track
            results.append(
                detection
            )
        return results
    # ======================================================
    # PERSON PRESENCE
    # ======================================================
    def handle_person_presence(
        self,
        name
    ):
        if name == "Unknown":
            return
        current_time = time.time()
        self.last_seen[name] = current_time
        if name not in self.present_announced:
            self.present_announced.add(
                name
            )
            message = (
                f"{name} is in front of you."
            )
            print(
                "\nPERSON ENTERED:",
                name
            )
            print(
                "SENDING TO TTS:",
                message
            )
            self.speech.speak(
                message,
                priority=0
            )
    # ======================================================
    # PERSON STILL PRESENT
    # ======================================================
    def is_person_present(
        self,
        name
    ):
        if name not in self.last_seen:
            return False
        elapsed = (
            time.time()
            - self.last_seen[name]
        )
        return elapsed <= self.leave_timeout
    # ======================================================
    # UPDATE PERSON PRESENCE
    # ======================================================
    def update_presence(self):
        current_time = time.time()
        people_to_remove = []
        for name in list(
            self.present_announced
        ):
            last_time = self.last_seen.get(
                name,
                0
            )
            if (
                current_time - last_time
                > self.leave_timeout
            ):
                people_to_remove.append(
                    name
                )
        for name in people_to_remove:
            print(
                f"{name} left camera view."
            )
            self.present_announced.discard(
                name
            )
            self.emotion_stabilizer.clear_person(
                name
            )
            self.last_gesture.pop(
                name,
                None
            )
            self.last_gesture_time.pop(
                name,
                None
            )
            with self.emotion_lock:
                self.cached_emotions.pop(
                    name,
                    None
                )
                self.last_emotion_time.pop(
                    name,
                    None
                )
                self.emotion_processing.discard(
                    name
                )
                self.emotion_announced.pop(
                    name,
                    None
                )
    # ======================================================
    # EMOTION WORKER
    # ======================================================
    def emotion_worker(
        self,
        name,
        face_crop
    ):
        try:
            raw_emotion = (
                self.emotion_agent.detect_emotion(
                    face_crop
                )
            )
            print(
                f"Raw emotion for {name}: "
                f"{raw_emotion}"
            )
            if not self.is_person_present(
                name
            ):
                print(
                    f"Discarding old emotion "
                    f"for {name}"
                )
                return
            if (
                raw_emotion
                and
                raw_emotion != "unknown"
            ):
                stable_emotion = (
                    self.emotion_stabilizer.update(
                        name,
                        raw_emotion
                    )
                )
                print(
                    f"Stable emotion for "
                    f"{name}: "
                    f"{stable_emotion}"
                )
                if (
                    stable_emotion
                    not in [
                        "unknown",
                        "analyzing"
                    ]
                ):
                    with self.emotion_lock:
                        self.cached_emotions[
                            name
                        ] = stable_emotion
        except Exception as error:
            print(
                "Emotion worker error:",
                error
            )
        finally:
            with self.emotion_lock:
                self.last_emotion_time[
                    name
                ] = time.time()
                self.emotion_processing.discard(
                    name
                )
    # ======================================================
    # UPDATE EMOTION
    # ======================================================
    def update_emotion(
        self,
        name,
        face_crop
    ):
        if name == "Unknown":
            return
        if (
            face_crop is None
            or
            face_crop.size == 0
        ):
            return
        current_time = time.time()
        with self.emotion_lock:
            previous_time = (
                self.last_emotion_time.get(
                    name,
                    0
                )
            )
            processing = (
                name
                in self.emotion_processing
            )
        if processing:
            return
        if (
            current_time - previous_time
            < EMOTION_INTERVAL
        ):
            return
        face_copy = face_crop.copy()
        with self.emotion_lock:
            if (
                name
                in self.emotion_processing
            ):
                return
            self.emotion_processing.add(
                name
            )
            self.last_emotion_time[
                name
            ] = current_time
        thread = threading.Thread(
            target=self.emotion_worker,
            args=(
                name,
                face_copy
            ),
            daemon=True
        )
        thread.start()
    # ======================================================
    # ANNOUNCE EMOTION
    # ======================================================
    def announce_emotion(
        self,
        name
    ):
        if name == "Unknown":
            return
        if not self.is_person_present(
            name
        ):
            return
        with self.emotion_lock:
            emotion = (
                self.cached_emotions.get(
                    name
                )
            )
        if (
            not emotion
            or
            emotion == "unknown"
            or
            emotion == "analyzing"
        ):
            return
        previous_emotion = (
            self.emotion_announced.get(
                name
            )
        )
        if previous_emotion == emotion:
            return
        if not self.is_person_present(
            name
        ):
            return
        self.emotion_announced[
            name
        ] = emotion
        message = (
            f"{name} looks {emotion}."
        )
        print(
            "SENDING EMOTION TO TTS:",
            message
        )
        self.speech.speak(
            message,
            priority=3
        )
    # ======================================================
    # GESTURE HANDLER
    # ======================================================
    def handle_gestures(
        self,
        frame,
        visible_known_people
    ):
        try:
            gestures = (
                self.gesture_agent.detect(
                    frame
                )
            )
        except Exception as error:
            print(
                "Gesture detection error:",
                error
            )
            return
        if not gestures:
            return
        # Do not announce gestures for unknown people
        if not visible_known_people:
            return
        # Current prototype associates detected gesture
        # with first known visible person.
        name = visible_known_people[0]
        current_time = time.time()
        for gesture in gestures:
            previous_gesture = (
                self.last_gesture.get(
                    name
                )
            )
            previous_time = (
                self.last_gesture_time.get(
                    name,
                    0
                )
            )
            if (
                previous_gesture == gesture
                and
                current_time - previous_time
                < self.gesture_cooldown
            ):
                continue
            self.last_gesture[
                name
            ] = gesture
            self.last_gesture_time[
                name
            ] = current_time
            if gesture == "waving":
                message = (
                    f"{name} is waving."
                )
                self.wave_display_name = name
                self.wave_display_time = (
                    current_time
                )
            elif gesture == "open palm":
                message = (
                    f"{name} is showing "
                    f"an open palm."
                )
            elif gesture == "thumbs up":
                message = (
                    f"{name} is showing "
                    f"thumbs up."
                )
            elif gesture == "thumbs down":
                message = (
                    f"{name} is showing "
                    f"thumbs down."
                )
            elif gesture == "victory":
                message = (
                    f"{name} is showing "
                    f"a victory sign."
                )
            elif gesture == "closed fist":
                message = (
                    f"{name} is showing "
                    f"a closed fist."
                )
            elif gesture == "pointing up":
                message = (
                    f"{name} is pointing up."
                )
            elif gesture == "i love you":
                message = (
                    f"{name} is showing "
                    f"the I love you gesture."
                )
            else:
                continue
            print(
                "\nGESTURE DETECTED:",
                name,
                "->",
                gesture
            )
            print(
                "SENDING GESTURE TO TTS:",
                message
            )
            self.speech.speak(
                message,
                priority=1
            )
    # ======================================================
    # WAKE WORD ACTIVATION
    # ======================================================
    def wake_word_activated(self):
        print("\nWake word activated.")
        if self.listening:
            return
        self.listen_for_command()

    # ======================================================
    # VOICE COMMAND LISTENER
    # ======================================================
    def listen_for_command(self):
        if self.listening:
            return
        self.listening = True
        try:
            self.speech.speak(
                "Listening for command.",
                wait=True,
                priority=0
            )
            command = (
                self.voice.listen()
            )
            if command:
                self.pending_command = command
                print(
                    "\nCommand received:",
                    command
                )
        except Exception as error:
            print(
                "Voice command error:",
                error
            )
        finally:
            self.listening = False
        # ======================================================
    # CONFIRM YES / NO
    # ======================================================

    def get_confirmation(self):

        yes_words = [
            "yes",
            "yeah",
            "yep",
            "yea",
            "yup",
            "correct",
            "right",
            "confirm",
            "confirmed",
            "okay",
            "ok",
            "sure",
            "that's correct",
            "that is correct",
            "name is correct",
            "the name is correct",
            "yes please",
            "yes that's right",
            "yes that is right"
        ]

        no_words = [
            "no",
            "nope",
            "nah",
            "wrong",
            "incorrect",
            "not correct",
            "that's wrong",
            "that is wrong",
            "name is wrong",
            "the name is wrong",
            "no that's wrong",
            "no that is wrong"
        ]

        for attempt in range(3):

            print(
                f"\nConfirmation attempt "
                f"{attempt + 1}/3"
            )

            self.speech.speak(
                "Please say yes or no.",
                wait=True,
                priority=0,
                allow_duplicate=True
            )

            answer = self.voice.listen()

            if not answer:
                print("No confirmation heard.")

                if attempt < 2:
                    self.speech.speak(
                        "I did not hear you. "
                        "Please say yes or no.",
                        wait=True,
                        priority=0,
                        allow_duplicate=True
                    )

                continue

            answer = answer.lower().strip()

            print(
                "Confirmation heard:",
                answer
            )

            # ----------------------------------------------
            # YES
            # ----------------------------------------------

            for word in yes_words:

                if word in answer:
                    print(
                        "Confirmation result: YES"
                    )
                    return True

            # ----------------------------------------------
            # NO
            # ----------------------------------------------

            for word in no_words:

                if word in answer:
                    print(
                        "Confirmation result: NO"
                    )
                    return False

            # ----------------------------------------------
            # COMMON SPEECH RECOGNITION ERRORS
            # ----------------------------------------------

            if answer in [
                "yess",
                "yeses",
                "yas",
                "yep",
                "yup"
            ]:
                print(
                    "Confirmation result: YES"
                )
                return True

            if answer in [
                "know",
                "noo",
                "noo",
                "nop"
            ]:
                print(
                    "Confirmation result: NO"
                )
                return False

            # ----------------------------------------------
            # NOT UNDERSTOOD
            # ----------------------------------------------

            print(
                "Could not understand "
                "yes or no."
            )

            if attempt < 2:
                self.speech.speak(
                    "I did not understand. "
                    "Please say yes or no.",
                    wait=True,
                    priority=0,
                    allow_duplicate=True
                )

        # --------------------------------------------------
        # ALL 3 ATTEMPTS FAILED
        # --------------------------------------------------

        print(
            "Confirmation failed after "
            "3 attempts."
        )

        self.speech.speak(
            "I could not understand your answer. "
            "Let's try again.",
            wait=True,
            priority=0,
            allow_duplicate=True
        )

        return None
    # ======================================================
    # ADD PERSON
    # ======================================================
    def add_person(self):
        print("\n====================================")
        print("STARTING VOICE ENROLLMENT")
        print("====================================")
        name = None
        # Ask for the name until the user confirms it.
        # No spelling or phonetic fallback.
        while name is None:
            self.speech.speak(
                "What is the person's name?",
                wait=True,
                priority=0
            )
            heard_name = self.voice.listen()
            if not heard_name:
                self.speech.speak(
                    "I could not understand the name. Please say the name again.",
                    wait=True,
                    priority=0
                )
                continue
            heard_name = heard_name.title().strip()
            print("Name heard:", heard_name)
            self.speech.speak(
                f"I heard {heard_name}. Is that correct?",
                wait=True,
                priority=0
            )
            confirmation = self.get_confirmation()
            if confirmation is True:
                name = heard_name
                print("Name confirmed:", name)
                break
            if confirmation is False:
                print("Name rejected:", heard_name)
                self.speech.speak(
                    "Okay. Please say the name again.",
                    wait=True,
                    priority=0
                )
                continue
            print("Confirmation not understood.")
            self.speech.speak(
                "I could not understand your answer. Please say the name again.",
                wait=True,
                priority=0
            )
        # ==================================================
        # NAME CONFIRMED
        # ==================================================
        print(
            "\n===================================="
        )
        print(
            "CONFIRMED NAME:",
            name
        )
        print(
            "===================================="
        )
        self.speech.speak(
            f"Okay. Adding {name}. "
            f"Please keep only {name} "
            f"in front of the camera.",
            wait=True,
            priority=0
        )
        # ==================================================
        # CAPTURE FACE SAMPLES
        # ==================================================
        embeddings = []
        last_capture = 0
        start_time = time.time()
        while len(embeddings) < 5:
            # ----------------------------------------------
            # TIMEOUT
            # ----------------------------------------------
            if (
                time.time() - start_time
                > 30
            ):
                self.speech.speak(
                    "Enrollment timed out.",
                    priority=0
                )
                return
            # ----------------------------------------------
            # CAMERA
            # ----------------------------------------------
            success, frame = (
                self.camera.read()
            )
            if not success:
                continue
            # ----------------------------------------------
            # FACE DETECTION
            # ----------------------------------------------
            faces = (
                self.face_recognizer.get_faces(
                    frame
                )
            )
            # ----------------------------------------------
            # DRAW BOX
            # ----------------------------------------------
            for face in faces:
                x1, y1, x2, y2 = (
                    face.bbox.astype(int)
                )
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )
            current_time = time.time()
            # ----------------------------------------------
            # AUTO CAPTURE
            # ----------------------------------------------
            if (
                len(faces) == 1
                and
                current_time - last_capture
                >= 0.7
            ):
                embedding = (
                    self.face_recognizer.get_embedding(
                        faces[0]
                    )
                )
                if embedding is not None:
                    embeddings.append(
                        embedding
                    )
                    last_capture = (
                        current_time
                    )
                    print(
                        f"Captured "
                        f"{len(embeddings)}/5"
                    )
            # ----------------------------------------------
            # ENROLLMENT DISPLAY
            # ----------------------------------------------
            cv2.putText(
                frame,
                f"Adding: {name}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"Samples: "
                f"{len(embeddings)}/5",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            if len(faces) == 0:
                status = (
                    "No face detected"
                )
            elif len(faces) > 1:
                status = (
                    "Keep only ONE face visible"
                )
            else:
                status = (
                    "Capturing automatically..."
                )
            cv2.putText(
                frame,
                status,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.imshow(
                "Smart Glasses AI",
                frame
            )
            key = (
                cv2.waitKey(1)
                & 0xFF
            )
            if key == ord("q"):
                self.speech.speak(
                    "Enrollment cancelled.",
                    priority=0
                )
                return
        # ==================================================
        # AVERAGE THE 5 FACE EMBEDDINGS
        # ==================================================
        final_embedding = (
            np.mean(
                embeddings,
                axis=0
            )
        )
        norm = np.linalg.norm(
            final_embedding
        )
        if norm == 0:
            self.speech.speak(
                "Face enrollment failed.",
                priority=0
            )
            return
        final_embedding = (
            final_embedding / norm
        )
        # ==================================================
        # SAVE PERSON
        # ==================================================
        self.database.add_person(
            name,
            final_embedding
        )
        self.reload_database()
        # ==================================================
        # CLEAR OLD STATE
        # ==================================================
        self.present_announced.discard(
            name
        )
        self.last_seen.pop(
            name,
            None
        )
        self.emotion_stabilizer.clear_person(
            name
        )
        self.last_gesture.pop(
            name,
            None
        )
        self.last_gesture_time.pop(
            name,
            None
        )
        with self.emotion_lock:
            self.cached_emotions.pop(
                name,
                None
            )
            self.last_emotion_time.pop(
                name,
                None
            )
            self.emotion_processing.discard(
                name
            )
            self.emotion_announced.pop(
                name,
                None
            )
        # Clear recognition tracking so an old track
        # cannot affect the newly enrolled identity.
        self.tracks.clear()
        self.recognition_stabilizer.clear_all()
        # ==================================================
        # SUCCESS
        # ==================================================
        print(
            "\n===================================="
        )
        print(
            f"{name} SUCCESSFULLY ADDED"
        )
        print(
            "====================================\n"
        )
        self.speech.speak(
            f"{name} has been added successfully.",
            priority=0
        )
    # ======================================================
    # PROCESS VOICE COMMAND
    # ======================================================
    def process_command(self):
        if not self.pending_command:
            return
        command = (
            self.pending_command
            .lower()
            .strip()
        )
        self.pending_command = None
        print(
            "\nHeard command:",
            command
        )
        add_commands = [
            "add person",
            "add a person",
            "add new person",
            "add someone",
            # Common speech recognition errors
            "at person",
            "ad person",
            "add people",
            "bad person"
        ]
        if any(
            phrase in command
            for phrase in add_commands
        ):
            self.add_person()
        elif (
            "who is there" in command
            or
            "who is this" in command
            or
            "who is in front" in command
            or
            "who is in front of me" in command
        ):
            visible_people = []
            for name in self.last_seen:
                if self.is_person_present(
                    name
                ):
                    visible_people.append(
                        name
                    )
            if visible_people:
                names = ", ".join(
                    visible_people
                )
                self.speech.speak(
                    f"{names} is in front of you.",
                    priority=0
                )
            else:
                self.speech.speak(
                    "No known person is detected.",
                    priority=0
                )
        else:
            self.speech.speak(
                "Command not recognized.",
                priority=0
            )

        # Voice command is completely finished.
        # Resume listening for "Hey Assistant".
        self.wake_word_agent.resume()

    # ======================================================
    # MAIN LOOP
    # ======================================================
    def run(self):
        self.speech.speak(
            "Smart glasses assistant started.",
            priority=0
        )

        # Start hands-free wake-word listening.
        self.wake_word_agent.start(
            self.wake_word_activated
        )

        while self.running:
            success, frame = (
                self.camera.read()
            )
            if not success:
                print(
                    "Camera error."
                )
                continue
            height, width = (
                frame.shape[:2]
            )
            # ==================================================
            # FACE DETECTION
            # ==================================================
            insight_faces = (
                self.face_recognizer.get_faces(
                    frame
                )
            )
            detections = []
            visible_known_people = []
            for face in insight_faces:
                x1, y1, x2, y2 = (
                    face.bbox.astype(
                        int
                    )
                )
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(
                    width,
                    x2
                )
                y2 = min(
                    height,
                    y2
                )
                detections.append(
                    {
                        "face": face,
                        "bbox": (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    }
                )
            # ==================================================
            # TRACK FACES
            # ==================================================
            detections = (
                self.assign_tracks(
                    detections
                )
            )
            # ==================================================
            # PROCESS EACH FACE
            # ==================================================
            for detection in detections:
                face = detection["face"]
                track_id = detection[
                    "track_id"
                ]
                x1, y1, x2, y2 = (
                    detection["bbox"]
                )
                raw_name, similarity = (
                    self.recognize_face(
                        face
                    )
                )
                name = (
                    self.recognition_stabilizer.update(
                        track_id,
                        raw_name
                    )
                )
                print(
                    f"Track {track_id} | "
                    f"Raw: {raw_name} | "
                    f"Stable: {name} | "
                    f"Similarity: "
                    f"{similarity:.3f}"
                )
                face_crop = frame[
                    y1:y2,
                    x1:x2
                ]
                # ==================================================
                # KNOWN PERSON
                # ==================================================
                if name != "Unknown":
                    if (
                        name
                        not in visible_known_people
                    ):
                        visible_known_people.append(
                            name
                        )
                    self.handle_person_presence(
                        name
                    )
                    self.update_emotion(
                        name,
                        face_crop
                    )
                    self.announce_emotion(
                        name
                    )
                    with self.emotion_lock:
                        emotion = (
                            self.cached_emotions.get(
                                name,
                                "analyzing"
                            )
                        )
                    color = (
                        0,
                        255,
                        0
                    )
                # ==================================================
                # UNKNOWN PERSON
                # ==================================================
                else:
                    emotion = "unknown"
                    color = (
                        0,
                        0,
                        255
                    )
                # ==================================================
                # DRAW FACE INFORMATION
                # ==================================================
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )
                cv2.putText(
                    frame,
                    name,
                    (
                        x1,
                        max(
                            25,
                            y1 - 60
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                cv2.putText(
                    frame,
                    f"Emotion: {emotion}",
                    (
                        x1,
                        max(
                            45,
                            y1 - 35
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )
                cv2.putText(
                    frame,
                    f"Similarity: "
                    f"{similarity:.3f}",
                    (
                        x1,
                        max(
                            65,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (
                        x1,
                        y2 + 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )
            # ==================================================
            # GESTURES
            # ==================================================
            self.handle_gestures(
                frame,
                visible_known_people
            )
            # ==================================================
            # WAVE DISPLAY
            # ==================================================
            if (
                self.wave_display_name is not None
                and
                time.time() - self.wave_display_time
                < self.wave_display_duration
            ):
                cv2.putText(
                    frame,
                    f"{self.wave_display_name} - WAVING",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                self.wave_display_name = None
            # ==================================================
            # UPDATE PRESENCE
            # ==================================================
            self.update_presence()
            # ==================================================
            # UI
            # ==================================================
            cv2.putText(
                frame,
                f"Faces: {len(detections)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                "V = Voice Command | Q = Quit",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.imshow(
                "Smart Glasses AI",
                frame
            )
            # ==================================================
            # PROCESS VOICE COMMAND
            # ==================================================
            self.process_command()
            # ==================================================
            # KEYBOARD
            # ==================================================
            key = (
                cv2.waitKey(1)
                & 0xFF
            )
            if key == ord("v"):
                if not self.listening:
                    thread = threading.Thread(
                        target=self.listen_for_command,
                        daemon=True
                    )
                    thread.start()
            elif key == ord("q"):
                self.running = False
        self.cleanup()
    # ======================================================
    # CLEANUP
    # ======================================================
    def cleanup(self):
        print(
            "\nClosing Smart Glasses AI..."
        )
        self.running = False
        self.wake_word_agent.stop()
        self.camera.release()
        self.database.close()
        self.gesture_agent.close()
        self.speech.stop()
        cv2.destroyAllWindows()
        print(
            "System closed."
        )
# ==========================================================
# START
# ==========================================================
if __name__ == "__main__":
    app = SmartGlassesAI()
    app.run()