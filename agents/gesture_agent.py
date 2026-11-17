import os
import cv2
import time
from collections import deque

import mediapipe as mp


class GestureAgent:

    def __init__(self):

        print("Loading gesture recognizer...")

        # ==================================================
        # MODEL
        # ==================================================

        model_path = os.path.join(
            "models",
            "gesture_recognizer.task"
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Gesture model not found: {model_path}"
            )

        BaseOptions = mp.tasks.BaseOptions

        GestureRecognizer = (
            mp.tasks.vision.GestureRecognizer
        )

        GestureRecognizerOptions = (
            mp.tasks.vision.GestureRecognizerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )

        options = GestureRecognizerOptions(

            base_options=BaseOptions(
                model_asset_path=model_path
            ),

            running_mode=RunningMode.IMAGE,

            num_hands=2
        )

        self.recognizer = (
            GestureRecognizer.create_from_options(
                options
            )
        )

        # ==================================================
        # WAVE DETECTION
        # ==================================================

        self.position_history = deque(
            maxlen=20
        )

        self.last_open_palm_time = 0

        self.last_hand_time = 0

        self.last_wave_time = 0

        self.wave_cooldown = 2.5

        self.wave_state = 0

        self.wave_start_x = None

        self.wave_distance = 0.045

        # ==================================================
        # FALLBACK STABILIZATION
        # ==================================================

        # MediaPipe must return None for multiple frames
        # before fallback is allowed.

        self.none_frame_count = 0

        self.none_frames_required = 2

        # Candidate fallback gesture
        self.fallback_candidate = None

        # Number of consecutive frames with same fallback
        self.fallback_count = 0

        # Require 3 matching frames
        self.fallback_frames_required = 3

        print("Gesture agent ready.")

    # ======================================================
    # GET PALM POSITION
    # ======================================================

    def _get_palm_x(self, result):

        if not result.hand_landmarks:
            return None

        landmarks = result.hand_landmarks[0]

        if not landmarks:
            return None

        wrist = landmarks[0]
        index_base = landmarks[5]
        middle_base = landmarks[9]
        pinky_base = landmarks[17]

        palm_x = (
            wrist.x
            + index_base.x
            + middle_base.x
            + pinky_base.x
        ) / 4.0

        return palm_x

    # ======================================================
    # RESET FALLBACK
    # ======================================================

    def _reset_fallback(self):

        self.none_frame_count = 0

        self.fallback_candidate = None

        self.fallback_count = 0

    # ======================================================
    # FINGER EXTENDED
    # ======================================================

    def _finger_extended(
        self,
        landmarks,
        tip,
        pip
    ):

        return (
            landmarks[tip].y
            <
            landmarks[pip].y
        )

    # ======================================================
    # LANDMARK FALLBACK
    # ======================================================

    def _detect_landmark_fallback(
        self,
        landmarks
    ):

        if not landmarks:

            return None

        if len(landmarks) < 21:

            return None

        # --------------------------------------------------
        # FOUR MAIN FINGERS
        # --------------------------------------------------

        index_extended = (
            self._finger_extended(
                landmarks,
                8,
                6
            )
        )

        middle_extended = (
            self._finger_extended(
                landmarks,
                12,
                10
            )
        )

        ring_extended = (
            self._finger_extended(
                landmarks,
                16,
                14
            )
        )

        pinky_extended = (
            self._finger_extended(
                landmarks,
                20,
                18
            )
        )

        # ==================================================
        # VICTORY
        # ==================================================
        #
        # Index  = extended
        # Middle = extended
        # Ring   = folded
        # Pinky  = folded
        # Index and middle must also be separated.
        # ==================================================

        if (
            index_extended
            and
            middle_extended
            and
            not ring_extended
            and
            not pinky_extended
        ):

            separation = abs(
                landmarks[8].x
                - landmarks[12].x
            )

            if separation > 0.025:

                return "victory"

        # ==================================================
        # CLOSED FIST
        # ==================================================
        #
        # All four fingers must be folded.
        #
        # IMPORTANT:
        # We additionally check the thumb so that
        # Thumb_Up is not classified as Closed Fist.
        # ==================================================

        all_four_folded = (

            not index_extended
            and
            not middle_extended
            and
            not ring_extended
            and
            not pinky_extended
        )

        if all_four_folded:

            wrist = landmarks[0]

            thumb_tip = landmarks[4]

            index_mcp = landmarks[5]

            # ------------------------------------------------
            # THUMB-UP PROTECTION
            # ------------------------------------------------
            #
            # In a thumbs-up pose, the thumb tip is usually
            # significantly above the index MCP and wrist.
            # ------------------------------------------------

            thumb_up = (

                thumb_tip.y
                <
                index_mcp.y - 0.035

                and

                thumb_tip.y
                <
                wrist.y - 0.08
            )

            if thumb_up:

                # Do NOT call this a fist.
                return None

            # ------------------------------------------------
            # ADDITIONAL FIST COMPACTNESS CHECK
            # ------------------------------------------------

            folded_count = 0

            finger_pairs = [
                (8, 5),
                (12, 9),
                (16, 13),
                (20, 17)
            ]

            for tip, mcp in finger_pairs:

                # For a folded finger, the fingertip should
                # stay relatively close to the palm region.

                tip_to_wrist = (
                    (
                        landmarks[tip].x
                        - wrist.x
                    ) ** 2
                    +
                    (
                        landmarks[tip].y
                        - wrist.y
                    ) ** 2
                ) ** 0.5

                mcp_to_wrist = (
                    (
                        landmarks[mcp].x
                        - wrist.x
                    ) ** 2
                    +
                    (
                        landmarks[mcp].y
                        - wrist.y
                    ) ** 2
                ) ** 0.5

                if (
                    tip_to_wrist
                    <
                    mcp_to_wrist * 1.55
                ):

                    folded_count += 1

            if folded_count >= 3:

                return "closed fist"

        return None

    # ======================================================
    # STABILIZE FALLBACK
    # ======================================================

    def _stabilize_fallback(
        self,
        gesture
    ):

        if gesture is None:

            self.fallback_candidate = None

            self.fallback_count = 0

            return None

        # Same candidate again
        if gesture == self.fallback_candidate:

            self.fallback_count += 1

        else:

            # New candidate
            self.fallback_candidate = gesture

            self.fallback_count = 1

        print(
            f"Fallback candidate: "
            f"{gesture} "
            f"({self.fallback_count}/"
            f"{self.fallback_frames_required})"
        )

        # Require consecutive frames
        if (
            self.fallback_count
            >= self.fallback_frames_required
        ):

            print(
                "STABLE FALLBACK:",
                gesture
            )

            return gesture

        return None

    # ======================================================
    # RESET WAVE
    # ======================================================

    def _reset_wave(self):

        self.wave_state = 0

        self.wave_start_x = None

        self.position_history.clear()

    # ======================================================
    # WAVE STATE MACHINE
    # ======================================================

    def _update_wave_state(
        self,
        palm_x
    ):

        if palm_x is None:

            return False

        self.position_history.append(
            palm_x
        )

        # --------------------------------------------------
        # STATE 0
        # --------------------------------------------------

        if self.wave_state == 0:

            self.wave_start_x = palm_x

            self.wave_state = 1

            return False

        # --------------------------------------------------
        # STATE 1
        # --------------------------------------------------

        if self.wave_state == 1:

            difference = (
                palm_x
                - self.wave_start_x
            )

            if difference > self.wave_distance:

                print(
                    "Wave movement: RIGHT"
                )

                self.wave_start_x = palm_x

                self.wave_state = 2

            elif difference < -self.wave_distance:

                print(
                    "Wave movement: LEFT"
                )

                self.wave_start_x = palm_x

                self.wave_state = 3

            return False

        # --------------------------------------------------
        # STATE 2
        # RIGHT -> wait LEFT
        # --------------------------------------------------

        if self.wave_state == 2:

            difference = (
                palm_x
                - self.wave_start_x
            )

            if difference < -self.wave_distance:

                print(
                    "Wave movement: LEFT"
                )

                self.wave_start_x = palm_x

                self.wave_state = 4

            return False

        # --------------------------------------------------
        # STATE 3
        # LEFT -> wait RIGHT
        # --------------------------------------------------

        if self.wave_state == 3:

            difference = (
                palm_x
                - self.wave_start_x
            )

            if difference > self.wave_distance:

                print(
                    "Wave movement: RIGHT"
                )

                self.wave_start_x = palm_x

                self.wave_state = 5

            return False

        # --------------------------------------------------
        # STATE 4
        # RIGHT -> LEFT -> RIGHT
        # --------------------------------------------------

        if self.wave_state == 4:

            difference = (
                palm_x
                - self.wave_start_x
            )

            if difference > self.wave_distance:

                print(
                    "Wave movement: RIGHT"
                )

                return True

            return False

        # --------------------------------------------------
        # STATE 5
        # LEFT -> RIGHT -> LEFT
        # --------------------------------------------------

        if self.wave_state == 5:

            difference = (
                palm_x
                - self.wave_start_x
            )

            if difference < -self.wave_distance:

                print(
                    "Wave movement: LEFT"
                )

                return True

            return False

        return False

    # ======================================================
    # MAIN DETECTION
    # ======================================================

    def detect(
        self,
        frame
    ):

        # --------------------------------------------------
        # BGR -> RGB
        # --------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb_frame
        )

        # --------------------------------------------------
        # MEDIAPIPE
        # --------------------------------------------------

        result = (
            self.recognizer.recognize(
                mp_image
            )
        )

        gestures = []

        current_time = time.time()

        # ==================================================
        # HAND POSITION
        # ==================================================

        palm_x = (
            self._get_palm_x(
                result
            )
        )

        if palm_x is not None:

            self.last_hand_time = (
                current_time
            )

        else:

            if (
                current_time
                - self.last_hand_time
                > 1.5
            ):

                self._reset_wave()

                self._reset_fallback()

        # ==================================================
        # NORMAL MEDIAPIPE GESTURE
        # ==================================================

        valid_mediapipe_gesture = False

        if result.gestures:

            for hand_index, hand_gestures in enumerate(
                result.gestures
            ):

                if not hand_gestures:

                    continue

                best_gesture = (
                    hand_gestures[0]
                )

                gesture_name = (
                    best_gesture.category_name
                )

                confidence = (
                    best_gesture.score
                )

                print(
                    f"Gesture: "
                    f"{gesture_name} | "
                    f"Confidence: "
                    f"{confidence:.2f}"
                )

                # ==================================================
                # REAL MEDIAPIPE GESTURE
                # ==================================================
                #
                # This ALWAYS has priority over our fallback.
                # ==================================================

                if (
                    gesture_name != "None"
                    and
                    confidence >= 0.55
                ):

                    valid_mediapipe_gesture = True

                    # Reset fallback immediately
                    self._reset_fallback()

                    gesture_map = {

                        "Open_Palm":
                            "open palm",

                        "Thumb_Up":
                            "thumbs up",

                        "Thumb_Down":
                            "thumbs down",

                        "Victory":
                            "victory",

                        "Pointing_Up":
                            "pointing up",

                        "Closed_Fist":
                            "closed fist",

                        "ILoveYou":
                            "i love you"
                    }

                    readable_name = (
                        gesture_map.get(
                            gesture_name
                        )
                    )

                    if readable_name:

                        gestures.append(
                            readable_name
                        )

                    # ------------------------------------------
                    # OPEN PALM FOR WAVE
                    # ------------------------------------------

                    if (
                        gesture_name
                        == "Open_Palm"
                    ):

                        self.last_open_palm_time = (
                            current_time
                        )

        # ==================================================
        # FALLBACK
        # ==================================================
        #
        # ONLY run this when MediaPipe does NOT have a
        # confident real gesture.
        # ==================================================

        if not valid_mediapipe_gesture:

            # ----------------------------------------------
            # CHECK IF MEDIAPIPE IS RETURNING NONE
            # ----------------------------------------------

            mediapipe_none = True

            if result.gestures:

                for hand_gestures in result.gestures:

                    if not hand_gestures:

                        continue

                    top = hand_gestures[0]

                    if (
                        top.category_name
                        != "None"
                        and
                        top.score >= 0.55
                    ):

                        mediapipe_none = False

                        break

            if mediapipe_none:

                self.none_frame_count += 1

            else:

                self.none_frame_count = 0

            # ----------------------------------------------
            # DON'T FALLBACK IMMEDIATELY
            # ----------------------------------------------

            if (
                self.none_frame_count
                >= self.none_frames_required
                and
                result.hand_landmarks
            ):

                for landmarks in (
                    result.hand_landmarks
                ):

                    candidate = (
                        self._detect_landmark_fallback(
                            landmarks
                        )
                    )

                    stable_gesture = (
                        self._stabilize_fallback(
                            candidate
                        )
                    )

                    if stable_gesture:

                        gestures.append(
                            stable_gesture
                        )

                        break

        # ==================================================
        # WAVE DETECTION
        # ==================================================

        palm_recently_open = (

            current_time
            - self.last_open_palm_time
            < 2.5
        )

        if (
            palm_recently_open
            and
            palm_x is not None
        ):

            wave_detected = (
                self._update_wave_state(
                    palm_x
                )
            )

            if wave_detected:

                if (
                    current_time
                    - self.last_wave_time
                    > self.wave_cooldown
                ):

                    self.last_wave_time = (
                        current_time
                    )

                    print(
                        "\n===================="
                    )

                    print(
                        "WAVE DETECTED"
                    )

                    print(
                        "====================\n"
                    )

                    # Waving is more meaningful than
                    # simply saying open palm.

                    gestures = [

                        gesture

                        for gesture
                        in gestures

                        if gesture
                        != "open palm"
                    ]

                    gestures.append(
                        "waving"
                    )

                    self._reset_wave()

        else:

            if (
                current_time
                - self.last_open_palm_time
                > 2.5
            ):

                self._reset_wave()

        # ==================================================
        # REMOVE DUPLICATES
        # ==================================================

        unique_gestures = []

        for gesture in gestures:

            if gesture not in unique_gestures:

                unique_gestures.append(
                    gesture
                )

        return unique_gestures

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):

        if self.recognizer:

            self.recognizer.close()

        print(
            "Gesture agent closed."
        )