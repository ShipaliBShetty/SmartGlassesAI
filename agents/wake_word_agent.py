import threading
import time
import speech_recognition as sr


class WakeWordAgent:

    def __init__(
        self,
        wake_phrase="hey assistant"
    ):

        print("Wake word agent ready.")

        self.wake_phrase = wake_phrase.lower()

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        self.running = False
        self.paused = False

        self.thread = None
        self.callback = None

        self.last_trigger_time = 0
        self.cooldown = 3.0

        # Speech recognition settings
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.5
        self.recognizer.phrase_threshold = 0.2
        self.recognizer.non_speaking_duration = 0.3

    # ======================================================
    # START
    # ======================================================

    def start(self, callback):

        if self.running:
            return

        self.callback = callback
        self.running = True

        self.thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )

        self.thread.start()

        print("\nWake word listener started.")
        print('Say "Hey Assistant" for voice commands.')

    # ======================================================
    # LISTEN LOOP
    # ======================================================

    def _listen_loop(self):

        # --------------------------------------------------
        # INITIAL MICROPHONE CALIBRATION
        # --------------------------------------------------

        try:

            with self.microphone as source:

                print(
                    "Calibrating wake word microphone..."
                )

                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.5
                )

        except Exception as error:

            print(
                "Wake word calibration error:",
                error
            )

        # --------------------------------------------------
        # CONTINUOUS LISTENING
        # --------------------------------------------------

        while self.running:

            # If main voice command is using microphone,
            # wake-word listener must wait.
            if self.paused:

                time.sleep(0.1)
                continue

            try:

                with self.microphone as source:

                    try:

                        audio = self.recognizer.listen(
                            source,
                            timeout=1,
                            phrase_time_limit=3
                        )

                    except sr.WaitTimeoutError:

                        continue

                # ------------------------------------------
                # SPEECH TO TEXT
                # ------------------------------------------

                try:

                    text = (
                        self.recognizer
                        .recognize_google(
                            audio,
                            language="en-IN"
                        )
                        .lower()
                        .strip()
                    )

                except sr.UnknownValueError:

                    continue

                except sr.RequestError as error:

                    print(
                        "Wake word speech service error:",
                        error
                    )

                    time.sleep(1)
                    continue

                print(
                    f"Wake listener heard: {text}"
                )

                # ------------------------------------------
                # CHECK WAKE PHRASE
                # ------------------------------------------

                wake_detected = (
                    self.wake_phrase in text
                    or "hey assistant" in text
                    or text == "assistant"
                )

                if not wake_detected:
                    continue

                current_time = time.time()

                # ------------------------------------------
                # COOLDOWN
                # ------------------------------------------

                if (
                    current_time
                    - self.last_trigger_time
                    < self.cooldown
                ):

                    continue

                self.last_trigger_time = current_time

                print(
                    "\n===================================="
                )

                print(
                    "HEY ASSISTANT DETECTED"
                )

                print(
                    "===================================="
                )

                # Pause immediately so this listener
                # does not fight with VoiceAgent.
                self.paused = True

                # ------------------------------------------
                # RUN MAIN CALLBACK
                # ------------------------------------------

                if self.callback:

                    try:

                        self.callback()

                    except Exception as error:

                        print(
                            "Wake word callback error:",
                            error
                        )

            except Exception as error:

                if self.running:

                    print(
                        "Wake word listener error:",
                        error
                    )

                    time.sleep(0.5)

    # ======================================================
    # PAUSE
    # ======================================================

    def pause(self):

        self.paused = True

    # ======================================================
    # RESUME
    # ======================================================

    def resume(self):

        # Small delay prevents the wake listener
        # from hearing the final TTS/audio.
        time.sleep(0.5)

        self.paused = False

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        print(
            "Stopping wake word listener..."
        )

        self.running = False
        self.paused = False

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):

            self.thread.join(
                timeout=2
            )

        print(
            "Wake word listener stopped."
        )