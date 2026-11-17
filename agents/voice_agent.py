import speech_recognition as sr
import winsound
import time


class VoiceAgent:

    def __init__(self):

        # ==================================================
        # SPEECH RECOGNIZER
        # ==================================================

        self.recognizer = sr.Recognizer()

        # ==================================================
        # MICROPHONE
        # ==================================================

        self.microphone = sr.Microphone()

        # ==================================================
        # RECOGNITION SETTINGS
        # ==================================================

        self.recognizer.dynamic_energy_threshold = True

        self.recognizer.energy_threshold = 250

        # How long silence is allowed before considering
        # the spoken phrase complete
        self.recognizer.pause_threshold = 0.6

        # Helps with short phrases such as:
        # "name correct" and "name wrong"
        self.recognizer.phrase_threshold = 0.15

        self.recognizer.non_speaking_duration = 0.3

        print("Voice agent ready.")

    # ======================================================
    # SPEAK NOW BEEP
    # ======================================================

    def play_listening_beep(self):

        try:

            # Frequency = 1000 Hz
            # Duration = 180 milliseconds
            winsound.Beep(
                1000,
                180
            )

        except Exception as error:

            print(
                "Beep error:",
                error
            )

    # ======================================================
    # LISTEN
    # ======================================================

    def listen(self):

        try:

            # ==================================================
            # OPEN MICROPHONE
            # ==================================================

            with self.microphone as source:

                print("\nPreparing microphone...")

                # ----------------------------------------------
                # CALIBRATE FIRST
                # ----------------------------------------------
                #
                # IMPORTANT:
                # The user should NOT speak during this part.
                #
                # The beep is played only AFTER calibration.
                # Therefore the blind user simply waits for
                # the beep before speaking.
                # ----------------------------------------------

                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.3
                )

                # Small pause after calibration
                time.sleep(0.1)

                # ----------------------------------------------
                # AUDIO CUE
                # ----------------------------------------------

                self.play_listening_beep()

                print("Listening... Speak now.")

                # ----------------------------------------------
                # RECORD SPEECH
                # ----------------------------------------------

                try:

                    audio = self.recognizer.listen(
                        source,
                        timeout=7,
                        phrase_time_limit=6
                    )

                except sr.WaitTimeoutError:

                    print(
                        "No speech detected."
                    )

                    return ""

            # ==================================================
            # SPEECH TO TEXT
            # ==================================================

            try:

                print(
                    "Recognizing..."
                )

                text = (
                    self.recognizer.recognize_google(
                        audio,
                        language="en-IN"
                    )
                )

                text = (
                    text
                    .lower()
                    .strip()
                )

                print(
                    f"You said: {text}"
                )

                return text

            # ==================================================
            # SPEECH NOT UNDERSTOOD
            # ==================================================

            except sr.UnknownValueError:

                print(
                    "Could not understand speech."
                )

                return ""

            # ==================================================
            # GOOGLE SPEECH SERVICE ERROR
            # ==================================================

            except sr.RequestError as error:

                print(
                    "Speech recognition "
                    "service error:",
                    error
                )

                return ""

        # ======================================================
        # MICROPHONE ERROR
        # ======================================================

        except Exception as error:

            print(
                "Microphone error:",
                error
            )

            return ""