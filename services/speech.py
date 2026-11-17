import queue
import threading
import time
import pythoncom
import pyttsx3


class SpeechService:

    def __init__(self):

        self.queue = queue.PriorityQueue()

        self.running = True

        self.counter = 0

        self.last_text = ""
        self.last_text_time = 0

        self.worker = threading.Thread(
            target=self._speech_worker,
            daemon=True
        )

        self.worker.start()

        print("Speech service ready.")

    # ======================================================
    # SPEECH WORKER
    # ======================================================

    def _speech_worker(self):

        pythoncom.CoInitialize()

        try:

            while self.running:

                item = self.queue.get()

                if item is None:
                    continue

                priority, counter, text, event = item

                if text is None:

                    self.queue.task_done()
                    break

                engine = None

                try:

                    print(
                        f"TTS >>> {text}"
                    )

                    engine = pyttsx3.init()

                    engine.setProperty(
                        "rate",
                        185
                    )

                    engine.setProperty(
                        "volume",
                        1.0
                    )

                    engine.say(
                        text
                    )

                    engine.runAndWait()

                    engine.stop()

                    print(
                        f"TTS DONE >>> {text}"
                    )

                except Exception as error:

                    print(
                        "Speech error:",
                        error
                    )

                finally:

                    if engine is not None:

                        try:
                            engine.stop()
                        except:
                            pass

                    if event:
                        event.set()

                    self.queue.task_done()

        finally:

            pythoncom.CoUninitialize()

    # ======================================================
    # SPEAK
    # ======================================================

    def speak(
        self,
        text,
        wait=False,
        priority=2,
        allow_duplicate=False
    ):

        if not text:
            return

        current_time = time.time()

        # ----------------------------------------------
        # DUPLICATE PROTECTION
        # ----------------------------------------------

        if not allow_duplicate:

            if (
                text == self.last_text
                and
                current_time
                - self.last_text_time
                < 2
            ):

                print(
                    "Duplicate speech ignored:",
                    text
                )

                return

        self.last_text = text

        self.last_text_time = (
            current_time
        )

        finished_event = (
            threading.Event()
            if wait
            else None
        )

        self.counter += 1

        print(
            f"QUEUEING TTS "
            f"[priority={priority}] >>> {text}"
        )

        self.queue.put(
            (
                priority,
                self.counter,
                text,
                finished_event
            )
        )

        if finished_event:

            finished_event.wait()

    # ======================================================
    # CLEAR LOW PRIORITY SPEECH
    # ======================================================

    def clear_low_priority(self):

        remaining = []

        while True:

            try:

                item = (
                    self.queue.get_nowait()
                )

                priority, counter, text, event = item

                if priority <= 1:

                    remaining.append(
                        item
                    )

                else:

                    if event:
                        event.set()

                self.queue.task_done()

            except queue.Empty:

                break

        for item in remaining:

            self.queue.put(
                item
            )

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        print(
            "Stopping speech service..."
        )

        self.running = False

        self.counter += 1

        self.queue.put(
            (
                0,
                self.counter,
                None,
                None
            )
        )