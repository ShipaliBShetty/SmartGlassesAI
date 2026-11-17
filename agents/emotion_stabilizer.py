from collections import defaultdict, deque, Counter


class EmotionStabilizer:

    def __init__(
        self,
        history_size=5,
        minimum_samples=3
    ):

        self.history_size = history_size
        self.minimum_samples = minimum_samples

        self.history = defaultdict(
            lambda: deque(
                maxlen=self.history_size
            )
        )

    # ==================================================
    # UPDATE EMOTION
    # ==================================================

    def update(
        self,
        person_name,
        emotion
    ):

        # Ignore invalid emotion
        if (
            not emotion
            or emotion == "unknown"
        ):

            return "analyzing"

        # Add latest emotion to history
        self.history[
            person_name
        ].append(
            emotion
        )

        emotions = list(
            self.history[
                person_name
            ]
        )

        print(
            f"Emotion history for {person_name}: "
            f"{emotions}"
        )

        # ----------------------------------------------
        # WAIT FOR ENOUGH SAMPLES
        # ----------------------------------------------

        if (
            len(emotions)
            < self.minimum_samples
        ):

            return "analyzing"

        # ----------------------------------------------
        # FIND MOST COMMON EMOTION
        # ----------------------------------------------

        counts = Counter(
            emotions
        )

        stable_emotion, count = (
            counts.most_common(1)[0]
        )

        # ----------------------------------------------
        # REQUIRE MAJORITY
        # ----------------------------------------------

        required_votes = (
            len(emotions) // 2
        ) + 1

        if count < required_votes:

            return "analyzing"

        return stable_emotion

    # ==================================================
    # GET CURRENT STABLE EMOTION
    # ==================================================

    def get_stable_emotion(
        self,
        person_name
    ):

        if (
            person_name
            not in self.history
        ):

            return "analyzing"

        emotions = list(
            self.history[
                person_name
            ]
        )

        if (
            len(emotions)
            < self.minimum_samples
        ):

            return "analyzing"

        counts = Counter(
            emotions
        )

        stable_emotion, count = (
            counts.most_common(1)[0]
        )

        required_votes = (
            len(emotions) // 2
        ) + 1

        if count < required_votes:

            return "analyzing"

        return stable_emotion

    # ==================================================
    # CLEAR ONE PERSON
    # ==================================================

    def clear_person(
        self,
        person_name
    ):

        if (
            person_name
            in self.history
        ):

            del self.history[
                person_name
            ]

    # ==================================================
    # CLEAR ALL
    # ==================================================

    def clear_all(self):

        self.history.clear()