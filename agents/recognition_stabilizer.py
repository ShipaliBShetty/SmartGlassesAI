from collections import defaultdict, deque, Counter


class RecognitionStabilizer:

    def __init__(
        self,
        history_size=5,
        minimum_votes=3
    ):

        self.history_size = history_size
        self.minimum_votes = minimum_votes

        self.history = defaultdict(
            lambda: deque(
                maxlen=self.history_size
            )
        )

    # ==================================================
    # UPDATE
    # ==================================================

    def update(
        self,
        track_id,
        name
    ):

        self.history[
            track_id
        ].append(
            name
        )

        names = list(
            self.history[
                track_id
            ]
        )

        # Not enough history yet
        if len(names) < self.minimum_votes:

            # If we already have a known name,
            # use the latest known one temporarily.
            for item in reversed(names):

                if item != "Unknown":
                    return item

            return "Unknown"

        # Count known names only
        known_names = [
            item
            for item in names
            if item != "Unknown"
        ]

        if not known_names:
            return "Unknown"

        counts = Counter(
            known_names
        )

        best_name, count = (
            counts.most_common(1)[0]
        )

        # Require enough votes
        if count >= self.minimum_votes:
            return best_name

        # If majority is not strong enough,
        # keep most recent known identity briefly
        for item in reversed(names):

            if item != "Unknown":
                return item

        return "Unknown"

    # ==================================================
    # CLEAR
    # ==================================================

    def clear(
        self,
        track_id
    ):

        if track_id in self.history:

            del self.history[
                track_id
            ]

    def clear_all(self):

        self.history.clear()