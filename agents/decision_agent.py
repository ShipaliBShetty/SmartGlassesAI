import time


class DecisionAgent:

    def __init__(self, cooldown=30):

        # Seconds before repeating information
        self.cooldown = cooldown

        # Stores last announcement time
        self.last_announced = {}

        print("Decision agent ready.")

    # ==================================================
    # DECIDE WHAT TO ANNOUNCE
    # ==================================================

    def decide(self, name, emotion):

        # ----------------------------------------------
        # UNKNOWN PERSON
        # ----------------------------------------------

        # Do not disturb the user for strangers
        if name == "Unknown":
            return None

        # ----------------------------------------------
        # CHECK COOLDOWN
        # ----------------------------------------------

        current_time = time.time()

        last_time = self.last_announced.get(
            name,
            0
        )

        if (
            current_time - last_time
            < self.cooldown
        ):
            return None

        # ----------------------------------------------
        # CREATE MESSAGE
        # ----------------------------------------------

        if (
            emotion
            and emotion != "unknown"
        ):

            message = (
                f"{name} is in front of you "
                f"and looks {emotion}."
            )

        else:

            message = (
                f"{name} is in front of you."
            )

        # Remember announcement time
        self.last_announced[name] = (
            current_time
        )

        return message

    # ==================================================
    # FORCE PERSON TO BE ANNOUNCED AGAIN
    # ==================================================

    def reset_person(self, name):

        if name in self.last_announced:
            del self.last_announced[name]

    # ==================================================
    # RESET ALL
    # ==================================================

    def reset_all(self):

        self.last_announced.clear()