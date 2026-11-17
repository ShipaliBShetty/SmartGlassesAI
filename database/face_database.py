import os
import sqlite3
import numpy as np

from config import DATABASE_PATH


class FaceDatabase:

    def __init__(self):

        os.makedirs(
            os.path.dirname(DATABASE_PATH),
            exist_ok=True
        )

        self.connection = sqlite3.connect(DATABASE_PATH)

        self.create_table()

    def create_table(self):

        cursor = self.connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL
            )
        """)

        self.connection.commit()

    def add_person(self, name, embedding):

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        embedding_bytes = embedding.tobytes()

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO faces
            (name, embedding)
            VALUES (?, ?)
            """,
            (name, embedding_bytes)
        )

        self.connection.commit()

    def get_all_people(self):

        cursor = self.connection.cursor()

        cursor.execute(
            "SELECT name, embedding FROM faces"
        )

        rows = cursor.fetchall()

        people = []

        for name, embedding_bytes in rows:

            embedding = np.frombuffer(
                embedding_bytes,
                dtype=np.float32
            )

            people.append({
                "name": name,
                "embedding": embedding
            })

        return people

    def delete_person(self, name):

        cursor = self.connection.cursor()

        cursor.execute(
            "DELETE FROM faces WHERE name = ?",
            (name,)
        )

        self.connection.commit()

        return cursor.rowcount > 0

    def person_exists(self, name):

        cursor = self.connection.cursor()

        cursor.execute(
            "SELECT id FROM faces WHERE name = ?",
            (name,)
        )

        return cursor.fetchone() is not None

    def close(self):

        self.connection.close()