import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognitionAgent:

    def __init__(self):

        print("Loading InsightFace model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("InsightFace loaded.")

    # ==================================================
    # DETECT FACES
    # ==================================================

    def get_faces(self, frame):

        if frame is None:
            return []

        return self.app.get(frame)

    # ==================================================
    # GET FACE EMBEDDING
    # ==================================================

    def get_embedding(self, face):

        if face is None:
            return None

        embedding = face.embedding

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        # Normalize embedding
        embedding = embedding / norm

        return embedding

    # ==================================================
    # COMPARE TWO FACE EMBEDDINGS
    # ==================================================

    def compare(
        self,
        embedding1,
        embedding2
    ):

        if embedding1 is None or embedding2 is None:
            return -1.0

        embedding1 = np.asarray(
            embedding1,
            dtype=np.float32
        )

        embedding2 = np.asarray(
            embedding2,
            dtype=np.float32
        )

        # Make sure both embeddings have
        # the same number of values
        if embedding1.shape != embedding2.shape:
            return -1.0

        norm1 = np.linalg.norm(
            embedding1
        )

        norm2 = np.linalg.norm(
            embedding2
        )

        if norm1 == 0 or norm2 == 0:
            return -1.0

        # IMPORTANT:
        # Normalize BOTH embeddings before comparison.
        #
        # This ensures that dot product behaves
        # like cosine similarity.

        embedding1 = (
            embedding1 / norm1
        )

        embedding2 = (
            embedding2 / norm2
        )

        similarity = np.dot(
            embedding1,
            embedding2
        )

        # Numerical safety.
        # Cosine similarity should remain
        # between -1 and +1.

        similarity = np.clip(
            similarity,
            -1.0,
            1.0
        )

        return float(similarity)