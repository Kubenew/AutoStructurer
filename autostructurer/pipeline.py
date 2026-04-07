import os
from tqdm import tqdm

from autostructurer.utils import sha1_id
from autostructurer.schema_detect import detect_schema
from autostructurer.entity_extract import extract_entities
from autostructurer.contradiction import detect_contradictions
from autostructurer.topic_cluster import cluster_topics

from autostructurer.embed_text import TextEmbedder
from autostructurer.embed_clip import CLIPEmbedder
from autostructurer.pack4bit import pack_4bit
from autostructurer.storage_sqlite import SQLiteStore
from autostructurer.dedup_phash import compute_phash, hamming_distance

from autostructurer.processors.text_processor import process_text_file
from autostructurer.processors.pdf_processor import process_pdf
from autostructurer.processors.image_processor import process_image
from autostructurer.processors.video_processor import process_video

class AutoStructurer:
    def __init__(self, db_path="memory.db"):
        self.store = SQLiteStore(db_path)
        self.text_embedder = TextEmbedder()
        self.clip_embedder = CLIPEmbedder()
        self._recent_hashes = []

    def _dedup_frame(self, ref_path: str, max_dist=6):
        if not ref_path:
            return None

        try:
            ph = compute_phash(ref_path)
        except:
            return None

        for old in self._recent_hashes[-200:]:
            if hamming_distance(ph, old) <= max_dist:
                return None

        self._recent_hashes.append(ph)
        return ph

    def ingest_file(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        doc_id = sha1_id(os.path.abspath(path))

        if ext in [".txt", ".md", ".log", ".html"]:
            raw_chunks = process_text_file(path)
        elif ext == ".pdf":
            raw_chunks = process_pdf(path)
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            raw_chunks = process_image(path)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            raw_chunks = process_video(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        texts = [c["text"] for c in raw_chunks]
        if not texts:
            return 0

        schema = detect_schema(" ".join(texts[:5]))
        contradictions = detect_contradictions(texts)

        text_vecs = self.text_embedder.embed(texts)
        topics = cluster_topics(text_vecs, k=10)

        inserted = 0

        for i, chunk in enumerate(tqdm(raw_chunks, desc="Indexing chunks")):
            cid = sha1_id(doc_id + str(i) + chunk["text"][:200])

            entities = extract_entities(chunk["text"])
            topic = f"topic_{topics[i]}"
            confidence = 0.85 if len(chunk["text"]) > 30 else 0.55

            phash = None
            if chunk.get("ref_path"):
                phash = self._dedup_frame(chunk["ref_path"])
                if phash is None:
                    continue

            record = {
                "chunk_id": cid,
                "doc_id": doc_id,
                "source": chunk.get("source", "unknown"),
                "schema": schema,
                "t_start": chunk.get("t_start", 0.0),
                "t_end": chunk.get("t_end", 0.0),
                "text": chunk["text"],
                "entities": entities,
                "topic": topic,
                "contradiction": contradictions[i],
                "confidence": confidence,
                "ref_path": chunk.get("ref_path"),
                "phash": phash
            }

            self.store.insert_chunk(record)

            packed, scale, zero = pack_4bit(text_vecs[i])
            self.store.insert_vector(cid, "text", text_vecs.shape[1], packed, scale, zero)

            if chunk.get("ref_path"):
                try:
                    clip_vec = self.clip_embedder.embed_image(chunk["ref_path"])
                    packed2, scale2, zero2 = pack_4bit(clip_vec)
                    self.store.insert_vector(cid, "clip", len(clip_vec), packed2, scale2, zero2)
                except:
                    pass

            inserted += 1

        return inserted
