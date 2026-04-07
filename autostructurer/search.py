import numpy as np
from autostructurer.pack4bit import unpack_4bit

def load_vectors(rows):
    ids, vecs = [], []
    for chunk_id, dim, packed, scale, zero in rows:
        v = unpack_4bit(packed, dim, scale, zero)
        ids.append(chunk_id)
        vecs.append(v)
    return ids, np.vstack(vecs)

def cosine_search(query_vec, vecs, ids, top_k=10):
    scores = vecs @ query_vec
    idx = np.argsort(-scores)[:top_k]
    return [(ids[i], float(scores[i])) for i in idx]
