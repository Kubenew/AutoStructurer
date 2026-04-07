import argparse

from autostructurer.pipeline import AutoStructurer
from autostructurer.search import load_vectors, cosine_search
from autostructurer.embed_text import TextEmbedder
from autostructurer.embed_clip import CLIPEmbedder

def main():
    parser = argparse.ArgumentParser("AutoStructurer-v4 GPU")
    sub = parser.add_subparsers(dest="cmd")

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("path")
    p_ing.add_argument("--db", default="memory.db")

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--db", default="memory.db")
    p_search.add_argument("--top-k", type=int, default=8)
    p_search.add_argument("--mode", choices=["text", "clip", "hybrid"], default="hybrid")

    args = parser.parse_args()

    if args.cmd == "ingest":
        a = AutoStructurer(db_path=args.db)
        n = a.ingest_file(args.path)
        print("Inserted chunks:", n)

    elif args.cmd == "search":
        store = AutoStructurer(db_path=args.db).store

        results = []

        if args.mode in ["text", "hybrid"]:
            text_embedder = TextEmbedder()
            qvec = text_embedder.embed([args.query])[0]
            rows = store.fetch_vectors(kind="text")
            ids, vecs = load_vectors(rows)
            results.extend([(cid, score, "text") for cid, score in cosine_search(qvec, vecs, ids, top_k=args.top_k)])

        if args.mode in ["clip", "hybrid"]:
            clip_embedder = CLIPEmbedder()
            qvec = clip_embedder.embed_text(args.query)
            rows = store.fetch_vectors(kind="clip")
            if rows:
                ids, vecs = load_vectors(rows)
                results.extend([(cid, score, "clip") for cid, score in cosine_search(qvec, vecs, ids, top_k=args.top_k)])

        merged = {}
        for cid, score, kind in results:
            if cid not in merged or score > merged[cid]["score"]:
                merged[cid] = {"score": score, "kind": kind}

        merged_sorted = sorted(merged.items(), key=lambda x: -x[1]["score"])[:args.top_k]

        print("\nQuery:", args.query, "| mode:", args.mode)

        for cid, info in merged_sorted:
            row = store.fetch_chunk(cid)
            if not row:
                continue

            (chunk_id, doc_id, source, schema, t_start, t_end, text,
             entities_json, topic, contradiction, confidence, ref_path, phash) = row

            print(f"\n[{info['score']:.3f}] via={info['kind']} schema={schema} source={source} topic={topic}")
            print(f"time: {t_start:.1f}-{t_end:.1f} conf={confidence:.2f} contradiction={contradiction:.2f}")

            if ref_path:
                print("ref:", ref_path)

            print(text[:350])

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
