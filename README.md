# AutoStructurer-v5 (Linux CUDA, IVF-PQ)

Production-grade ingestion + indexing system for unstructured data:
- Text/PDF/Image/Video ingestion
- Whisper transcript extraction (GPU)
- OCR (EasyOCR GPU)
- Scene keyframes
- Batch embedding (SentenceTransformer + CLIP)
- FAISS GPU IVF-PQ index (text + clip)
- SQLite metadata store
- Topic centroid incremental assignment
- Memory decay
- Contradiction graph (rule-based MVP)
- Export to TurboMemory `.tm` bundle + zip

## Install (Linux CUDA)
```bash
sudo apt install ffmpeg
pip install -r requirements.txt
```

If `faiss-gpu-cu12` doesn't match your CUDA version, install FAISS GPU via conda.

## Run daemon (watch folder)
```bash
mkdir -p inbox
python daemon.py --watch inbox --db memory.sqlite --gpu
```

Drop files into `inbox/` (mp4/pdf/png/txt/etc).

## Ingest single file
```bash
python cli.py ingest myvideo.mp4 --db memory.sqlite --gpu
```

## Search
```bash
python cli.py search "compression ratio slide" --db memory.sqlite --mode hybrid --top-k 10
```

## Export `.tm` and zip
```bash
python cli.py export --db memory.sqlite --out memory.tm --zip export.zip
```

## Notes
- IVF-PQ needs training. This system auto-trains once it has enough vectors.
- For small datasets, it falls back to FlatIP until trained.
