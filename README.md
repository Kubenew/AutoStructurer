# AutoStructurer-v4 (GPU)

Universal unstructured → structured memory system with:
- Whisper transcription (video/audio)
- OCR extraction (EasyOCR)
- Scene detection for keyframes
- Text embeddings (SentenceTransformer)
- CLIP embeddings for images/video frames
- Frame deduplication using perceptual hash
- SQLite metadata store
- Packed 4-bit embeddings for storage

## Install
```bash
pip install -r requirements.txt
```

Linux:
```bash
sudo apt install ffmpeg
```

## Ingest
```bash
python cli.py ingest demo.mp4 --db memory.db
python cli.py ingest contract.pdf --db memory.db
python cli.py ingest slide.png --db memory.db
```

## Search (hybrid)
```bash
python cli.py search "compression ratio slide" --db memory.db --mode hybrid
```

## Search (visual-only CLIP)
```bash
python cli.py search "a chart on a presentation slide" --db memory.db --mode clip
```

## Search (text-only)
```bash
python cli.py search "invoice payment eur" --db memory.db --mode text
```
