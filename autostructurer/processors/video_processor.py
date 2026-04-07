import os
import ffmpeg
import whisper
import easyocr
from tqdm import tqdm
from autostructurer.scene_detect import extract_scene_keyframes

def extract_audio(video_path: str, out_wav: str):
    ffmpeg.input(video_path).output(out_wav, ac=1, ar=16000).overwrite_output().run(quiet=True)

def process_video(path: str, temp_dir="tmp_video", keyframe_threshold=30.0):
    os.makedirs(temp_dir, exist_ok=True)

    audio_path = os.path.join(temp_dir, "audio.wav")
    extract_audio(path, audio_path)

    w = whisper.load_model("base")
    result = w.transcribe(audio_path)

    chunks = []
    for seg in result["segments"]:
        chunks.append({
            "text": seg["text"].strip(),
            "t_start": float(seg["start"]),
            "t_end": float(seg["end"]),
            "ref_path": None,
            "source": "speech"
        })

    frames_dir = os.path.join(temp_dir, "scenes")
    scenes = extract_scene_keyframes(path, frames_dir, threshold=keyframe_threshold, step=10)

    reader = easyocr.Reader(["en", "ru", "cs"], gpu=True)

    for t, frame_path, score in tqdm(scenes, desc="OCR scenes"):
        lines = reader.readtext(frame_path, detail=0)
        text = " ".join([x.strip() for x in lines if x.strip()]).strip()

        if not text:
            continue

        chunks.append({
            "text": text,
            "t_start": t,
            "t_end": t + 2.0,
            "ref_path": frame_path,
            "source": "ocr_scene"
        })

    return chunks
