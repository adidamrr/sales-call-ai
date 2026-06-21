from functools import lru_cache

from faster_whisper import WhisperModel

from src.config import settings


@lru_cache(maxsize=1)
def get_whisper_model():
    return WhisperModel(
        settings.WHISPER_MODEL,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )


def format_segments_as_dialogue_text(segments):
    lines = []

    for segment in segments:
        text = segment["text"].strip()
        if text:
            lines.append(f"Спикер: {text}")

    return "\n".join(lines)


def transcribe_audio(audio_path):
    model = get_whisper_model()
    segments_iterator, _ = model.transcribe(audio_path)

    segments = [
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        for segment in segments_iterator
    ]
    text = format_segments_as_dialogue_text(segments)

    return {
        "text": text,
        "segments": segments,
    }
