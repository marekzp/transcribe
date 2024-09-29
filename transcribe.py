import argparse
import logging
import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def transcribe_audio(file_path: str, model_id: str, language: str = None) -> str:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info("Loading model %s", model_id)
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        logger.info("Transcribing audio file: %s", file_path)

        generate_kwargs = {"task": "transcribe", "return_timestamps": True}
        if language:
            generate_kwargs["language"] = language

        result = pipe(file_path, generate_kwargs=generate_kwargs)

        return result["text"]
    except Exception:
        logger.exception("Error during model loading or transcription")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Whisper model."
    )
    parser.add_argument("file_path", help="Path to the audio file to transcribe")
    parser.add_argument(
        "--model_id", default="openai/whisper-large-v3", help="Whisper model ID to use"
    )
    parser.add_argument(
        "--output_file",
        default="transcription_output.txt",
        help="Path to the output text file",
    )
    parser.add_argument(
        "--language",
        help="Language code for transcription (e.g., 'en' for English, 'fr' for French)",
    )
    args = parser.parse_args()

    # Input validation
    if not os.path.exists(args.file_path):
        logger.error("The file '%s' does not exist.", args.file_path)
        return

    if not os.path.isfile(args.file_path):
        logger.error("'%s' is not a file.", args.file_path)
        return

    valid_audio_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    if not any(args.file_path.lower().endswith(ext) for ext in valid_audio_extensions):
        logger.error("'%s' does not appear to be a valid audio file.", args.file_path)
        return

    try:
        transcription = transcribe_audio(args.file_path, args.model_id, args.language)

        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(transcription)

        logger.info(
            "Transcription completed successfully. Output saved to '%s'.",
            args.output_file,
        )
    except Exception:
        logger.exception("An error occurred during transcription")


if __name__ == "__main__":
    main()
