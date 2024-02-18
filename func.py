import zlib
from pathlib import Path
from typing import Union, Optional

import torch
from loguru import logger
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Define constants for clarity and easy maintenance
MIN_PORT: int = 10000
MAX_PORT: int = 60000
DEFAULT_MODEL_ID: str = "openai/whisper-large-v3"
DEFAULT_BATCH_SIZE: int = 8  # Approx. 7GB of GPU memory
CHUNK_LENGTH_SECONDS: int = 30
MAX_NEW_TOKENS: int = 128


def string_to_port_crc32(input_string: str) -> int:
    """
    Converts a string to a CRC32 hash and then maps it to a port number within a specified range.
    """

    crc32_hash: int = zlib.crc32(input_string.encode()) & 0xFFFFFFFF
    port_number: int = MIN_PORT + crc32_hash % (MAX_PORT - MIN_PORT)
    return port_number


def set_device() -> torch.device:
    """
    Determines the best available device for PyTorch operations (MPS, CUDA, or CPU).
    """

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device


class SpeechToTextPipeline:
    """
    Encapsulates the Speech-to-Text conversion process using a Hugging Face pipeline.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id: str = model_id
        self.model: Optional[AutoModelForSpeechSeq2Seq] = None
        self.device: torch.device = set_device()
        self.load_model()

    def load_model(self):
        """
        Loads the model based on the predefined model ID.
        """

        if self.model is not None:
            logger.info("Model already loaded.")
            return
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            ).to(self.device)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def __call__(
        self,
        audio_path: Union[str, Path],
        batch_size: int = DEFAULT_BATCH_SIZE,
        return_timestamps: bool = False,
    ):
        """
        Transcribes the audio file specified by `audio_path`.
        """

        audio_path = self.validate_audio_path(audio_path)
        processor = AutoProcessor.from_pretrained(self.model_id)
        pipe = self.setup_pipeline(processor, batch_size, return_timestamps)
        logger.info("Transcribing audio...")
        text = pipe(str(audio_path))
        logger.info(f"Transcription complete: {text}")
        return text

    @staticmethod
    def validate_audio_path(audio_path: Union[str, Path]) -> Path:
        """
        Validates the existence of the audio file.
        """

        audio_path = (
            Path(audio_path) if not isinstance(audio_path, Path) else audio_path
        )
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file {audio_path} not found.")
        return audio_path

    def setup_pipeline(
        self, processor, batch_size: int, return_timestamps: bool
    ) -> pipeline:
        """
        Configures the Hugging Face pipeline for ASR (Automatic Speech Recognition).
        """

        try:
            from transformers.utils import is_flash_attn_2_available
        except ImportError:
            # Defaults to False if not available
            is_flash_attn_2_available = lambda: False

        return pipeline(
            "automatic-speech-recognition",
            model=self.model,
            torch_dtype=torch.float16,
            chunk_length_s=CHUNK_LENGTH_SECONDS,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
            device=self.device,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            model_kwargs={
                "attn_implementation": (
                    "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
                )
            },
            generate_kwargs={"task": "transcribe"},
        )
