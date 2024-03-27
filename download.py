from pathlib import Path

from loguru import logger

from func import SpeechToTextPipeline

# Constants
TEST_VOICE_FILE_PATH = Path("/workspace/sample.mp3")
DEFAULT_MODEL_ID = "openai/whisper-large-v3"


def model_init_wrapper(model_id: str = DEFAULT_MODEL_ID) -> str:
    """
    Initializes the model to test its loading and basic functionality.
    """

    try:
        pipeline = SpeechToTextPipeline(model_id=model_id)
        _text = pipeline(
            str(
                TEST_VOICE_FILE_PATH
            ),  # Conversion to str if Path is not directly supported
            batch_size=4,
            task="transcribe",
            language="auto",
        )
        assert _text["text"].strip() == "Hello, world."
        return (
            "Model loaded and tested successfully." if _text else "Model test failed."
        )
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        return f"Error during model initialization: {e}"


if __name__ == "__main__":
    logger.info(model_init_wrapper(DEFAULT_MODEL_ID))
