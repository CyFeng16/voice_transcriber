import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict

from fastapi import FastAPI, File, UploadFile, Query
from loguru import logger

from func import SpeechToTextPipeline

app = FastAPI()

# Constants for default values
DEFAULT_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_TASK = "transcribe"
DEFAULT_LANGUAGE = "auto"
DEFAULT_BATCH_SIZE = 12


@app.post("/process_voice", response_model=Dict[str, Union[str, int]])
async def process_voice(
    audio_file: UploadFile = File(...),
    model_id: Optional[str] = Query(
        DEFAULT_MODEL_ID, title="The ID of the model to use"
    ),
    task: Optional[str] = Query(
        DEFAULT_TASK,
        title="Task to perform",
        description="Either transcribe or translate",
    ),
    language: Optional[str] = Query(
        DEFAULT_LANGUAGE,
        title="The language of the audio or target language for translation",
    ),
    batch_size: Optional[int] = Query(
        DEFAULT_BATCH_SIZE, title="The batch size for processing", ge=1
    ),
) -> Dict[str, Union[str, int]]:
    """
    Processes voice input based on the specified task: transcription or translation.
    """
    temp_file_path = Path(tempfile.mktemp(suffix=".tmp"))
    try:
        with temp_file_path.open("wb") as temp_file:
            shutil.copyfileobj(audio_file.file, temp_file)

        pipeline = SpeechToTextPipeline(model_id=model_id)
        result = pipeline(
            temp_file_path, batch_size=batch_size, task=task, language=language
        )
        return {"status": "success", "message": result.get("text", "").strip()}
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return {"status": "error", "message": f"Error during processing: {e}"}
    finally:
        temp_file_path.unlink(missing_ok=True)


@app.post("/init_model", response_model=Dict[str, str])
async def init_model(
    model_id: Optional[str] = Query(
        DEFAULT_MODEL_ID, title="The ID of the model to initialize"
    )
) -> Dict[str, str]:
    """
    Initializes the specified model for testing its loading and basic functionality.
    """
    try:
        pipeline = SpeechToTextPipeline(model_id=model_id)
        # It's better to use a predefined sample or mock for testing rather than a fixed path
        result = pipeline(
            Path("/workspace/sample.mp3"),
            batch_size=4,
            task="transcribe",
            language="auto",
        )
        result_text = result.get("text", "").strip()
        if result_text == "Hello, world.":
            return {
                "status": "success",
                "message": "Model loaded and tested successfully.",
            }
        else:
            return {"status": "error", "message": "Model test failed."}
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        return {"status": "error", "message": f"Error during processing: {e}"}


@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
async def catch_all(full_path: Optional[str] = "") -> Dict[str, str]:
    """
    Catches all unmatched paths and returns a helpful message.
    """
    return {
        "status": "error",
        "message": "This path is not supported.",
        "help": "Use '/process_voice' to process voice with optional query parameters 'model_id', 'task', 'language', and 'batch_size'. Example: POST /process_voice?task=transcribe",
    }
