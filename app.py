import mimetypes
import os
from pathlib import Path

import gradio as gr
import requests
from typing import Tuple

from gradio import Dropdown, Number
from loguru import logger

# Constants
LOCAL_IP = os.environ.get("LOCAL_IP", "0.0.0.0")
LOCAL_PORT = int(os.environ.get("LOCAL_PORT", 49800))
REMOTE_IP = os.environ.get("REMOTE_IP", "0.0.0.0")
REMOTE_PORT = int(os.environ.get("REMOTE_PORT", 49799))
SERVER_URL = f"http://{REMOTE_IP}:{REMOTE_PORT}"


def setup_gradio_interface() -> gr.Blocks:
    """
    Sets up the Gradio interface for voice transcription.
    """
    with gr.Blocks(title="VoiceTranscriber") as interface:
        with gr.Row():
            gr.Markdown(
                "# [Voice Transcriber](https://github.com/CyFeng16/voice_transcriber)"
            )

        with gr.Row():
            with gr.Column():
                input_voice = gr.Audio(label="Input voice: ", type="filepath")
                (
                    model_id,
                    model_task,
                    priori_language,
                    batch_size,
                ) = setup_model_options()
                init_btn, transcribe_btn = setup_action_buttons()
            with gr.Column():
                output_text = setup_output_text()

        setup_button_actions(
            init_btn,
            transcribe_btn,
            model_id,
            input_voice,
            batch_size,
            model_task,
            priori_language,
            output_text,
        )

    return interface


def setup_model_options() -> tuple[Dropdown, Dropdown, Dropdown, Number]:
    """
    Sets up the model options for the Gradio interface.
    """
    model_id = gr.Dropdown(
        choices=["openai/whisper-large-v3", "distil-whisper/large-v2"],
        value="openai/whisper-large-v3",
        label="ASR model to use: ",
    )
    model_task = gr.Dropdown(
        choices=["transcribe", "translate"],
        value="transcribe",
        label="Model task: ",
    )
    priori_language = gr.Dropdown(
        choices=[
            "auto",
            "arabic",
            "chinese",
            "english",
            "french",
            "russian",
            "spanish",
        ],
        value="auto",
        label="Priori language: ",
    )
    batch_size = gr.Number(
        value=12,
        precision=0,
        minimum=4,
        step=4,
        label="BatchSize: (reduce if OOM occurs)",
    )
    return model_id, model_task, priori_language, batch_size


def setup_action_buttons() -> Tuple[gr.Button, gr.Button]:
    """
    Sets up the action buttons for the Gradio interface.
    """
    init_btn = gr.Button(value="Model Init.")
    transcribe_btn = gr.Button(value="Transcribe/Translate!")
    return init_btn, transcribe_btn


def setup_output_text() -> gr.Textbox:
    """
    Sets up the output text box for the Gradio interface.
    """
    return gr.Textbox(label="Result: ", placeholder="Transcribed text")


def setup_button_actions(
    init_btn,
    transcribe_btn,
    model_id,
    input_voice,
    batch_size,
    model_task,
    priori_language,
    output_text,
):
    """
    Configures the actions for the buttons in the Gradio interface.
    """
    init_btn.click(
        fn=model_init_wrapper,
        inputs=model_id,
        outputs=output_text,
    )
    transcribe_btn.click(
        fn=transcribe_voice_wrapper,
        inputs=[input_voice, model_id, batch_size, model_task, priori_language],
        outputs=output_text,
    )


def transcribe_voice_wrapper(
    voice_fp: str,
    model_id: str,
    batch_size: int,
    model_task: str,
    priori_language: str,
    progress=gr.Progress(),
) -> str:
    """
    Wraps the speech-to-text pipeline for use with Gradio.
    """
    try:
        url = f"{SERVER_URL}/process_voice"
        params = {
            "model_id": model_id,
            "task": model_task,
            "language": priori_language,
            "batch_size": batch_size,
        }
        mime_type, _ = mimetypes.guess_type(voice_fp)
        mime_type = mime_type if mime_type else "application/octet-stream"

        with open(voice_fp, "rb") as audio_file:
            files = {"audio_file": (Path(voice_fp).name, audio_file, mime_type)}
            response = requests.post(url, files=files, params=params).json()
        logger.info(f"Response: {response}")

        assert response["status"] == "success"
        return response["message"]
    except Exception as e:
        logger.error(f"Error during {model_task}: {e}")
        return f"Error during {model_task}: {e}"


def model_init_wrapper(model_id: str, progress=gr.Progress()) -> str:
    """
    Initializes the model for the Gradio interface.
    """
    try:
        logger.info(f"Initializing model: {model_id}")
        url = f"{SERVER_URL}/init_model"
        logger.info(f"Using server api: {url}")
        params = {"model_id": model_id}
        response = requests.post(url, params=params).json()
        assert response["status"] == "success"
        return response["message"]
    except Exception as e:
        logger.error(f"Error during model initialization: {e}")
        return f"Error during model initialization: {e}"


if __name__ == "__main__":
    demo = setup_gradio_interface()
    demo.queue()
    demo.launch(
        server_name=LOCAL_IP,
        server_port=LOCAL_PORT,
    )
