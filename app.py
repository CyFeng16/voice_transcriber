import gradio as gr

from func import SpeechToTextPipeline, string_to_port_crc32

# Constants
LOCAL_CLIENT_IP: str = "0.0.0.0"
APP_NAME: str = "voice_transcriber"
DEFAULT_PORT: int = string_to_port_crc32(APP_NAME)  # 49799


def setup_gradio_interface() -> gr.Blocks:
    """
    Sets up the Gradio interface for voice transcription.

    Returns:
        gr.Blocks: A Gradio Blocks interface for transcribing voice to text.
    """

    with gr.Blocks(title="VoiceTranscriber") as interface:
        with gr.Row():
            gr.Markdown(
                "# [Voice Transcriber](https://github.com/CyFeng16/voice_transcriber)"
            )
        with gr.Row():
            with gr.Column():
                input_voice = gr.Audio(label="Input voice: ", type="filepath")
                with gr.Row():
                    with gr.Column():
                        model_id = gr.Dropdown(
                            choices=[
                                "openai/whisper-large-v3",
                                "distil-whisper/large-v2",
                            ],
                            value="openai/whisper-large-v3",
                            label="ASR model to use: ",
                        )
                        batch_size = gr.Number(
                            value=8,
                            precision=0,
                            minimum=4,
                            step=4,
                            label="BatchSize: (reduce if OOM occurs)",
                        )
                    with gr.Column():
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
            with gr.Column():
                output_text = gr.Textbox(
                    label="Result: ",
                    placeholder="Transcribed text",
                )
        with gr.Row():
            init_btn = gr.Button(value="Download Model Weights")
            transcribe_btn = gr.Button(value="Transcribe/Translate!")

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

    return interface


def transcribe_voice_wrapper(
    voice_fp: str, model_id: str, batch_size: int, model_task: str, priori_language: str
) -> str:
    """
    Wraps the speech-to-text pipeline for use with Gradio, handling the transcription process.
    """

    try:
        pipeline = SpeechToTextPipeline(model_id=model_id)
        transcribed_text = pipeline(
            voice_fp,
            batch_size=batch_size,
            task=model_task,
            language=priori_language,
        )
        return transcribed_text["text"]
    except Exception as e:
        return f"Error during transcription: {e}"


def model_init_wrapper(model_id: str) -> str:
    """
    Initializes the model to test its loading and basic functionality.
    """

    test_voice_fp: str = "/workspace/sample.mp3"
    try:
        _text = transcribe_voice_wrapper(
            voice_fp=test_voice_fp,
            model_id=model_id,
            batch_size=2,
            model_task="transcribe",
            priori_language="auto",
        )
        return (
            "Model loaded and tested successfully." if _text else "Model test failed."
        )
    except Exception as e:
        return f"Error during model initialization: {e}"


if __name__ == "__main__":
    demo = setup_gradio_interface()
    demo.queue()
    demo.launch(server_name=LOCAL_CLIENT_IP, server_port=DEFAULT_PORT)
