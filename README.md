# Free Voice Transcriber in 30 Seconds

Voice Transcriber is a sophisticated speech-to-text tool utilizing the OpenAI Whisper model, aimed at delivering efficient and accurate speech recognition across multiple languages. It excels in tasks such as speech recognition, translation, and language identification through its multilingual support.

## Features

- **Multilingual Support**: Utilizes the Whisper model to perform speech-to-text tasks across various languages.
- **Efficient Performance**: Achieves efficient audio processing through optimized models and hardware acceleration.
- **Flexible Deployment**: Offers easy access and usability via a Gradio-based web interface.

## Installation

Ensure Python 3.8+, Docker, and NVIDIA Container Toolkit are installed on your system. Install the project dependencies as follows:

1. **For NVIDIA GPU Instances**:

   - Install Docker and NVIDIA Container Toolkit if not already installed.
   - Use the command below to start the container:

     ```bash
     docker run --gpus all -p 49799:49799 -d cyfeng/voice_transcriber:latest
     ```

2. **For MPS Supported Mac Computers**:

   - Clone the repository and set up the environment:

     ```bash
     git clone https://github.com/CyFeng16/voice_transcriber
     cd voice_transcriber
     conda create -n voice_transcriber python=3.8
     conda activate voice_transcriber
     pip install -r requirements.txt
     python app.py
     ```

After installation, access the application via `http://localhost:49799`.

## Dependencies

   - PyTorch: [https://pytorch.org/](https://pytorch.org/)
   - openai-whisper: [https://github.com/openai/whisper](https://github.com/openai/whisper)
   - Gradio: [https://gradio.app/](https://gradio.app/)
   - Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)

## References

This project is built on the OpenAI Whisper model, detailed at its [GitHub repository](https://github.com/openai/whisper).

## License

Licensed under Apache-2.0, with the Whisper model and associated code under the MIT License.

## Acknowledgements

Special thanks to OpenAI for the Whisper model and to all contributors of this project.
