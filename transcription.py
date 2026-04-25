'''
Before you run this, make sure these are installed:
pip install gradio
pip install openai-whisper

You also need the following in your PATH environment variable: https://www.ffmpeg.org/download.html
ffmpeg
ffprobe

Get the Shared LGPL version for this if you plan to distribute/keep code private:
https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip

Finally, when you first run this, it'll download the whisper model (base, small, medium, or large).
Some networks prevent python from downloading, so you may need to manually download to ~/.cache/whisper
https://github.com/openai/whisper/blob/main/whisper/__init__.py#L21
'''

import whisper
import gradio as gr
from datetime import datetime
import os

# Language mapping
LANGUAGE_CODES = {
    "English": "en",
    "Japanese": "ja",
    "Chinese": "zh",
}

# Available models
AVAILABLE_MODELS = [
    "base",
    "base.en",
    "tiny",
    "tiny.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo"
]

# Model cache
loaded_model = None
current_model_name = None

def load_model(model_name: str):
    """Load model from local cache or download from internet."""
    global loaded_model, current_model_name
    
    if loaded_model and current_model_name == model_name:
        return loaded_model
    
    model_path = os.path.join(os.getcwd(), f"{model_name}.pt")
    
    # Check if local file exists
    if os.path.exists(model_path):
        print(f"Loading local model from {model_path}")
    else:
        print(f"Local model not found at {model_path}. Downloading {model_name}...")
    
    # Load model (will use local cache if available, otherwise download)
    model = whisper.load_model(model_name)
    
    loaded_model = model
    current_model_name = model_name
    return model

def format_srt_timestamp(seconds):
    """Format seconds into SRT timestamp HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def is_recorded_audio(audio_path: str) -> bool:
    """Return True only for microphone-recorded audio files."""
    if not audio_path:
        return False
    basename = os.path.basename(audio_path)
    return basename.startswith("tmp") or basename.startswith("audio")

def prepare_audio_download(audio_path: str):
    """Expose the recorded audio path for download only for recordings."""
    if not is_recorded_audio(audio_path):
        return gr.update(visible=False, value=None)

    try:
        dest_dir = os.environ.get("TMP") or os.environ.get("TEMP") or os.getcwd()
        file_ext = os.path.splitext(audio_path)[1] or ".wav"
        download_filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
        download_path = os.path.join(dest_dir, download_filename)
        with open(audio_path, "rb") as src, open(download_path, "wb") as dst:
            dst.write(src.read())
        return gr.update(visible=True, value=download_path)
    except Exception:
        return gr.update(visible=False, value=None)

def transcribe(audio_path: str, language: str, model_name: str, translate: bool) -> tuple:
    """Transcribe audio file using OpenAI Whisper model."""
    input_path = audio_path
    
    if not input_path:
        return "Please upload a file or record audio.", gr.update(visible=False)
    
    # Load the selected model
    model = load_model(model_name)
    
    # Map user-friendly language name to code, or use custom input as-is
    lang_code = LANGUAGE_CODES.get(language, language)
    
    # Determine if translation is needed
    is_english = lang_code == "en"
    task = "translate" if translate and not is_english else "transcribe"
    
    result = model.transcribe(input_path, language=lang_code, task=task, verbose=True)
    
    # Build SRT transcription text with timestamps
    segments = result.get("segments", [])
    transcription = ""
    if segments:
        for idx, segment in enumerate(segments, start=1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            start_str = format_srt_timestamp(start)
            end_str = format_srt_timestamp(end)
            transcription += f"{idx}\n{start_str} --> {end_str}\n{text}\n\n"
    else:
        text = result.get("text", "").strip()
        transcription = f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    transcription_filename = f"[{model_name}]_{timestamp}_{input_filename}.srt"
    dest_dir = os.environ.get("TMP") or os.environ.get("TEMP") or os.getcwd()
    transcription_path = os.path.join(dest_dir, transcription_filename)
    with open(transcription_path, 'w', encoding='utf-8') as f:
        f.write(transcription)

    return transcription, gr.update(visible=True, value=transcription_path)

def toggle_translate_visibility(language: str):
    """Show translate checkbox only for non-English languages."""
    lang_code = LANGUAGE_CODES.get(language, language)
    return gr.update(visible=lang_code != "en")

with gr.Blocks() as iface:
    gr.Markdown("# OpenAI Whisper - Speech Recognition")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Audio")
        
        with gr.Column():
            language_dropdown = gr.Dropdown(
                choices=["English", "Japanese", "Chinese"],
                value="English",
                label="Select Language",
                allow_custom_value=True
            )
            model_dropdown = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value="base",
                label="Select Model"
            )
            translate_checkbox = gr.Checkbox(label="Translate to English", value=False, visible=False)
            transcribe_btn = gr.Button("Transcribe", scale=1)
    
    transcription_output = gr.Textbox(label="Transcription", lines=10)
    
    download_audio_output = gr.File(label="Download Recorded Audio", visible=False, type="filepath")
    transcription_download_output = gr.File(label="Download Transcription", visible=False, type="filepath")
    
    language_dropdown.change(
        toggle_translate_visibility,
        inputs=[language_dropdown],
        outputs=[translate_checkbox]
    )
    
    transcribe_btn.click(
        transcribe,
        inputs=[audio_input, language_dropdown, model_dropdown, translate_checkbox],
        outputs=[transcription_output, transcription_download_output]
    )
    
    audio_input.change(
        prepare_audio_download,
        inputs=[audio_input],
        outputs=[download_audio_output]
    )

iface.launch(server_name="0.0.0.0", server_port=7860)
