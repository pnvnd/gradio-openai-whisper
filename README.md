# gradio-openai-whisper
User-friendly application to transcribe audio.

## Instructions
Before you run this locally, make sure these are installed:
```
pip install gradio
pip install openai-whisper
```

You also need the following in your PATH environment variable:  
https://www.ffmpeg.org/download.html
```
ffmpeg
ffprobe
```

Get the Shared LGPL version for this if you plan to distribute/keep code private. Windows example:
```
https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip
```

Finally, when you first run this, it'll download the whisper model (base, small, medium, or large).
Some networks prevent python from downloading these models, so you may need to manually download to `~/.cache/whisper`
```
https://github.com/openai/whisper/blob/main/whisper/__init__.py#L21
```

## Docker Instructions
Build this docker image with:
```
docker build -t transcription:latest .
```

To run the container, use:
```
docker run --name openai-whisper -dp 7860:7860 transcription:latest
```

Once the container is running, go to http://localhost:7860 to use the app.
