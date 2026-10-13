# Running CSM on Windows

This checkout uses Python 3.11 and an isolated `.venv`, upstream pinned dependencies, PyTorch 2.4.0 with CUDA 12.4, and `triton-windows==3.1.0.post17`. Windows requirements also pin NumPy 1.26.4 for PyTorch's ABI and include setuptools for Triton's runtime imports. The installed NVIDIA driver can run the bundled CUDA runtime; a separate CUDA toolkit is not needed for eager inference.

`bitsandbytes==0.45.5` is included because Moshi 0.2.2 imports it during prompt audio encoding even when quantization is disabled. If updating an existing setup, rerun `setup-windows.ps1` to install it.

From PowerShell in this directory, install or repeat setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1
```

The installer uses the existing `uv` command. For manual installation without uv:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
.\.venv\Scripts\python.exe -m pip check
powershell -ExecutionPolicy Bypass -File .\setup-ffmpeg.ps1
```

## Sentence UI

After setup and model approval, start the local UI:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-ui.ps1
```

Open http://127.0.0.1:7860 in your browser. The page immediately starts loading the model; you can enter up to 200 characters while it loads. Once **Model ready** appears, click **Generate voice**. Audio can be played or downloaded as WAV. Subsequent requests and page reloads reuse the loaded model. Failed loading shows a retry button. Each sentence uses a random speaker identity and no audio context, following upstream's basic example. Output is capped at 30 seconds and retains the original watermark. One sentence is generated at a time. Press Ctrl+C in the terminal to stop and release GPU memory. The server listens only on localhost.

## Model access

Accept the model terms and obtain access using your own Hugging Face account:

- https://huggingface.co/sesame/csm-1b
- https://huggingface.co/meta-llama/Llama-3.2-1B

Log in locally with a read token; do not put it in project files or chat:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
```

Then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run-windows.ps1
```

The launcher checks access to both model repositories before importing the model stack or downloading weights. If an approval is pending, it prints the relevant model URL and exits cleanly. Login alone does not grant access: the account that owns the token must be approved for both repositories, and the token must permit reading them.

The launcher sets `NO_TORCH_COMPILE=1` before any imports and adds the project-local FFmpeg directory to PATH. The first successful run downloads CSM, the Llama tokenizer, Mimi, SilentCipher weights, and the audio prompts. Successful generation writes `full_conversation.wav`. Watermarking remains enabled.

The installer downloads a checksum-verified [Gyan FFmpeg essentials build](https://www.gyan.dev/ffmpeg/builds/) to `.tools/ffmpeg`, including FFmpeg and ffprobe. It changes no global PATH settings. For your own Python commands:

```powershell
$env:NO_TORCH_COMPILE = '1'
$env:PATH = "$PWD\.tools\ffmpeg\bin;" + $env:PATH
```

Check dependency consistency and local inference prerequisites:

```powershell
uv pip check --python .venv/Scripts/python.exe
.\.venv\Scripts\python.exe verify_setup.py
```

The verification script checks CUDA BF16 operations, project imports, Moshi's audio linear path and bitsandbytes runtime import, WAV I/O, FFmpeg/ffprobe MP3 conversion, and model access without downloading weights. It exits with status 1 if either gated model is inaccessible. To check only access, run `.\.venv\Scripts\python.exe check_model_access.py`.
