"""Local sentence-to-speech UI following Sesame's context-free example."""
import io
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['NO_TORCH_COMPILE'] = '1'
os.environ['PATH'] = str(ROOT / '.tools/ffmpeg/bin') + os.pathsep + os.environ['PATH']

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4096
_generator = None
_generation_lock = threading.Lock()


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/generate')
def generate():
    data = request.get_json(silent=True)
    text = data.get('text') if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        return jsonify(error='Type a sentence to generate speech.'), 400
    text = text.strip()
    if len(text) > 200:
        return jsonify(error='Use 200 characters or fewer.'), 400
    if not _generation_lock.acquire(blocking=False):
        return jsonify(error='Another sentence is being generated. Try again when it finishes.'), 409
    try:
        import torch
        import torchaudio
        from generator import load_csm_1b
        from huggingface_hub.errors import GatedRepoError

        global _generator
        if not torch.cuda.is_available():
            return jsonify(error='CUDA is unavailable. Check your NVIDIA driver and run setup-windows.ps1.'), 503
        try:
            if _generator is None:
                _generator = load_csm_1b(device='cuda')
            audio = _generator.generate(
                text=text, speaker=0, context=[], max_audio_length_ms=30_000,
            )
            if audio.numel() == 0 or not torch.isfinite(audio).all().item():
                raise RuntimeError('The model returned empty or invalid audio.')
            output = io.BytesIO()
            torchaudio.save(output, audio.unsqueeze(0).cpu(), _generator.sample_rate, format='wav')
        except GatedRepoError:
            return jsonify(error='Model access is missing. Approve both Sesame CSM-1B and Llama-3.2-1B, then log in with huggingface-cli.'), 403
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return jsonify(error='The GPU ran out of memory. Close other GPU apps and try a shorter sentence.'), 503
        output.seek(0)
        response = send_file(output, mimetype='audio/wav', download_name='speech.wav')
        response.headers['Cache-Control'] = 'no-store'
        return response
    except Exception:
        app.logger.exception('Speech generation failed')
        return jsonify(error='Speech generation failed. Check the terminal for details and try again.'), 500
    finally:
        _generation_lock.release()


if __name__ == '__main__':
    print('Open http://127.0.0.1:7860 in your browser. Press Ctrl+C to stop.', flush=True)
    app.run(host='127.0.0.1', port=7860, threaded=True, debug=False, use_reloader=False)
