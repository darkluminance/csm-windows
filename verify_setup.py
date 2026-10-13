"""Check local inference prerequisites without downloading model weights."""
import os
import tempfile
from pathlib import Path

os.environ['NO_TORCH_COMPILE'] = '1'
os.environ['PATH'] = str(Path(__file__).resolve().parent / '.tools/ffmpeg/bin') + os.pathsep + os.environ['PATH']

import torch
import torchaudio
import triton
from generator import load_csm_1b
from check_model_access import check_model_access


def main():
    print(f'PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}')
    print(f'Triton: {triton.__version__}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    print(f'Device: {torch.cuda.get_device_name(0) if device == "cuda" else "CPU"}')
    matrix = torch.ones((32, 32), device=device, dtype=dtype)
    assert torch.all(matrix @ matrix == 32).item()
    print(f'{device} matrix operation: OK')
    from moshi.utils import quantize
    projection = torch.nn.Linear(8, 4, bias=False, device=device)
    inputs = torch.ones(1, 2, 8, device=device)
    torch.testing.assert_close(quantize.linear(projection, inputs), projection(inputs))
    print('Moshi audio linear path and bitsandbytes runtime import: OK')
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / 'check.wav')
        audio = torch.zeros(1, 2400)
        torchaudio.save(path, audio, 24000)
        restored, rate = torchaudio.load(path)
        assert rate == 24000 and restored.shape == audio.shape
        assert torchaudio.functional.resample(restored, rate, 16000).shape == (1, 1600)
    print('Project imports and WAV save/load/resampling: OK')
    from pydub import AudioSegment
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / 'check.mp3')
        AudioSegment.silent(duration=100).export(path, format='mp3')
        assert len(AudioSegment.from_file(path)) == 100
    print('FFmpeg/ffprobe MP3 encode/decode: OK')
    return 0 if check_model_access() else 1


if __name__ == '__main__':
    raise SystemExit(main())
