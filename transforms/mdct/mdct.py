import numpy as np
from scipy.io import wavfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import hashlib
import numpy as np
import io

FFMPEG_BIN_DIR = r"D:\5_year\nirs\ffmpeg\bin"
os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ["PATH"]

from pydub import AudioSegment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../dataset/test_sounds"))

OUT_ENC_INVERT_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "invert")
OUT_DEC_INVERT_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "invert")

OUT_ENC_SHUFFLE_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "shuffle")
OUT_DEC_SHUFFLE_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "shuffle")

OUT_ENC_SIGNFLIP_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "signflip")
OUT_DEC_SIGNFLIP_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "signflip")

BASE_OUT_DEC_AFTER_CODEC_DIR = os.path.join(BASE_DIR, "assets", "decrypted_after_codec")

DEC_AFTER_CODEC_INVERT_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_mp3_64k")
DEC_AFTER_CODEC_INVERT_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_mp3_128k")
DEC_AFTER_CODEC_INVERT_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_mp3_320k")
DEC_AFTER_CODEC_INVERT_OPUS_32_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_opus_32k")
DEC_AFTER_CODEC_INVERT_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_opus_96k")

DEC_AFTER_CODEC_SHUFFLE_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_mp3_64k")
DEC_AFTER_CODEC_SHUFFLE_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_mp3_128k")
DEC_AFTER_CODEC_SHUFFLE_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_mp3_320k")
DEC_AFTER_CODEC_SHUFFLE_OPUS_32_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_opus_32k")
DEC_AFTER_CODEC_SHUFFLE_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_opus_96k")

DEC_AFTER_CODEC_SIGNFLIP_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_mp3_64k")
DEC_AFTER_CODEC_SIGNFLIP_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_mp3_128k")
DEC_AFTER_CODEC_SIGNFLIP_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_mp3_320k")
DEC_AFTER_CODEC_SIGNFLIP_OPUS_32_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_opus_32k")
DEC_AFTER_CODEC_SIGNFLIP_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_opus_96k")

KEY_PATH = os.path.join(BASE_DIR, "key.bin")
AudioSegment.converter = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")

os.makedirs(OUT_ENC_INVERT_DIR, exist_ok=True)
os.makedirs(OUT_DEC_INVERT_DIR, exist_ok=True)
os.makedirs(OUT_ENC_SHUFFLE_DIR, exist_ok=True)
os.makedirs(OUT_DEC_SHUFFLE_DIR, exist_ok=True)
os.makedirs(OUT_ENC_SIGNFLIP_DIR, exist_ok=True)
os.makedirs(OUT_DEC_SIGNFLIP_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_INVERT_MP3_64_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_INVERT_MP3_128_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_INVERT_MP3_320_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_INVERT_OPUS_32_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_INVERT_OPUS_96_DIR, exist_ok=True)

os.makedirs(DEC_AFTER_CODEC_SHUFFLE_MP3_64_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SHUFFLE_MP3_128_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SHUFFLE_MP3_320_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SHUFFLE_OPUS_32_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SHUFFLE_OPUS_96_DIR, exist_ok=True)

os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_MP3_64_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_MP3_128_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_MP3_320_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_OPUS_32_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_OPUS_96_DIR, exist_ok=True)

M = 2500
N = 2 * M
hop = M
N_FILES = 23


def mdct4(x):
    N = x.shape[0]
    M = N // 2
    N4 = N // 4
    
    rot = np.roll(x, N4)
    rot[:N4] = -rot[:N4]
    t = np.arange(0, N4)
    w = np.exp(-1j*2*np.pi*(t + 1./8.) / N)
    c = np.take(rot,2*t) - np.take(rot, N-2*t-1) - 1j * (np.take(rot, M+2*t) - np.take(rot,M-2*t-1))
    c = (2./np.sqrt(N)) * w * np.fft.fft(0.5 * c * w, N4)
    y = np.zeros(M)
    y[2*t] = np.real(c[t])
    y[M-2*t-1] = -np.imag(c[t])
    return y

def imdct4(x):
    N = x.shape[0]
    M = N // 2
    N2 = N*2
    
    t = np.arange(0,M)
    w = np.exp(-1j*2*np.pi*(t + 1./8.) / N2)
    c = np.take(x,2*t) + 1j * np.take(x,N-2*t-1)
    c = 0.5 * w * c
    c = np.fft.fft(c,M)
    c = ((8 / np.sqrt(N2))*w)*c
    
    rot = np.zeros(N2)
    
    rot[2*t] = np.real(c[t])
    rot[N+2*t] = np.imag(c[t])
    
    t = np.arange(1,N2,2)
    rot[t] = -rot[N2-t-1]
    
    t = np.arange(0,3*M)
    y = np.zeros(N2)
    y[t] = rot[t+M]
    t = np.arange(3*M,N2)
    y[t] = -rot[t-3*M]
    return y






def load_key_bytes(path):
    with open(path, "rb") as f:
        return f.read()
    
def rng_from_key(key_bytes, nonce):
    h = hashlib.sha256()
    h.update(key_bytes)
    h.update(str(nonce).encode("utf-8"))
    seed = int.from_bytes(h.digest()[:8], "big")
    return np.random.default_rng(seed)

def build_perm(size, rng):
    perm = rng.permutation(size)
    inv_perm = np.argsort(perm)
    return perm, inv_perm

def sign_mask_vector(size, rng):
    return rng.choice(np.array([-1.0, 1.0]), size=size).astype(np.float64)

def process_mdct_blocks(data, modify_func=None):
    n = np.arange(N)
    window = np.sin(np.pi / N * (n + 0.5))

    original_len = len(data)
    pad = (N - (len(data) % hop)) % hop
    data = np.pad(data, (0, pad))

    output = np.zeros_like(data, dtype=np.float64)

    for start in range(0, len(data) - N + 1, hop):
        frame = data[start:start + N] * window
        Y = mdct4(frame)

        if modify_func is not None:
            Y = modify_func(Y)

        Z = imdct4(Y)
        output[start:start + N] += Z * window

    return output[:original_len]


def invert(Y):
    return Y[::-1]

def shuffle_with_perm(Y, perm):
    return Y[perm]

def unshuffle_with_inv_perm(Y, inv_perm):
    return Y[inv_perm]





key_bytes = load_key_bytes(KEY_PATH)






def write_wav(path, fs, sig):
    # sig = np.asarray(sig, dtype=np.float32)
    # sig = np.clip(sig, -32768.0, 32767.0)
    wavfile.write(path, fs, sig.astype(np.int16))



def compress_decompress_array(sig, fs, codec="mp3", bitrate="64k", target_fs=8000):
    sig = np.asarray(sig, dtype=np.int16)

    in_buf = io.BytesIO()
    wavfile.write(in_buf, fs, sig)
    in_buf.seek(0)

    audio = AudioSegment.from_file(in_buf, format="wav")
    audio = audio.set_frame_rate(target_fs).set_channels(1)

    compressed_buf = io.BytesIO()

    if codec == "mp3":
        audio.export(compressed_buf, format="mp3", bitrate=bitrate)
        compressed_buf.seek(0)
        decoded_audio = AudioSegment.from_file(compressed_buf, format="mp3")

    elif codec == "opus":
        audio.export(compressed_buf, format="ogg", codec="libopus", bitrate=bitrate)
        compressed_buf.seek(0)
        decoded_audio = AudioSegment.from_file(compressed_buf, format="ogg")

    else:
        raise ValueError(f"unsupported codec: {codec}")

    decoded_audio = decoded_audio.set_frame_rate(target_fs).set_channels(1)
    decoded = np.array(decoded_audio.get_array_of_samples(), dtype=np.float64)

    return decoded


CODEC_VARIANTS = [
    ("mp3", "64k"),
    ("mp3", "128k"),
    ("mp3", "320k"),
    ("opus", "32k"),
    ("opus", "96k"),
]


def get_after_codec_dir(mode: str, codec: str, bitrate: str) -> str:
    dirs = {
        ("invert", "mp3", "64k"): DEC_AFTER_CODEC_INVERT_MP3_64_DIR,
        ("invert", "mp3", "128k"): DEC_AFTER_CODEC_INVERT_MP3_128_DIR,
        ("invert", "mp3", "320k"): DEC_AFTER_CODEC_INVERT_MP3_320_DIR,
        ("invert", "opus", "32k"): DEC_AFTER_CODEC_INVERT_OPUS_32_DIR,
        ("invert", "opus", "96k"): DEC_AFTER_CODEC_INVERT_OPUS_96_DIR,

        ("shuffle", "mp3", "64k"): DEC_AFTER_CODEC_SHUFFLE_MP3_64_DIR,
        ("shuffle", "mp3", "128k"): DEC_AFTER_CODEC_SHUFFLE_MP3_128_DIR,
        ("shuffle", "mp3", "320k"): DEC_AFTER_CODEC_SHUFFLE_MP3_320_DIR,
        ("shuffle", "opus", "32k"): DEC_AFTER_CODEC_SHUFFLE_OPUS_32_DIR,
        ("shuffle", "opus", "96k"): DEC_AFTER_CODEC_SHUFFLE_OPUS_96_DIR,

        ("signflip", "mp3", "64k"): DEC_AFTER_CODEC_SIGNFLIP_MP3_64_DIR,
        ("signflip", "mp3", "128k"): DEC_AFTER_CODEC_SIGNFLIP_MP3_128_DIR,
        ("signflip", "mp3", "320k"): DEC_AFTER_CODEC_SIGNFLIP_MP3_320_DIR,
        ("signflip", "opus", "32k"): DEC_AFTER_CODEC_SIGNFLIP_OPUS_32_DIR,
        ("signflip", "opus", "96k"): DEC_AFTER_CODEC_SIGNFLIP_OPUS_96_DIR,
    }
    return dirs[(mode, codec, bitrate)]



for i in range(1, N_FILES + 1):
    in_path = os.path.join(DATASET_DIR, f"test_sound{i}.wav")

    fs, data = wavfile.read(in_path)

    if data.ndim > 1:
        data = data[:, 0]

    x = data.astype(np.float64)

    # invert
    inv_audio = process_mdct_blocks(x, invert)
    inv_dec = process_mdct_blocks(inv_audio, invert)

    write_wav(os.path.join(OUT_ENC_INVERT_DIR, f"test_sound{i}_invert_enc.wav"), fs, inv_audio)
    write_wav(os.path.join(OUT_DEC_INVERT_DIR, f"test_sound{i}_invert_dec.wav"), fs, inv_dec)

    # shuffle
    shuffle_nonce = f"test_sound{i}_shuffle"
    shuffle_rng = rng_from_key(key_bytes, nonce=shuffle_nonce)
    perm, inv_perm = build_perm(M, shuffle_rng)

    shuf_audio = process_mdct_blocks(x, lambda Y: shuffle_with_perm(Y, perm))
    shuf_dec = process_mdct_blocks(shuf_audio, lambda Y: unshuffle_with_inv_perm(Y, inv_perm))

    write_wav(os.path.join(OUT_ENC_SHUFFLE_DIR, f"test_sound{i}_shuffle_enc.wav"), fs, shuf_audio)
    write_wav(os.path.join(OUT_DEC_SHUFFLE_DIR, f"test_sound{i}_shuffle_dec.wav"), fs, shuf_dec)

    # signflip
    signflip_nonce = f"test_sound{i}_signflip"

    signflip_rng_enc = rng_from_key(key_bytes, nonce=signflip_nonce)
    sign_audio = process_mdct_blocks(x, lambda Y: Y * sign_mask_vector(len(Y), signflip_rng_enc))

    signflip_rng_dec = rng_from_key(key_bytes, nonce=signflip_nonce)
    sign_dec = process_mdct_blocks(sign_audio, lambda Y: Y * sign_mask_vector(len(Y), signflip_rng_dec))

    write_wav(os.path.join(OUT_ENC_SIGNFLIP_DIR, f"test_sound{i}_signflip_enc.wav"), fs, sign_audio)
    write_wav(os.path.join(OUT_DEC_SIGNFLIP_DIR, f"test_sound{i}_signflip_dec.wav"), fs, sign_dec)

    # ===== decrypted_after_codec datasets =====

    # invert
    inv_audio = process_mdct_blocks(x, invert)

    for codec, bitrate in CODEC_VARIANTS:
        inv_audio_after_codec = compress_decompress_array(inv_audio, fs, codec=codec, bitrate=bitrate)
        inv_dec_after_codec = process_mdct_blocks(inv_audio_after_codec, invert)
        out_dir = get_after_codec_dir("invert", codec, bitrate)
        out_name = f"test_sound{i}_invert_dec.wav"
        write_wav(os.path.join(out_dir, out_name), fs, inv_dec_after_codec)

    # shuffle
    shuffle_nonce = f"test_sound{i}_shuffle"
    shuffle_rng = rng_from_key(key_bytes, nonce=shuffle_nonce)
    perm, inv_perm = build_perm(M, shuffle_rng)
    shuf_audio = process_mdct_blocks(x, lambda Y: shuffle_with_perm(Y, perm))

    for codec, bitrate in CODEC_VARIANTS:
        shuf_audio_after_codec = compress_decompress_array(shuf_audio, fs, codec=codec, bitrate=bitrate)
        shuf_dec_after_codec = process_mdct_blocks(shuf_audio_after_codec, lambda Y: unshuffle_with_inv_perm(Y, inv_perm))
        out_dir = get_after_codec_dir("shuffle", codec, bitrate)
        out_name = f"test_sound{i}_shuffle_dec.wav"
        write_wav(os.path.join(out_dir, out_name), fs, shuf_dec_after_codec)

    # signflip
    signflip_nonce = f"test_sound{i}_signflip"
    signflip_rng_enc = rng_from_key(key_bytes, nonce=signflip_nonce)
    sign_audio = process_mdct_blocks(x, lambda Y: Y * sign_mask_vector(len(Y), signflip_rng_enc))

    for codec, bitrate in CODEC_VARIANTS:
        sign_audio_after_codec = compress_decompress_array(sign_audio, fs, codec=codec, bitrate=bitrate)
        signflip_rng_dec = rng_from_key(key_bytes, nonce=signflip_nonce)
        sign_dec_after_codec = process_mdct_blocks(sign_audio_after_codec, lambda Y: Y * sign_mask_vector(len(Y), signflip_rng_dec))
        out_dir = get_after_codec_dir("signflip", codec, bitrate)
        out_name = f"test_sound{i}_signflip_dec.wav"
        write_wav(os.path.join(out_dir, out_name), fs, sign_dec_after_codec)