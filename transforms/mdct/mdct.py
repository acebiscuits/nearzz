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
from concurrent.futures import ProcessPoolExecutor, as_completed

FFMPEG_BIN_DIR = r"D:\5_year\nirs\ffmpeg\bin"
os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ["PATH"]

from pydub import AudioSegment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../dataset/test_sounds3"))

OUT_ENC_INVERT_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "invert")
OUT_DEC_INVERT_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "invert")

OUT_ENC_SHUFFLE_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "shuffle")
OUT_DEC_SHUFFLE_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "shuffle")
OUT_DEC_SHUFFLE_WK_DIR = os.path.join(BASE_DIR, "assets", "decrypted_wrong_key", "shuffle")

OUT_ENC_SIGNFLIP_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "signflip")
OUT_DEC_SIGNFLIP_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "signflip")
OUT_DEC_SIGNFLIP_WK_DIR = os.path.join(BASE_DIR, "assets", "decrypted_wrong_key", "signflip")

BASE_OUT_DEC_AFTER_CODEC_DIR = os.path.join(BASE_DIR, "assets", "decrypted_after_codec")

DEC_AFTER_CODEC_INVERT_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_mp3_64k")
DEC_AFTER_CODEC_INVERT_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_mp3_128k")
DEC_AFTER_CODEC_INVERT_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_mp3_320k")
DEC_AFTER_CODEC_INVERT_OPUS_16_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_opus_16k")
DEC_AFTER_CODEC_INVERT_OPUS_48_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_opus_48k")
DEC_AFTER_CODEC_INVERT_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_opus_96k")

DEC_AFTER_CODEC_SHUFFLE_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_mp3_64k")
DEC_AFTER_CODEC_SHUFFLE_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_mp3_128k")
DEC_AFTER_CODEC_SHUFFLE_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_mp3_320k")
DEC_AFTER_CODEC_SHUFFLE_OPUS_16_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_opus_16k")
DEC_AFTER_CODEC_SHUFFLE_OPUS_48_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_opus_48k")
DEC_AFTER_CODEC_SHUFFLE_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_opus_96k")

DEC_AFTER_CODEC_SIGNFLIP_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_mp3_64k")
DEC_AFTER_CODEC_SIGNFLIP_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_mp3_128k")
DEC_AFTER_CODEC_SIGNFLIP_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_mp3_320k")
DEC_AFTER_CODEC_SIGNFLIP_OPUS_16_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_opus_16k")
DEC_AFTER_CODEC_SIGNFLIP_OPUS_48_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_opus_48k")
DEC_AFTER_CODEC_SIGNFLIP_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_opus_96k")

KEY_PATH = os.path.join(BASE_DIR, "key.bin")
WRONG_KEY_PATH = os.path.join(BASE_DIR, "wrong_key.bin")
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
os.makedirs(DEC_AFTER_CODEC_INVERT_OPUS_16_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_INVERT_OPUS_48_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_INVERT_OPUS_96_DIR, exist_ok=True)

os.makedirs(DEC_AFTER_CODEC_SHUFFLE_MP3_64_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SHUFFLE_MP3_128_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SHUFFLE_MP3_320_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SHUFFLE_OPUS_16_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SHUFFLE_OPUS_48_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SHUFFLE_OPUS_96_DIR, exist_ok=True)

os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_MP3_64_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_MP3_128_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_MP3_320_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_OPUS_16_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_OPUS_48_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_OPUS_96_DIR, exist_ok=True)

os.makedirs(OUT_DEC_SHUFFLE_WK_DIR, exist_ok=True)
os.makedirs(OUT_DEC_SIGNFLIP_WK_DIR, exist_ok=True)

M = 2500
N = 2 * M
MDCT_WINDOW = np.sin(np.pi / N * (np.arange(N) + 0.5))
N_FILES = 1722

def clip_to_int16(sig: np.ndarray, peak: float = 0.98) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    # max_abs = np.max(np.abs(sig)) + 1e-12
    # if max_abs > 0:
    #     sig = sig / max_abs

    # sig = sig * (32767.0 * peak)
    sig = np.clip(sig, -32768, 32767)
    return sig.astype(np.int16)



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
    h = hashlib.sha512()
    h.update(key_bytes)
    h.update(str(nonce).encode("utf-8"))
    digest = h.digest()
    seed_words = np.frombuffer(digest, dtype=np.uint32)
    seed_seq = np.random.SeedSequence(seed_words, pool_size=8)
    return np.random.default_rng(seed_seq)

def build_perm(size, rng):
    perm = rng.permutation(size)
    inv_perm = np.argsort(perm)
    return perm, inv_perm

def signflip_block_with_block_mask(Y, block_index, key_bytes, base_nonce):
    block_nonce = f"{base_nonce}_block_{block_index}"
    rng = rng_from_key(key_bytes, nonce=block_nonce)
    mask = rng.choice(np.array([-1.0, 1.0]), size=len(Y)).astype(np.float64)
    return Y * mask

def process_mdct_blocks(data, modify_func, *modify_args):
    data = np.asarray(data, dtype=np.float64)
    output = np.zeros_like(data, dtype=np.float64)

    block_index = 0

    for start in range(0, len(data) - N + 1, M):
        frame = data[start:start + N] * MDCT_WINDOW
        Y = mdct4(frame)
        Y = modify_func(Y, block_index, *modify_args)
        Z = imdct4(Y)
        output[start:start + N] += Z * MDCT_WINDOW
        block_index += 1

    return output


def invert(Y, block_index):
    return Y[::-1]

def shuffle_block_with_block_perm(Y, block_index, key_bytes, base_nonce):
    block_nonce = f"{base_nonce}_block_{block_index}"
    rng = rng_from_key(key_bytes, nonce=block_nonce)
    perm, _ = build_perm(len(Y), rng)
    return Y[perm]

def unshuffle_block_with_block_perm(Y, block_index, key_bytes, base_nonce):
    block_nonce = f"{base_nonce}_block_{block_index}"
    rng = rng_from_key(key_bytes, nonce=block_nonce)
    _, inv_perm = build_perm(len(Y), rng)
    return Y[inv_perm]

def do_shuffle_enc(Y, block_index, key_bytes, base_nonce):
    return shuffle_block_with_block_perm(Y, block_index, key_bytes, base_nonce)

def do_shuffle_dec(Y, block_index, key_bytes, base_nonce):
    return unshuffle_block_with_block_perm(Y, block_index, key_bytes, base_nonce)

def do_signflip(Y, block_index, key_bytes, base_nonce):
    return signflip_block_with_block_mask(Y, block_index, key_bytes, base_nonce)

def write_wav(path, fs, sig):
    sig = clip_to_int16(sig)
    wavfile.write(path, fs, sig)



def compress_decompress_array(sig, fs, codec="mp3", bitrate="64k", target_fs=16000):
    sig = clip_to_int16(sig)

    in_buf = io.BytesIO()
    wavfile.write(in_buf, fs, sig)
    in_buf.seek(0)

    audio = AudioSegment.from_file(in_buf, format="wav")
    audio = audio.set_frame_rate(target_fs).set_channels(1)

    compressed_buf = io.BytesIO()

    if codec == "mp3":
        # audio.export(compressed_buf, format="mp3", bitrate=bitrate)
        audio.export(
            compressed_buf,
            format="mp3",
            bitrate=bitrate,
            parameters=["-b:a", bitrate]
        )
        size_bytes = compressed_buf.tell()
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
    ("opus", "16k"),
    ("opus", "48k"),
    ("opus", "96k"),
]


def get_after_codec_dir(mode: str, codec: str, bitrate: str) -> str:
    dirs = {
        ("invert", "mp3", "64k"): DEC_AFTER_CODEC_INVERT_MP3_64_DIR,
        ("invert", "mp3", "128k"): DEC_AFTER_CODEC_INVERT_MP3_128_DIR,
        ("invert", "mp3", "320k"): DEC_AFTER_CODEC_INVERT_MP3_320_DIR,
        ("invert", "opus", "16k"): DEC_AFTER_CODEC_INVERT_OPUS_16_DIR,
        ("invert", "opus", "48k"): DEC_AFTER_CODEC_INVERT_OPUS_48_DIR,
        ("invert", "opus", "96k"): DEC_AFTER_CODEC_INVERT_OPUS_96_DIR,

        ("shuffle", "mp3", "64k"): DEC_AFTER_CODEC_SHUFFLE_MP3_64_DIR,
        ("shuffle", "mp3", "128k"): DEC_AFTER_CODEC_SHUFFLE_MP3_128_DIR,
        ("shuffle", "mp3", "320k"): DEC_AFTER_CODEC_SHUFFLE_MP3_320_DIR,
        ("shuffle", "opus", "16k"): DEC_AFTER_CODEC_SHUFFLE_OPUS_16_DIR,
        ("shuffle", "opus", "48k"): DEC_AFTER_CODEC_SHUFFLE_OPUS_48_DIR,
        ("shuffle", "opus", "96k"): DEC_AFTER_CODEC_SHUFFLE_OPUS_96_DIR,

        ("signflip", "mp3", "64k"): DEC_AFTER_CODEC_SIGNFLIP_MP3_64_DIR,
        ("signflip", "mp3", "128k"): DEC_AFTER_CODEC_SIGNFLIP_MP3_128_DIR,
        ("signflip", "mp3", "320k"): DEC_AFTER_CODEC_SIGNFLIP_MP3_320_DIR,
        ("signflip", "opus", "16k"): DEC_AFTER_CODEC_SIGNFLIP_OPUS_16_DIR,
        ("signflip", "opus", "48k"): DEC_AFTER_CODEC_SIGNFLIP_OPUS_48_DIR,
        ("signflip", "opus", "96k"): DEC_AFTER_CODEC_SIGNFLIP_OPUS_96_DIR,
    }
    return dirs[(mode, codec, bitrate)]

def process_one_file(i):
    in_path = os.path.join(DATASET_DIR, f"test_sound{i}.wav")

    fs, data = wavfile.read(in_path)

    if data.ndim > 1:
        data = data[:, 0]

    x = data.astype(np.float64)

    # invert
    # inv_audio = process_mdct_blocks(x, invert)
    # inv_dec = process_mdct_blocks(inv_audio, invert)

    # write_wav(os.path.join(OUT_ENC_INVERT_DIR, f"test_sound{i}_invert_enc.wav"), fs, inv_audio)
    # write_wav(os.path.join(OUT_DEC_INVERT_DIR, f"test_sound{i}_invert_dec.wav"), fs, inv_dec)

    # shuffle
    shuffle_nonce = f"test_sound{i}_shuffle"
    shuf_audio = process_mdct_blocks(x, do_shuffle_enc, key_bytes, shuffle_nonce)
    shuf_dec = process_mdct_blocks(shuf_audio, do_shuffle_dec, wrong_key_bytes, shuffle_nonce)

    write_wav(os.path.join(OUT_ENC_SHUFFLE_DIR, f"test_sound{i}_shuffle_enc.wav"), fs, shuf_audio)
    write_wav(os.path.join(OUT_DEC_SHUFFLE_WK_DIR, f"test_sound{i}_shuffle_dec.wav"), fs, shuf_dec)

    # signflip
    signflip_nonce = f"test_sound{i}_signflip"
    sign_audio = process_mdct_blocks(x, do_signflip, key_bytes, signflip_nonce)
    sign_dec = process_mdct_blocks(sign_audio, do_signflip, wrong_key_bytes, signflip_nonce)

    write_wav(os.path.join(OUT_ENC_SIGNFLIP_DIR, f"test_sound{i}_signflip_enc.wav"), fs, sign_audio)
    write_wav(os.path.join(OUT_DEC_SIGNFLIP_WK_DIR, f"test_sound{i}_signflip_dec.wav"), fs, sign_dec)

    # ===== decrypted_after_codec datasets =====

    # invert

    # for codec, bitrate in CODEC_VARIANTS:
    #     inv_audio_after_codec = compress_decompress_array(inv_audio, fs, codec=codec, bitrate=bitrate)
    #     inv_dec_after_codec = process_mdct_blocks(inv_audio_after_codec, invert)
    #     out_dir = get_after_codec_dir("invert", codec, bitrate)
    #     out_name = f"test_sound{i}_invert_dec.wav"
    #     write_wav(os.path.join(out_dir, out_name), fs, inv_dec_after_codec)

    # # shuffle

    # for codec, bitrate in CODEC_VARIANTS:
    #     shuf_audio_after_codec = compress_decompress_array(shuf_audio, fs, codec=codec, bitrate=bitrate)
    #     shuf_dec_after_codec = process_mdct_blocks(shuf_audio_after_codec, do_shuffle_dec, key_bytes, shuffle_nonce)
    #     out_dir = get_after_codec_dir("shuffle", codec, bitrate)
    #     out_name = f"test_sound{i}_shuffle_dec.wav"
    #     write_wav(os.path.join(out_dir, out_name), fs, shuf_dec_after_codec)

    # # signflip

    # for codec, bitrate in CODEC_VARIANTS:
    #     sign_audio_after_codec = compress_decompress_array(sign_audio, fs, codec=codec, bitrate=bitrate)
    #     sign_dec_after_codec = process_mdct_blocks(sign_audio_after_codec, do_signflip, key_bytes, signflip_nonce)
    #     out_dir = get_after_codec_dir("signflip", codec, bitrate)
    #     out_name = f"test_sound{i}_signflip_dec.wav"
    #     write_wav(os.path.join(out_dir, out_name), fs, sign_dec_after_codec)

    return f"done: test_sound{i}.wav"

key_bytes = load_key_bytes(KEY_PATH)
wrong_key_bytes = load_key_bytes(WRONG_KEY_PATH)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_one_file, i) for i in range(1, N_FILES + 1)]

        for future in as_completed(futures):
            print(future.result())