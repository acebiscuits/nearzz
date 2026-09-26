from scipy.io import wavfile
import numpy as np
from scipy import signal
from scipy.signal import kaiser_beta
import os
import hashlib
import io
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

FFMPEG_BIN_DIR = r"D:\5_year\nirs\ffmpeg\bin"
os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ["PATH"]

from pydub import AudioSegment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../dataset/test_sounds3"))

BASE_OUT_ENC_SIMPLE_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "simple")
BASE_OUT_DEC_SIMPLE_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "simple")

BASE_OUT_ENC_SPLIT_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "split")
BASE_OUT_DEC_SPLIT_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "split")

BASE_OUT_ENC_SIMPLE_SIGN_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "simple_sign")
BASE_OUT_DEC_SIMPLE_SIGN_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "simple_sign")

BASE_OUT_ENC_SPLIT_SIGN_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "split_sign")
BASE_OUT_DEC_SPLIT_SIGN_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "split_sign")

BASE_OUT_DEC_AFTER_CODEC_DIR = os.path.join(BASE_DIR, "assets", "decrypted_after_codec")

DEC_AFTER_CODEC_SIMPLE_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_mp3_64k")
DEC_AFTER_CODEC_SIMPLE_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_mp3_128k")
DEC_AFTER_CODEC_SIMPLE_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_mp3_320k")
DEC_AFTER_CODEC_SIMPLE_OPUS_32_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_opus_32k")
DEC_AFTER_CODEC_SIMPLE_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_opus_96k")

DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_sign_mp3_64k")
DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_sign_mp3_128k")
DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_sign_mp3_320k")
DEC_AFTER_CODEC_SIMPLE_SIGN_OPUS_32_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_sign_opus_32k")
DEC_AFTER_CODEC_SIMPLE_SIGN_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "simple_sign_opus_96k")

DEC_AFTER_CODEC_SPLIT_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_mp3_64k")
DEC_AFTER_CODEC_SPLIT_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_mp3_128k")
DEC_AFTER_CODEC_SPLIT_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_mp3_320k")
DEC_AFTER_CODEC_SPLIT_OPUS_32_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_opus_32k")
DEC_AFTER_CODEC_SPLIT_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_opus_96k")

DEC_AFTER_CODEC_SPLIT_SIGN_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_sign_mp3_64k")
DEC_AFTER_CODEC_SPLIT_SIGN_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_sign_mp3_128k")
DEC_AFTER_CODEC_SPLIT_SIGN_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_sign_mp3_320k")
DEC_AFTER_CODEC_SPLIT_SIGN_OPUS_32_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_sign_opus_32k")
DEC_AFTER_CODEC_SPLIT_SIGN_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "split_sign_opus_96k")

KEY_PATH = os.path.join(BASE_DIR, "key.bin")
AudioSegment.converter = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")

BLOCK_SIZE = 80
MAX_WORKERS = 8

CODEC_VARIANTS = [
    ("mp3", "64k"),
    ("mp3", "128k"),
    ("mp3", "320k"),
    ("opus", "32k"),
    ("opus", "96k"),
]


def ensure_dirs():
    dirs = [
        BASE_OUT_ENC_SIMPLE_DIR, BASE_OUT_DEC_SIMPLE_DIR,
        BASE_OUT_ENC_SPLIT_DIR, BASE_OUT_DEC_SPLIT_DIR,
        BASE_OUT_ENC_SIMPLE_SIGN_DIR, BASE_OUT_DEC_SIMPLE_SIGN_DIR,
        BASE_OUT_ENC_SPLIT_SIGN_DIR, BASE_OUT_DEC_SPLIT_SIGN_DIR,

        DEC_AFTER_CODEC_SIMPLE_MP3_64_DIR, DEC_AFTER_CODEC_SIMPLE_MP3_128_DIR, DEC_AFTER_CODEC_SIMPLE_MP3_320_DIR,
        DEC_AFTER_CODEC_SIMPLE_OPUS_32_DIR, DEC_AFTER_CODEC_SIMPLE_OPUS_96_DIR,

        DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_64_DIR, DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_128_DIR, DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_320_DIR,
        DEC_AFTER_CODEC_SIMPLE_SIGN_OPUS_32_DIR, DEC_AFTER_CODEC_SIMPLE_SIGN_OPUS_96_DIR,

        DEC_AFTER_CODEC_SPLIT_MP3_64_DIR, DEC_AFTER_CODEC_SPLIT_MP3_128_DIR, DEC_AFTER_CODEC_SPLIT_MP3_320_DIR,
        DEC_AFTER_CODEC_SPLIT_OPUS_32_DIR, DEC_AFTER_CODEC_SPLIT_OPUS_96_DIR,

        DEC_AFTER_CODEC_SPLIT_SIGN_MP3_64_DIR, DEC_AFTER_CODEC_SPLIT_SIGN_MP3_128_DIR, DEC_AFTER_CODEC_SPLIT_SIGN_MP3_320_DIR,
        DEC_AFTER_CODEC_SPLIT_SIGN_OPUS_32_DIR, DEC_AFTER_CODEC_SPLIT_SIGN_OPUS_96_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def list_input_files(dataset_dir: str):
    rx = re.compile(r"^test_sound(\d+)\.wav$", re.IGNORECASE)
    items = []
    for name in os.listdir(dataset_dir):
        m = rx.match(name)
        if m:
            items.append((int(m.group(1)), os.path.join(dataset_dir, name)))
    items.sort(key=lambda x: x[0])
    return items


def to_int16_safe(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    sig = np.clip(sig, -32768, 32767)
    return sig.astype(np.int16)


def to_mono_float(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(np.float64)


def write_wav(path, fs, sig):
    sig = to_int16_safe(sig)
    wavfile.write(path, fs, sig)


def dc_remove(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def design_fir_lowpass(length_sec: float, cutoff_hz: float, fs: float, atten_db: float) -> np.ndarray:
    max_len = 2047
    taps_len = int(2 * round(fs * length_sec) + 1)
    taps_len = min(taps_len, max_len)
    if taps_len < 3:
        taps_len = 3

    beta = kaiser_beta(atten_db)
    taps = signal.firwin(taps_len, cutoff=cutoff_hz, window=("kaiser", beta), fs=fs)
    return taps.astype(float)


def inverter_offline(x: np.ndarray, fs: float, freq_prefilter: float, freq_shift: float, freq_postfilter: float,
                     length_sec: float = 0.0024, atten_db: float = 60.0) -> np.ndarray:
    n = np.arange(len(x), dtype=float)
    omega = 2.0 * np.pi * freq_shift / fs
    carrier = np.cos(omega * n)

    taps_pre = design_fir_lowpass(length_sec, freq_prefilter, fs, atten_db)
    x_pre = signal.lfilter(taps_pre, [1.0], x)

    mixed = x_pre * carrier

    taps_post = design_fir_lowpass(length_sec, freq_postfilter, fs, atten_db)
    return signal.lfilter(taps_post, [1.0], mixed)


def simple_process(x: np.ndarray, fs: int, prefilter_cutoff: float, shift_freq: float, postfilter_cutoff: float) -> np.ndarray:
    return 1.8 * inverter_offline(
        x, fs,
        freq_prefilter=prefilter_cutoff,
        freq_shift=shift_freq,
        freq_postfilter=postfilter_cutoff,
    )


def split_process(x: np.ndarray, fs: int,
                  band1_prefilter_cutoff: float, band1_shift_freq: float, band1_postfilter_cutoff: float,
                  band2_prefilter_cutoff: float, band2_shift_freq: float, band2_postfilter_cutoff: float) -> np.ndarray:
    y1 = inverter_offline(
        x, fs,
        freq_prefilter=band1_prefilter_cutoff,
        freq_shift=band1_shift_freq,
        freq_postfilter=band1_postfilter_cutoff,
    )

    y2 = inverter_offline(
        x, fs,
        freq_prefilter=band2_prefilter_cutoff,
        freq_shift=band2_shift_freq,
        freq_postfilter=band2_postfilter_cutoff,
    )

    return 1.8 * (y1 + y2)


def load_key_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def rng_from_key(key_bytes, nonce):
    h = hashlib.sha256()
    h.update(key_bytes)
    h.update(str(nonce).encode("utf-8"))
    seed = int.from_bytes(h.digest()[:8], "big")
    return np.random.default_rng(seed)


def sign_mask_stream(n_samples, rng, block_size=80):
    n_blocks = (n_samples + block_size - 1) // block_size
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_blocks)
    return np.repeat(signs, block_size)[:n_samples]


def simple_sign_process(x, fs, nonce, prefilter_cutoff, shift_freq, postfilter_cutoff, block_size, decrypt, key_bytes):
    rng = rng_from_key(key_bytes, nonce=nonce)
    if not decrypt:
        y = simple_process(x, fs, prefilter_cutoff, shift_freq, postfilter_cutoff)
        mask = sign_mask_stream(len(y), rng=rng, block_size=block_size)
        return y * mask

    mask = sign_mask_stream(len(x), rng=rng, block_size=block_size)
    y = x * mask
    return simple_process(y, fs, prefilter_cutoff, shift_freq, postfilter_cutoff)


def split_sign_process(x, fs, nonce,
                       band1_prefilter_cutoff, band1_shift_freq, band1_postfilter_cutoff,
                       band2_prefilter_cutoff, band2_shift_freq, band2_postfilter_cutoff,
                       block_size, decrypt, key_bytes):
    rng = rng_from_key(key_bytes, nonce=nonce)
    if not decrypt:
        y = split_process(
            x, fs,
            band1_prefilter_cutoff=band1_prefilter_cutoff,
            band1_shift_freq=band1_shift_freq,
            band1_postfilter_cutoff=band1_postfilter_cutoff,
            band2_prefilter_cutoff=band2_prefilter_cutoff,
            band2_shift_freq=band2_shift_freq,
            band2_postfilter_cutoff=band2_postfilter_cutoff,
        )
        mask = sign_mask_stream(len(y), rng=rng, block_size=block_size)
        return y * mask

    mask = sign_mask_stream(len(x), rng=rng, block_size=block_size)
    y = x * mask
    return split_process(
        y, fs,
        band1_prefilter_cutoff=band1_prefilter_cutoff,
        band1_shift_freq=band1_shift_freq,
        band1_postfilter_cutoff=band1_postfilter_cutoff,
        band2_prefilter_cutoff=band2_prefilter_cutoff,
        band2_shift_freq=band2_shift_freq,
        band2_postfilter_cutoff=band2_postfilter_cutoff,
    )


def compress_decompress_array(sig, fs, codec="mp3", bitrate="64k", target_fs=16000):
    sig = to_int16_safe(sig)

    in_buf = io.BytesIO()
    wavfile.write(in_buf, fs, sig)
    in_buf.seek(0)

    audio = AudioSegment.from_file(in_buf, format="wav")
    audio = audio.set_frame_rate(target_fs).set_channels(1)

    compressed_buf = io.BytesIO()

    if codec == "mp3":
        audio.export(compressed_buf, format="mp3", bitrate=bitrate, parameters=["-b:a", bitrate])
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


def get_after_codec_dir(mode: str, codec: str, bitrate: str) -> str:
    dirs = {
        ("simple", "mp3", "64k"): DEC_AFTER_CODEC_SIMPLE_MP3_64_DIR,
        ("simple", "mp3", "128k"): DEC_AFTER_CODEC_SIMPLE_MP3_128_DIR,
        ("simple", "mp3", "320k"): DEC_AFTER_CODEC_SIMPLE_MP3_320_DIR,
        ("simple", "opus", "32k"): DEC_AFTER_CODEC_SIMPLE_OPUS_32_DIR,
        ("simple", "opus", "96k"): DEC_AFTER_CODEC_SIMPLE_OPUS_96_DIR,

        ("simple_sign", "mp3", "64k"): DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_64_DIR,
        ("simple_sign", "mp3", "128k"): DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_128_DIR,
        ("simple_sign", "mp3", "320k"): DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_320_DIR,
        ("simple_sign", "opus", "32k"): DEC_AFTER_CODEC_SIMPLE_SIGN_OPUS_32_DIR,
        ("simple_sign", "opus", "96k"): DEC_AFTER_CODEC_SIMPLE_SIGN_OPUS_96_DIR,

        ("split", "mp3", "64k"): DEC_AFTER_CODEC_SPLIT_MP3_64_DIR,
        ("split", "mp3", "128k"): DEC_AFTER_CODEC_SPLIT_MP3_128_DIR,
        ("split", "mp3", "320k"): DEC_AFTER_CODEC_SPLIT_MP3_320_DIR,
        ("split", "opus", "32k"): DEC_AFTER_CODEC_SPLIT_OPUS_32_DIR,
        ("split", "opus", "96k"): DEC_AFTER_CODEC_SPLIT_OPUS_96_DIR,

        ("split_sign", "mp3", "64k"): DEC_AFTER_CODEC_SPLIT_SIGN_MP3_64_DIR,
        ("split_sign", "mp3", "128k"): DEC_AFTER_CODEC_SPLIT_SIGN_MP3_128_DIR,
        ("split_sign", "mp3", "320k"): DEC_AFTER_CODEC_SPLIT_SIGN_MP3_320_DIR,
        ("split_sign", "opus", "32k"): DEC_AFTER_CODEC_SPLIT_SIGN_OPUS_32_DIR,
        ("split_sign", "opus", "96k"): DEC_AFTER_CODEC_SPLIT_SIGN_OPUS_96_DIR,
    }
    return dirs[(mode, codec, bitrate)]


def process_one_file(task):
    i, in_path, key_bytes = task

    fs, data = wavfile.read(in_path)
    x = dc_remove(to_mono_float(data))

    fs_norm = 16000
    simple_params = {
        "prefilter_cutoff": 2632.0 * fs_norm / 8000.0,
        "shift_freq": 2632.0 * fs_norm / 8000.0,
        "postfilter_cutoff": 2632.0 * fs_norm / 8000.0,
    }
    split_params = {
        "band1_prefilter_cutoff": 500.0 * fs_norm / 8000.0,
        "band1_shift_freq": 500.0 * fs_norm / 8000.0,
        "band1_postfilter_cutoff": 500.0 * fs_norm / 8000.0,
        "band2_prefilter_cutoff": 2632.0 * fs_norm / 8000.0,
        "band2_shift_freq": 2632.0 * fs_norm / 8000.0,
        "band2_postfilter_cutoff": 2632.0 * fs_norm / 8000.0,
    }

    # simple
    simple_enc = simple_process(
        x, fs,
        prefilter_cutoff=simple_params["prefilter_cutoff"],
        shift_freq=simple_params["shift_freq"],
        postfilter_cutoff=simple_params["postfilter_cutoff"]
    )
    simple_dec = simple_process(
        simple_enc, fs,
        prefilter_cutoff=simple_params["prefilter_cutoff"],
        shift_freq=simple_params["shift_freq"],
        postfilter_cutoff=simple_params["postfilter_cutoff"]
    )
    write_wav(os.path.join(BASE_OUT_ENC_SIMPLE_DIR, f"test_sound{i}_simple_enc.wav"), fs, simple_enc)
    write_wav(os.path.join(BASE_OUT_DEC_SIMPLE_DIR, f"test_sound{i}_simple_dec.wav"), fs, simple_dec)

    # simple_sign
    simple_sign_nonce = f"test_sound{i}_simple_sign"
    simple_sign_enc = simple_sign_process(
        x, fs, nonce=simple_sign_nonce,
        prefilter_cutoff=simple_params["prefilter_cutoff"],
        shift_freq=simple_params["shift_freq"],
        postfilter_cutoff=simple_params["postfilter_cutoff"],
        block_size=BLOCK_SIZE,
        decrypt=False,
        key_bytes=key_bytes
    )
    simple_sign_dec = simple_sign_process(
        simple_sign_enc, fs, nonce=simple_sign_nonce,
        prefilter_cutoff=simple_params["prefilter_cutoff"],
        shift_freq=simple_params["shift_freq"],
        postfilter_cutoff=simple_params["postfilter_cutoff"],
        block_size=BLOCK_SIZE,
        decrypt=True,
        key_bytes=key_bytes
    )
    write_wav(os.path.join(BASE_OUT_ENC_SIMPLE_SIGN_DIR, f"test_sound{i}_simple_sign_enc.wav"), fs, simple_sign_enc)
    write_wav(os.path.join(BASE_OUT_DEC_SIMPLE_SIGN_DIR, f"test_sound{i}_simple_sign_dec.wav"), fs, simple_sign_dec)

    # split
    split_enc = split_process(
        x, fs,
        band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"],
        band1_shift_freq=split_params["band1_shift_freq"],
        band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"],
        band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"],
        band2_shift_freq=split_params["band2_shift_freq"],
        band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"]
    )
    split_dec = split_process(
        split_enc, fs,
        band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"],
        band1_shift_freq=split_params["band1_shift_freq"],
        band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"],
        band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"],
        band2_shift_freq=split_params["band2_shift_freq"],
        band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"]
    )
    write_wav(os.path.join(BASE_OUT_ENC_SPLIT_DIR, f"test_sound{i}_split_enc.wav"), fs, split_enc)
    write_wav(os.path.join(BASE_OUT_DEC_SPLIT_DIR, f"test_sound{i}_split_dec.wav"), fs, split_dec)

    # split_sign
    split_sign_nonce = f"test_sound{i}_split_sign"
    split_sign_enc = split_sign_process(
        x, fs, nonce=split_sign_nonce,
        band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"],
        band1_shift_freq=split_params["band1_shift_freq"],
        band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"],
        band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"],
        band2_shift_freq=split_params["band2_shift_freq"],
        band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"],
        block_size=BLOCK_SIZE,
        decrypt=False,
        key_bytes=key_bytes
    )
    split_sign_dec = split_sign_process(
        split_sign_enc, fs, nonce=split_sign_nonce,
        band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"],
        band1_shift_freq=split_params["band1_shift_freq"],
        band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"],
        band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"],
        band2_shift_freq=split_params["band2_shift_freq"],
        band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"],
        block_size=BLOCK_SIZE,
        decrypt=True,
        key_bytes=key_bytes
    )
    write_wav(os.path.join(BASE_OUT_ENC_SPLIT_SIGN_DIR, f"test_sound{i}_split_sign_enc.wav"), fs, split_sign_enc)
    write_wav(os.path.join(BASE_OUT_DEC_SPLIT_SIGN_DIR, f"test_sound{i}_split_sign_dec.wav"), fs, split_sign_dec)

    # after codec: simple
    for codec, bitrate in CODEC_VARIANTS:
        simple_enc_after_codec = compress_decompress_array(simple_enc, fs, codec=codec, bitrate=bitrate)
        simple_dec_after_codec = simple_process(
            simple_enc_after_codec, fs,
            prefilter_cutoff=simple_params["prefilter_cutoff"],
            shift_freq=simple_params["shift_freq"],
            postfilter_cutoff=simple_params["postfilter_cutoff"]
        )
        out_dir = get_after_codec_dir("simple", codec, bitrate)
        write_wav(os.path.join(out_dir, f"test_sound{i}_simple_dec.wav"), fs, simple_dec_after_codec)

    # after codec: simple_sign
    for codec, bitrate in CODEC_VARIANTS:
        simple_sign_enc_after_codec = compress_decompress_array(simple_sign_enc, fs, codec=codec, bitrate=bitrate)
        simple_sign_dec_after_codec = simple_sign_process(
            simple_sign_enc_after_codec, fs, nonce=simple_sign_nonce,
            prefilter_cutoff=simple_params["prefilter_cutoff"],
            shift_freq=simple_params["shift_freq"],
            postfilter_cutoff=simple_params["postfilter_cutoff"],
            block_size=BLOCK_SIZE,
            decrypt=True,
            key_bytes=key_bytes
        )
        out_dir = get_after_codec_dir("simple_sign", codec, bitrate)
        write_wav(os.path.join(out_dir, f"test_sound{i}_simple_sign_dec.wav"), fs, simple_sign_dec_after_codec)

    # after codec: split
    for codec, bitrate in CODEC_VARIANTS:
        split_enc_after_codec = compress_decompress_array(split_enc, fs, codec=codec, bitrate=bitrate)
        split_dec_after_codec = split_process(
            split_enc_after_codec, fs,
            band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"],
            band1_shift_freq=split_params["band1_shift_freq"],
            band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"],
            band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"],
            band2_shift_freq=split_params["band2_shift_freq"],
            band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"]
        )
        out_dir = get_after_codec_dir("split", codec, bitrate)
        write_wav(os.path.join(out_dir, f"test_sound{i}_split_dec.wav"), fs, split_dec_after_codec)

    # after codec: split_sign
    for codec, bitrate in CODEC_VARIANTS:
        split_sign_enc_after_codec = compress_decompress_array(split_sign_enc, fs, codec=codec, bitrate=bitrate)
        split_sign_dec_after_codec = split_sign_process(
            split_sign_enc_after_codec, fs, nonce=split_sign_nonce,
            band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"],
            band1_shift_freq=split_params["band1_shift_freq"],
            band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"],
            band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"],
            band2_shift_freq=split_params["band2_shift_freq"],
            band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"],
            block_size=BLOCK_SIZE,
            decrypt=True,
            key_bytes=key_bytes
        )
        out_dir = get_after_codec_dir("split_sign", codec, bitrate)
        write_wav(os.path.join(out_dir, f"test_sound{i}_split_sign_dec.wav"), fs, split_sign_dec_after_codec)

    return f"done: test_sound{i}.wav"


def main():
    ensure_dirs()
    key_bytes = load_key_bytes(KEY_PATH)
    files = list_input_files(DATASET_DIR)
    print(f"Найдено файлов: {len(files)}")

    tasks = [(i, path, key_bytes) for i, path in files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_file, task): task[0] for task in tasks}

        for future in as_completed(futures):
            i = futures[future]
            try:
                print(future.result())
            except Exception as e:
                print(f"fail: test_sound{i}.wav -> {e}")


if __name__ == "__main__":
    main()