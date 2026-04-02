from scipy.io import wavfile
import numpy as np
from scipy import signal
from scipy.signal import kaiser_beta
import os
import hashlib
FFMPEG_BIN_DIR = r"D:\5_year\nirs\ffmpeg\bin"
os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ["PATH"]
from pydub import AudioSegment
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#датасет имя, 2 места частот, число файлов
DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../dataset/test_sounds2"))

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

N_FILES = 20
KEY_PATH = os.path.join(BASE_DIR, "key.bin")
AudioSegment.converter = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
AudioSegment.ffmpeg = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")

BLOCK_SIZE = 80

def to_int16_safe(sig: np.ndarray, peak: float = 0.98) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs = np.max(np.abs(sig)) + 1e-12
    if max_abs > 0:
        sig = sig / max_abs

    sig = sig * (32767.0 * peak)
    return sig.astype(np.int16)

def to_mono_float(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data[:, 0]
    return data

def design_fir_lowpass(length_sec: float, cutoff_hz: float, fs: float, atten_db: float) -> np.ndarray:
    max_len = 2047
    taps_len = int(2 * round(fs * length_sec) + 1)
    taps_len = min(taps_len, max_len)
    if taps_len < 3:
        taps_len = 3#вычисление длины фильтра

    beta = kaiser_beta(atten_db)
    taps = signal.firwin(taps_len, cutoff=cutoff_hz, window=("kaiser", beta), fs=fs)
    return taps.astype(float)

def inverter_offline(x: np.ndarray, fs: float, freq_prefilter: float, freq_shift: float, freq_postfilter: float,  length_sec: float = 0.0024, atten_db: float = 60.0) -> np.ndarray:
    n = np.arange(len(x), dtype=float)#n — индексы отсчётов
    omega = 2.0 * np.pi * freq_shift / fs
    carrier = np.cos(omega * n)
    
    # prefilter
    taps_pre = design_fir_lowpass(length_sec, freq_prefilter, fs, atten_db)
    x_pre = signal.lfilter(taps_pre, [1.0], x)

    mixed = x_pre * carrier

    # postfilter
    taps_post = design_fir_lowpass(length_sec, freq_postfilter, fs, atten_db)
    return signal.lfilter(taps_post, [1.0], mixed)

def simple_process(x: np.ndarray, fs: int, prefilter_cutoff: float, shift_freq: float, postfilter_cutoff: float,) -> np.ndarray:
    return 1.8 * inverter_offline( x, fs, freq_prefilter=prefilter_cutoff, freq_shift=shift_freq, freq_postfilter=postfilter_cutoff,)

def split_process(x: np.ndarray,fs: int, band1_prefilter_cutoff: float, band1_shift_freq: float, band1_postfilter_cutoff: float, band2_prefilter_cutoff: float, band2_shift_freq: float, band2_postfilter_cutoff: float,) -> np.ndarray:
    y1 = inverter_offline(x,fs,freq_prefilter=band1_prefilter_cutoff,freq_shift=band1_shift_freq,freq_postfilter=band1_postfilter_cutoff,)

    y2 = inverter_offline( x, fs, freq_prefilter=band2_prefilter_cutoff, freq_shift=band2_shift_freq, freq_postfilter=band2_postfilter_cutoff,)

    return 1.8 * (y1 + y2)

# def write_wav(path, fs, sig):
#     sig = np.asarray(sig, dtype=np.float32)
#     # sig = np.clip(sig, -32768.0, 32767.0)
#     wavfile.write(path, fs, sig.astype(np.int16))
# def write_wav(path, fs, sig):
#     sig = np.asarray(sig, dtype=np.float64)
#     sig = np.nan_to_num(sig, nan=0.0, posinf=32767.0, neginf=-32768.0)
#     sig = np.clip(sig, -32768.0, 32767.0)
#     wavfile.write(path, fs, sig.astype(np.int16))
def write_wav(path, fs, sig):
    sig = to_int16_safe(sig)
    wavfile.write(path, fs, sig)

def dc_remove(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)





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




def compress_decompress_array(sig, fs, codec="mp3", bitrate="64k", target_fs=16000):
    sig = to_int16_safe(sig)

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
        # print("mp3", bitrate, "size_bytes=", size_bytes)
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
    # print(codec, bitrate, decoded_audio.frame_rate, decoded_audio.channels, decoded_audio.sample_width, len(decoded))
    # print("sum=", np.sum(decoded), "mean=", np.mean(decoded), "std=", np.std(decoded))

    return decoded



def simple_sign_process(x, fs, nonce, prefilter_cutoff, shift_freq, postfilter_cutoff, block_size, decrypt):
    rng = rng_from_key(key_bytes, nonce=nonce)
    if not decrypt:
        y = simple_process( x, fs, prefilter_cutoff=prefilter_cutoff, shift_freq=shift_freq, postfilter_cutoff=postfilter_cutoff)
        mask = sign_mask_stream(len(y), rng=rng, block_size=block_size)
        return y * mask
    else:
        mask = sign_mask_stream(len(x), rng=rng, block_size=block_size)
        y = x * mask
        return simple_process( y, fs, prefilter_cutoff=prefilter_cutoff, shift_freq=shift_freq, postfilter_cutoff=postfilter_cutoff)


def split_sign_process( x, fs, nonce, band1_prefilter_cutoff, band1_shift_freq, band1_postfilter_cutoff, band2_prefilter_cutoff, band2_shift_freq, band2_postfilter_cutoff, block_size, decrypt):
    rng = rng_from_key(key_bytes, nonce=nonce)
    if not decrypt:
        y = split_process( x, fs, band1_prefilter_cutoff=band1_prefilter_cutoff, band1_shift_freq=band1_shift_freq, band1_postfilter_cutoff=band1_postfilter_cutoff, band2_prefilter_cutoff=band2_prefilter_cutoff, band2_shift_freq=band2_shift_freq, band2_postfilter_cutoff=band2_postfilter_cutoff)
        mask = sign_mask_stream(len(y), rng=rng, block_size=block_size)
        return y * mask
    else:
        mask = sign_mask_stream(len(x), rng=rng, block_size=block_size)
        y = x * mask
        return split_process( y, fs, band1_prefilter_cutoff=band1_prefilter_cutoff, band1_shift_freq=band1_shift_freq, band1_postfilter_cutoff=band1_postfilter_cutoff, band2_prefilter_cutoff=band2_prefilter_cutoff, band2_shift_freq=band2_shift_freq, band2_postfilter_cutoff=band2_postfilter_cutoff)
    

key_bytes = load_key_bytes(KEY_PATH)#эта штука берется в функциях из глобальной области видимости. мб пофиксить надо, но пока что норм.

fs = 16000
simple_params = {
    "prefilter_cutoff": 2632.0 * fs / 8000.0,
    "shift_freq": 2632.0 * fs / 8000.0,
    "postfilter_cutoff": 2632.0 * fs / 8000.0,
}
split_params = {
    "band1_prefilter_cutoff": 500.0 * fs / 8000.0,
    "band1_shift_freq": 500.0 * fs / 8000.0,
    "band1_postfilter_cutoff": 500.0 * fs / 8000.0,

    "band2_prefilter_cutoff": 2632.0 * fs / 8000.0,
    "band2_shift_freq": 2632.0 * fs / 8000.0,
    "band2_postfilter_cutoff": 2632.0 * fs / 8000.0,
}

os.makedirs(BASE_OUT_ENC_SIMPLE_DIR, exist_ok=True)
os.makedirs(BASE_OUT_DEC_SIMPLE_DIR, exist_ok=True)
os.makedirs(BASE_OUT_ENC_SIMPLE_SIGN_DIR, exist_ok=True)
os.makedirs(BASE_OUT_DEC_SIMPLE_SIGN_DIR, exist_ok=True)

os.makedirs(BASE_OUT_ENC_SPLIT_DIR, exist_ok=True)
os.makedirs(BASE_OUT_DEC_SPLIT_DIR, exist_ok=True)
os.makedirs(BASE_OUT_ENC_SPLIT_SIGN_DIR, exist_ok=True)
os.makedirs(BASE_OUT_DEC_SPLIT_SIGN_DIR, exist_ok=True)

os.makedirs(DEC_AFTER_CODEC_SIMPLE_MP3_64_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIMPLE_MP3_128_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIMPLE_MP3_320_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIMPLE_OPUS_32_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIMPLE_OPUS_96_DIR, exist_ok=True)

os.makedirs(DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_64_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_128_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIMPLE_SIGN_MP3_320_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIMPLE_SIGN_OPUS_32_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SIMPLE_SIGN_OPUS_96_DIR, exist_ok=True)

os.makedirs(DEC_AFTER_CODEC_SPLIT_MP3_64_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SPLIT_MP3_128_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SPLIT_MP3_320_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SPLIT_OPUS_32_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SPLIT_OPUS_96_DIR, exist_ok=True)

os.makedirs(DEC_AFTER_CODEC_SPLIT_SIGN_MP3_64_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SPLIT_SIGN_MP3_128_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SPLIT_SIGN_MP3_320_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SPLIT_SIGN_OPUS_32_DIR, exist_ok=True)
os.makedirs(DEC_AFTER_CODEC_SPLIT_SIGN_OPUS_96_DIR, exist_ok=True)

CODEC_VARIANTS = [
    ("mp3", "64k"),
    ("mp3", "128k"),
    ("mp3", "320k"),
    ("opus", "32k"),
    ("opus", "96k"),
]


def codec_tag(codec: str, bitrate: str) -> str:
    return f"{codec}_{bitrate}"


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

for i in range(1, N_FILES + 1):
    in_path = os.path.join(DATASET_DIR, f"test_sound{i}.wav")
    fs, data = wavfile.read(in_path)
    print(f"test_sound{i}: fs={fs}")
    # x = to_mono_float(data)
    x = dc_remove(to_mono_float(data))

    simple_enc = simple_process(x, fs, prefilter_cutoff=simple_params["prefilter_cutoff"], shift_freq=simple_params["shift_freq"], postfilter_cutoff=simple_params["postfilter_cutoff"])
    simple_dec = simple_process(simple_enc, fs, prefilter_cutoff=simple_params["prefilter_cutoff"], shift_freq=simple_params["shift_freq"], postfilter_cutoff=simple_params["postfilter_cutoff"])
    write_wav(os.path.join(BASE_OUT_ENC_SIMPLE_DIR, f"test_sound{i}_simple_enc.wav"), fs, simple_enc)
    write_wav(os.path.join(BASE_OUT_DEC_SIMPLE_DIR, f"test_sound{i}_simple_dec.wav"), fs, simple_dec)

    simple_sign_nonce = f"test_sound{i}_simple_sign"
    simple_sign_enc_file_name = f"{simple_sign_nonce}_enc.wav"
    simple_sign_dec_file_name = f"{simple_sign_nonce}_dec.wav"
    simple_sign_enc = simple_sign_process(x, fs, nonce=simple_sign_nonce, prefilter_cutoff=simple_params["prefilter_cutoff"], shift_freq=simple_params["shift_freq"], postfilter_cutoff=simple_params["postfilter_cutoff"], block_size=BLOCK_SIZE, decrypt=False)
    simple_sign_dec = simple_sign_process(simple_sign_enc, fs, nonce=simple_sign_nonce, prefilter_cutoff=simple_params["prefilter_cutoff"], shift_freq=simple_params["shift_freq"], postfilter_cutoff=simple_params["postfilter_cutoff"], block_size=BLOCK_SIZE, decrypt=True)
    write_wav(os.path.join(BASE_OUT_ENC_SIMPLE_SIGN_DIR, simple_sign_enc_file_name), fs, simple_sign_enc)
    write_wav(os.path.join(BASE_OUT_DEC_SIMPLE_SIGN_DIR, simple_sign_dec_file_name), fs, simple_sign_dec)

    split_enc = split_process(x, fs, band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"], band1_shift_freq=split_params["band1_shift_freq"], band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"], band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"], band2_shift_freq=split_params["band2_shift_freq"], band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"])
    split_dec = split_process(split_enc, fs, band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"], band1_shift_freq=split_params["band1_shift_freq"], band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"], band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"], band2_shift_freq=split_params["band2_shift_freq"], band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"])
    write_wav(os.path.join(BASE_OUT_ENC_SPLIT_DIR, f"test_sound{i}_split_enc.wav"), fs, split_enc)
    write_wav(os.path.join(BASE_OUT_DEC_SPLIT_DIR, f"test_sound{i}_split_dec.wav"), fs, split_dec)

    split_sign_nonce = f"test_sound{i}_split_sign"
    split_sign_enc_file_name = f"{split_sign_nonce}_enc.wav"
    split_sign_dec_file_name = f"{split_sign_nonce}_dec.wav"
    split_sign_enc = split_sign_process(x, fs, nonce=split_sign_nonce, band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"], band1_shift_freq=split_params["band1_shift_freq"], band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"], band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"], band2_shift_freq=split_params["band2_shift_freq"], band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"], block_size=BLOCK_SIZE, decrypt=False)
    split_sign_dec = split_sign_process(split_sign_enc, fs, nonce=split_sign_nonce, band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"], band1_shift_freq=split_params["band1_shift_freq"], band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"], band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"], band2_shift_freq=split_params["band2_shift_freq"], band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"], block_size=BLOCK_SIZE, decrypt=True)
    write_wav(os.path.join(BASE_OUT_ENC_SPLIT_SIGN_DIR, split_sign_enc_file_name), fs, split_sign_enc)
    write_wav(os.path.join(BASE_OUT_DEC_SPLIT_SIGN_DIR, split_sign_dec_file_name), fs, split_sign_dec)







    # ===== decrypted_after_codec datasets =====

    # simple
    simple_enc = simple_process(x, fs, prefilter_cutoff=simple_params["prefilter_cutoff"], shift_freq=simple_params["shift_freq"], postfilter_cutoff=simple_params["postfilter_cutoff"])
    for codec, bitrate in CODEC_VARIANTS:
        simple_enc_after_codec = compress_decompress_array(simple_enc, fs, codec=codec, bitrate=bitrate)
        simple_dec_after_codec = simple_process(simple_enc_after_codec, fs, prefilter_cutoff=simple_params["prefilter_cutoff"], shift_freq=simple_params["shift_freq"], postfilter_cutoff=simple_params["postfilter_cutoff"])
        out_dir = get_after_codec_dir("simple", codec, bitrate)
        out_name = f"test_sound{i}_simple_dec.wav"
        write_wav(os.path.join(out_dir, out_name), fs, simple_dec_after_codec)

    # simple_sign
    simple_sign_nonce = f"test_sound{i}_simple_sign"
    simple_sign_enc = simple_sign_process(x, fs, nonce=simple_sign_nonce, prefilter_cutoff=simple_params["prefilter_cutoff"], shift_freq=simple_params["shift_freq"], postfilter_cutoff=simple_params["postfilter_cutoff"], block_size=BLOCK_SIZE, decrypt=False)

    for codec, bitrate in CODEC_VARIANTS:
        simple_sign_enc_after_codec = compress_decompress_array(simple_sign_enc, fs, codec=codec, bitrate=bitrate)
        simple_sign_dec_after_codec = simple_sign_process(simple_sign_enc_after_codec, fs, nonce=simple_sign_nonce, prefilter_cutoff=simple_params["prefilter_cutoff"], shift_freq=simple_params["shift_freq"], postfilter_cutoff=simple_params["postfilter_cutoff"], block_size=BLOCK_SIZE, decrypt=True)
        out_dir = get_after_codec_dir("simple_sign", codec, bitrate)
        out_name = f"test_sound{i}_simple_sign_dec.wav"
        write_wav(os.path.join(out_dir, out_name), fs, simple_sign_dec_after_codec)

    # split
    split_enc = split_process(x, fs, band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"], band1_shift_freq=split_params["band1_shift_freq"], band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"], band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"], band2_shift_freq=split_params["band2_shift_freq"], band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"])

    for codec, bitrate in CODEC_VARIANTS:
        split_enc_after_codec = compress_decompress_array(split_enc, fs, codec=codec, bitrate=bitrate)
        split_dec_after_codec = split_process(split_enc_after_codec, fs, band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"], band1_shift_freq=split_params["band1_shift_freq"], band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"], band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"], band2_shift_freq=split_params["band2_shift_freq"], band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"])
        out_dir = get_after_codec_dir("split", codec, bitrate)
        out_name = f"test_sound{i}_split_dec.wav"
        write_wav(os.path.join(out_dir, out_name), fs, split_dec_after_codec)

    # split_sign
    split_sign_nonce = f"test_sound{i}_split_sign"
    split_sign_enc = split_sign_process(x, fs, nonce=split_sign_nonce, band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"], band1_shift_freq=split_params["band1_shift_freq"], band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"], band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"], band2_shift_freq=split_params["band2_shift_freq"], band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"], block_size=BLOCK_SIZE, decrypt=False)

    for codec, bitrate in CODEC_VARIANTS:
        split_sign_enc_after_codec = compress_decompress_array(split_sign_enc, fs, codec=codec, bitrate=bitrate)
        split_sign_dec_after_codec = split_sign_process(split_sign_enc_after_codec, fs, nonce=split_sign_nonce, band1_prefilter_cutoff=split_params["band1_prefilter_cutoff"], band1_shift_freq=split_params["band1_shift_freq"], band1_postfilter_cutoff=split_params["band1_postfilter_cutoff"], band2_prefilter_cutoff=split_params["band2_prefilter_cutoff"], band2_shift_freq=split_params["band2_shift_freq"], band2_postfilter_cutoff=split_params["band2_postfilter_cutoff"], block_size=BLOCK_SIZE, decrypt=True)
        out_dir = get_after_codec_dir("split_sign", codec, bitrate)
        out_name = f"test_sound{i}_split_sign_dec.wav"
        write_wav(os.path.join(out_dir, out_name), fs, split_sign_dec_after_codec)