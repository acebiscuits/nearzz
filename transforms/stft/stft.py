import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import hann
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import os
import hashlib
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

KEY_PATH = os.path.join(BASE_DIR, "key", "key.bin")
WRONG_KEY_PATH = os.path.join(BASE_DIR, "key", "wrong_key.bin")
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


def to_int16_safe(sig: np.ndarray, peak: float = 0.98) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    sig = np.clip(sig, -32768, 32767)
    # max_abs = np.max(np.abs(sig)) + 1e-12
    # if max_abs > 0:
    #     sig = sig / max_abs

    # sig = sig * (32767.0 * peak)

    return sig.astype(np.int16)

def window_nonzero(window_function, segment_length):
    zero_exist = 1
    zero_count = 0
    
    window_vector = window_function(segment_length + zero_count)

    while zero_exist:
        start = int(zero_count / 2)
        stop = int(len(window_vector) - zero_count / 2)
        window_vector = window_vector[start:stop]

        zero_count = len(window_vector) - np.count_nonzero(window_vector)

        if zero_count > 0:
            window_vector = window_function(segment_length + zero_count)
        else:
            zero_exist = 0

    return window_vector

segment_length = 2048
segment_length_padded = 2048
shift_length = 1024
window_function = hann
p = 1
N_FILES = 1722
WINDOW_VECTOR = window_nonzero(window_function, segment_length)


def create_overlapping_segments(x, segment_length, shift_length):

    x = np.squeeze(x)

    start_list = np.arange(0, x.shape[0], shift_length)
    stop_list = start_list + segment_length

    index = [i <= x.shape[0] for i in stop_list]
    start_list = start_list[index]
    stop_list = stop_list[index]

    if stop_list[-1] != x.shape[0]:
        stop_list = np.append(stop_list, x.shape[0])
        start_list = np.append(start_list, x.shape[0] - segment_length)

    x_segments = [x[start:stop, ...] for start, stop in zip(start_list, stop_list)]
    x_segments = np.stack(x_segments, axis=1)

    return x_segments, start_list, stop_list


def stft(x, segment_length, segment_length_padded, shift_length, window_function):

    window_vector = WINDOW_VECTOR

    x_segments, start_list, stop_list = create_overlapping_segments(x, segment_length, shift_length)

    window_array = np.ones(x_segments.shape)
    for i in range(window_array.shape[0]):
        window_array[i] = window_array[i] * window_vector[i]

    x_segments = window_array * x_segments

    x_stft = np.fft.rfft(x_segments, n=segment_length_padded, axis=0)

    return x_stft, start_list, stop_list



def istft(x_stft, segment_length, segment_length_padded, start_list, stop_list, original_size, window_function, p):

    x_segments = np.fft.irfft(x_stft, n=segment_length_padded, axis=0)
    x_segments = x_segments[0:segment_length]

    window_vector = WINDOW_VECTOR

    window_array = np.ones(x_segments.shape)
    for i in range(window_array.shape[0]):
        window_array[i] = window_array[i] * window_vector[i]
    window_array = window_array ** (p - 1)

    x_segments = window_array * x_segments

    window_overlap_add = np.zeros(original_size[0])
    number_segments = len(start_list)
    for i in range(number_segments):
        window_overlap_add[start_list[i]:stop_list[i]] += window_vector ** p
    window_overlap_add = (window_overlap_add) ** -1

    x = np.zeros(original_size)
    for i, (start, stop) in enumerate(zip(start_list, stop_list)):
        x[start:stop, ...] += x_segments[:, i, ...]

    window_overlap_add_array = np.zeros(original_size)
    for i in range(x.shape[0]):
        window_overlap_add_array[i] = window_overlap_add[i]
    x = x * window_overlap_add_array

    return x


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


def sign_mask_matrix(shape, rng):
    return rng.choice(np.array([-1.0, 1.0]), size=shape).astype(np.float64)


def build_perm_matrix(n_freq_bins, n_frames, rng):
    perms = np.empty((n_freq_bins, n_frames), dtype=np.int64)
    inv_perms = np.empty((n_freq_bins, n_frames), dtype=np.int64)

    for t in range(n_frames):
        perm = rng.permutation(n_freq_bins)
        inv_perm = np.argsort(perm)
        perms[:, t] = perm
        inv_perms[:, t] = inv_perm

    return perms, inv_perms

def stft_signflip_with_mask(X, mask):
    X2 = X.copy()

    if X2.shape[0] > 2:
        X2[1:-1, :] = X2[1:-1, :] * mask

    return X2

def stft_shuffle_with_perms(X, perms):
    X2 = np.empty_like(X)
    for t in range(X.shape[1]):
        X2[:, t] = X[:, t][perms[:, t]]
    return X2

def stft_unshuffle_with_inv_perms(X, inv_perms):
    X2 = np.empty_like(X)
    for t in range(X.shape[1]):
        X2[:, t] = X[:, t][inv_perms[:, t]]
    return X2

def write_wav(path, fs, sig):
    sig = to_int16_safe(sig)
    wavfile.write(path, fs, sig)






def compress_decompress_array(sig, fs, codec="mp3", bitrate="64k", target_fs=16000):
    sig = to_int16_safe(sig)

    in_buf = io.BytesIO()
    wavfile.write(in_buf, fs, sig)
    in_buf.seek(0)

    audio = AudioSegment.from_file(in_buf, format="wav")
    audio = audio.set_frame_rate(target_fs).set_channels(1).set_sample_width(2)

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

    decoded_audio = decoded_audio.set_frame_rate(target_fs).set_channels(1).set_sample_width(2)
    decoded = np.array(decoded_audio.get_array_of_samples(), dtype=np.float64)
    return target_fs, decoded


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

    data = data.astype(np.float64)
    original_size = data.shape

    X_stft, start_list, stop_list = stft(data, segment_length, segment_length_padded, shift_length, window_function)

    # invert
    # X_inv = X_stft[::-1, :]
    # inv_audio = istft(X_inv, segment_length, segment_length_padded, start_list, stop_list, original_size, window_function, p)
    # X_inv_dec = X_inv[::-1, :]
    # inv_dec = istft(X_inv_dec, segment_length, segment_length_padded, start_list, stop_list, original_size, window_function, p)

    # write_wav(os.path.join(OUT_ENC_INVERT_DIR, f"test_sound{i}_invert_enc.wav"), fs, inv_audio)
    # write_wav(os.path.join(OUT_DEC_INVERT_DIR, f"test_sound{i}_invert_dec.wav"), fs, inv_dec)

    # shuffle
    shuffle_nonce = f"test_sound{i}_shuffle"
    shuffle_rng = rng_from_key(key_bytes, nonce=shuffle_nonce)
    perms, inv_perms = build_perm_matrix(X_stft.shape[0], X_stft.shape[1], shuffle_rng)

    shuffle_rng_wrong_key = rng_from_key(wrong_key_bytes, nonce=shuffle_nonce)
    perms_wrong_key, inv_perms_wrong_key = build_perm_matrix(X_stft.shape[0], X_stft.shape[1], shuffle_rng_wrong_key)

    X_shuffle = stft_shuffle_with_perms(X_stft, perms)
    shuffle_audio = istft(X_shuffle, segment_length, segment_length_padded, start_list, stop_list, original_size, window_function, p)
    X_shuffle_dec = stft_unshuffle_with_inv_perms(X_shuffle, inv_perms_wrong_key)
    shuffle_dec = istft(X_shuffle_dec, segment_length, segment_length_padded, start_list, stop_list, original_size, window_function, p)

    write_wav(os.path.join(OUT_ENC_SHUFFLE_DIR, f"test_sound{i}_shuffle_enc.wav"), fs, shuffle_audio)
    write_wav(os.path.join(OUT_DEC_SHUFFLE_WK_DIR, f"test_sound{i}_shuffle_dec.wav"), fs, shuffle_dec)

    # signflip
    signflip_nonce = f"test_sound{i}_signflip"
    signflip_rng = rng_from_key(key_bytes, nonce=signflip_nonce)
    mask = sign_mask_matrix((X_stft.shape[0] - 2, X_stft.shape[1]), signflip_rng)

    signflip_rng_wrong_key = rng_from_key(wrong_key_bytes, nonce=signflip_nonce)
    mask_wrong_key = sign_mask_matrix((X_stft.shape[0] - 2, X_stft.shape[1]), signflip_rng_wrong_key)

    X_sign = stft_signflip_with_mask(X_stft, mask)
    sign_audio = istft(X_sign, segment_length, segment_length_padded, start_list, stop_list, original_size, window_function, p)
    X_sign_dec = stft_signflip_with_mask(X_sign, mask_wrong_key)
    sign_dec = istft(X_sign_dec, segment_length, segment_length_padded, start_list, stop_list, original_size, window_function, p)

    write_wav(os.path.join(OUT_ENC_SIGNFLIP_DIR, f"test_sound{i}_signflip_enc.wav"), fs, sign_audio)
    write_wav(os.path.join(OUT_DEC_SIGNFLIP_WK_DIR, f"test_sound{i}_signflip_dec.wav"), fs, sign_dec)

    # after codec: invert
    # for codec, bitrate in CODEC_VARIANTS:
    #     fs_c, inv_audio_after_codec = compress_decompress_array(inv_audio, fs, codec=codec, bitrate=bitrate)
    #     X_inv_after_codec, start_list_c, stop_list_c = stft(inv_audio_after_codec, segment_length, segment_length_padded, shift_length, window_function)
    #     X_inv_dec_after_codec = X_inv_after_codec[::-1, :]
    #     inv_dec_after_codec = istft(X_inv_dec_after_codec, segment_length, segment_length_padded, start_list_c, stop_list_c, inv_audio_after_codec.shape, window_function, p)

    #     out_dir = get_after_codec_dir("invert", codec, bitrate)
    #     write_wav(os.path.join(out_dir, f"test_sound{i}_invert_dec.wav"), fs_c, inv_dec_after_codec)

    # # after codec: shuffle
    # for codec, bitrate in CODEC_VARIANTS:
    #     fs_c, shuffle_audio_after_codec = compress_decompress_array(shuffle_audio, fs, codec=codec, bitrate=bitrate)
    #     X_shuffle_after_codec, start_list_c, stop_list_c = stft(shuffle_audio_after_codec, segment_length, segment_length_padded, shift_length, window_function)
    #     X_shuffle_dec_after_codec = stft_unshuffle_with_inv_perms(X_shuffle_after_codec, inv_perms)
    #     shuffle_dec_after_codec = istft(X_shuffle_dec_after_codec, segment_length, segment_length_padded, start_list_c, stop_list_c, shuffle_audio_after_codec.shape, window_function, p)

    #     out_dir = get_after_codec_dir("shuffle", codec, bitrate)
    #     write_wav(os.path.join(out_dir, f"test_sound{i}_shuffle_dec.wav"), fs_c, shuffle_dec_after_codec)

    # # after codec: signflip
    # for codec, bitrate in CODEC_VARIANTS:
    #     fs_c, sign_audio_after_codec = compress_decompress_array(sign_audio, fs, codec=codec, bitrate=bitrate)
    #     X_sign_after_codec, start_list_c, stop_list_c = stft(sign_audio_after_codec, segment_length, segment_length_padded, shift_length, window_function)
    #     X_sign_dec_after_codec = stft_signflip_with_mask(X_sign_after_codec, mask)
    #     sign_dec_after_codec = istft(X_sign_dec_after_codec, segment_length, segment_length_padded, start_list_c, stop_list_c, sign_audio_after_codec.shape, window_function, p)

    #     out_dir = get_after_codec_dir("signflip", codec, bitrate)
    #     write_wav(os.path.join(out_dir, f"test_sound{i}_signflip_dec.wav"), fs_c, sign_dec_after_codec)

    return f"done: test_sound{i}.wav"

key_bytes = load_key_bytes(KEY_PATH)
wrong_key_bytes = load_key_bytes(WRONG_KEY_PATH)

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one_file, i): i for i in range(1, N_FILES + 1)}

        for future in as_completed(futures):
            i = futures[future]
            try:
                print(future.result())
            except Exception as e:
                print(f"fail: test_sound{i}.wav -> {e}")