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
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

FFMPEG_BIN_DIR = r"D:\5_year\nirs\ffmpeg\bin"
os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ["PATH"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../../dataset/test_sounds3"))

# ===== СТАРЫЕ ПУТИ: УЖЕ ПОСЧИТАНО. НЕ СОЗДАВАТЬ =====
# OUT_ENC_INVERT_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "invert")
# OUT_DEC_INVERT_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "invert")
#
# OUT_ENC_SHUFFLE_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "shuffle")
# OUT_DEC_SHUFFLE_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "shuffle")
# OUT_DEC_SHUFFLE_WK_DIR = os.path.join(BASE_DIR, "assets", "decrypted_wrong_key", "shuffle")
#
# OUT_ENC_SIGNFLIP_DIR = os.path.join(BASE_DIR, "assets", "encrypted", "signflip")
# OUT_DEC_SIGNFLIP_DIR = os.path.join(BASE_DIR, "assets", "decrypted", "signflip")
# OUT_DEC_SIGNFLIP_WK_DIR = os.path.join(BASE_DIR, "assets", "decrypted_wrong_key", "signflip")
#
# BASE_OUT_DEC_AFTER_CODEC_DIR = os.path.join(BASE_DIR, "assets", "decrypted_after_codec")
#
# DEC_AFTER_CODEC_INVERT_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_mp3_64k")
# DEC_AFTER_CODEC_INVERT_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_mp3_128k")
# DEC_AFTER_CODEC_INVERT_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_mp3_320k")
# DEC_AFTER_CODEC_INVERT_OPUS_16_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_opus_16k")
# DEC_AFTER_CODEC_INVERT_OPUS_48_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_opus_48k")
# DEC_AFTER_CODEC_INVERT_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "invert_opus_96k")
#
# DEC_AFTER_CODEC_SHUFFLE_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_mp3_64k")
# DEC_AFTER_CODEC_SHUFFLE_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_mp3_128k")
# DEC_AFTER_CODEC_SHUFFLE_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_mp3_320k")
# DEC_AFTER_CODEC_SHUFFLE_OPUS_16_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_opus_16k")
# DEC_AFTER_CODEC_SHUFFLE_OPUS_48_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_opus_48k")
# DEC_AFTER_CODEC_SHUFFLE_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "shuffle_opus_96k")
#
# DEC_AFTER_CODEC_SIGNFLIP_MP3_64_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_mp3_64k")
# DEC_AFTER_CODEC_SIGNFLIP_MP3_128_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_mp3_128k")
# DEC_AFTER_CODEC_SIGNFLIP_MP3_320_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_mp3_320k")
# DEC_AFTER_CODEC_SIGNFLIP_OPUS_16_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_opus_16k")
# DEC_AFTER_CODEC_SIGNFLIP_OPUS_48_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_opus_48k")
# DEC_AFTER_CODEC_SIGNFLIP_OPUS_96_DIR = os.path.join(BASE_OUT_DEC_AFTER_CODEC_DIR, "signflip_opus_96k")

# ===== НОВЫЕ ЭКСПЕРИМЕНТЫ: ОТДЕЛЬНО ОТ СТАРОГО assets/ =====
NEW_RESULTS_DIR = os.path.join(BASE_DIR, "new_experiments")
NEW_ASSETS_DIR = os.path.join(NEW_RESULTS_DIR, "assets")
NEW_DEC_AFTER_CODEC_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted_after_codec")
NEW_DEC_AFTER_CODEC_WK_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted_after_codec_wrong_key")

# orthomix
OUT_ENC_ORTHOMIX_DIR = os.path.join(NEW_ASSETS_DIR, "encrypted", "orthomix")
OUT_DEC_ORTHOMIX_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted", "orthomix")
OUT_DEC_ORTHOMIX_WK_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted_wrong_key", "orthomix")
DEC_AFTER_CODEC_ORTHOMIX_MP3_64_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "orthomix_mp3_64k")
DEC_AFTER_CODEC_ORTHOMIX_MP3_128_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "orthomix_mp3_128k")
DEC_AFTER_CODEC_ORTHOMIX_MP3_320_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "orthomix_mp3_320k")
DEC_AFTER_CODEC_ORTHOMIX_OPUS_16_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "orthomix_opus_16k")
DEC_AFTER_CODEC_ORTHOMIX_OPUS_48_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "orthomix_opus_48k")
DEC_AFTER_CODEC_ORTHOMIX_OPUS_96_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "orthomix_opus_96k")
DEC_AFTER_CODEC_ORTHOMIX_MP3_64_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "orthomix_mp3_64k")
DEC_AFTER_CODEC_ORTHOMIX_MP3_128_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "orthomix_mp3_128k")
DEC_AFTER_CODEC_ORTHOMIX_MP3_320_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "orthomix_mp3_320k")
DEC_AFTER_CODEC_ORTHOMIX_OPUS_16_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "orthomix_opus_16k")
DEC_AFTER_CODEC_ORTHOMIX_OPUS_48_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "orthomix_opus_48k")
DEC_AFTER_CODEC_ORTHOMIX_OPUS_96_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "orthomix_opus_96k")

# amplitude-sign
OUT_ENC_AMPLITUDE_SIGN_DIR = os.path.join(NEW_ASSETS_DIR, "encrypted", "amplitude-sign")
OUT_DEC_AMPLITUDE_SIGN_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted", "amplitude-sign")
OUT_DEC_AMPLITUDE_SIGN_WK_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted_wrong_key", "amplitude-sign")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_64_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "amplitude-sign_mp3_64k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_128_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "amplitude-sign_mp3_128k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_320_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "amplitude-sign_mp3_320k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_16_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "amplitude-sign_opus_16k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_48_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "amplitude-sign_opus_48k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_96_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "amplitude-sign_opus_96k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_64_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "amplitude-sign_mp3_64k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_128_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "amplitude-sign_mp3_128k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_320_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "amplitude-sign_mp3_320k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_16_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "amplitude-sign_opus_16k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_48_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "amplitude-sign_opus_48k")
DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_96_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "amplitude-sign_opus_96k")

# adaptive-orthomix
OUT_ENC_ADAPTIVE_ORTHOMIX_DIR = os.path.join(NEW_ASSETS_DIR, "encrypted", "adaptive-orthomix")
OUT_DEC_ADAPTIVE_ORTHOMIX_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted", "adaptive-orthomix")
OUT_DEC_ADAPTIVE_ORTHOMIX_WK_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted_wrong_key", "adaptive-orthomix")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_64_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "adaptive-orthomix_mp3_64k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_128_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "adaptive-orthomix_mp3_128k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_320_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "adaptive-orthomix_mp3_320k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_16_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "adaptive-orthomix_opus_16k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_48_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "adaptive-orthomix_opus_48k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_96_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "adaptive-orthomix_opus_96k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_64_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "adaptive-orthomix_mp3_64k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_128_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "adaptive-orthomix_mp3_128k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_320_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "adaptive-orthomix_mp3_320k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_16_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "adaptive-orthomix_opus_16k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_48_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "adaptive-orthomix_opus_48k")
DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_96_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "adaptive-orthomix_opus_96k")

# sbox-16: настоящая двухбайтовая AES S-box по int16 MDCT коэффициентам.
# Промежуточный encrypted WAV для этого режима — FLOAT64 (без клиппинга).
OUT_ENC_SBOX16_DIR = os.path.join(NEW_ASSETS_DIR, "encrypted", "sbox-16")
OUT_DEC_SBOX16_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted", "sbox-16")
OUT_DEC_SBOX16_WK_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted_wrong_key", "sbox-16")

# rot-xor-16: отдельный эксперимент с int16 квантованием и FLOAT64 WAV.
# Старый 8-битный rot-xor оставлен без изменений для сравнения.
OUT_ENC_ROT_XOR16_DIR = os.path.join(NEW_ASSETS_DIR, "encrypted", "rot-xor-16")
OUT_DEC_ROT_XOR16_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted", "rot-xor-16")
OUT_DEC_ROT_XOR16_WK_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted_wrong_key", "rot-xor-16")

# rot-xor
OUT_ENC_ROT_XOR_DIR = os.path.join(NEW_ASSETS_DIR, "encrypted", "rot-xor")
OUT_DEC_ROT_XOR_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted", "rot-xor")
OUT_DEC_ROT_XOR_WK_DIR = os.path.join(NEW_ASSETS_DIR, "decrypted_wrong_key", "rot-xor")
DEC_AFTER_CODEC_ROT_XOR_MP3_64_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "rot-xor_mp3_64k")
DEC_AFTER_CODEC_ROT_XOR_MP3_128_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "rot-xor_mp3_128k")
DEC_AFTER_CODEC_ROT_XOR_MP3_320_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "rot-xor_mp3_320k")
DEC_AFTER_CODEC_ROT_XOR_OPUS_16_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "rot-xor_opus_16k")
DEC_AFTER_CODEC_ROT_XOR_OPUS_48_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "rot-xor_opus_48k")
DEC_AFTER_CODEC_ROT_XOR_OPUS_96_DIR = os.path.join(NEW_DEC_AFTER_CODEC_DIR, "rot-xor_opus_96k")
DEC_AFTER_CODEC_ROT_XOR_MP3_64_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "rot-xor_mp3_64k")
DEC_AFTER_CODEC_ROT_XOR_MP3_128_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "rot-xor_mp3_128k")
DEC_AFTER_CODEC_ROT_XOR_MP3_320_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "rot-xor_mp3_320k")
DEC_AFTER_CODEC_ROT_XOR_OPUS_16_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "rot-xor_opus_16k")
DEC_AFTER_CODEC_ROT_XOR_OPUS_48_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "rot-xor_opus_48k")
DEC_AFTER_CODEC_ROT_XOR_OPUS_96_WK_DIR = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, "rot-xor_opus_96k")

NEW_OUTPUT_DIRS = {
    "orthomix": {
        "encrypted": OUT_ENC_ORTHOMIX_DIR,
        "decrypted": OUT_DEC_ORTHOMIX_DIR,
        "wrong": OUT_DEC_ORTHOMIX_WK_DIR,
    },
    "amplitude-sign": {
        "encrypted": OUT_ENC_AMPLITUDE_SIGN_DIR,
        "decrypted": OUT_DEC_AMPLITUDE_SIGN_DIR,
        "wrong": OUT_DEC_AMPLITUDE_SIGN_WK_DIR,
    },
    "adaptive-orthomix": {
        "encrypted": OUT_ENC_ADAPTIVE_ORTHOMIX_DIR,
        "decrypted": OUT_DEC_ADAPTIVE_ORTHOMIX_DIR,
        "wrong": OUT_DEC_ADAPTIVE_ORTHOMIX_WK_DIR,
    },
    "rot-xor": {
        "encrypted": OUT_ENC_ROT_XOR_DIR,
        "decrypted": OUT_DEC_ROT_XOR_DIR,
        "wrong": OUT_DEC_ROT_XOR_WK_DIR,
    },
}

NEW_OUTPUT_DIRS["sbox-16"] = {
    "encrypted": OUT_ENC_SBOX16_DIR,
    "decrypted": OUT_DEC_SBOX16_DIR,
    "wrong": OUT_DEC_SBOX16_WK_DIR,
}
NEW_OUTPUT_DIRS["rot-xor-16"] = {
    "encrypted": OUT_ENC_ROT_XOR16_DIR,
    "decrypted": OUT_DEC_ROT_XOR16_DIR,
    "wrong": OUT_DEC_ROT_XOR16_WK_DIR,
}

NEW_CODEC_DIRS = {
    ("orthomix", "mp3", "64k"): DEC_AFTER_CODEC_ORTHOMIX_MP3_64_DIR,
    ("orthomix", "mp3", "128k"): DEC_AFTER_CODEC_ORTHOMIX_MP3_128_DIR,
    ("orthomix", "mp3", "320k"): DEC_AFTER_CODEC_ORTHOMIX_MP3_320_DIR,
    ("orthomix", "opus", "16k"): DEC_AFTER_CODEC_ORTHOMIX_OPUS_16_DIR,
    ("orthomix", "opus", "48k"): DEC_AFTER_CODEC_ORTHOMIX_OPUS_48_DIR,
    ("orthomix", "opus", "96k"): DEC_AFTER_CODEC_ORTHOMIX_OPUS_96_DIR,
    ("amplitude-sign", "mp3", "64k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_64_DIR,
    ("amplitude-sign", "mp3", "128k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_128_DIR,
    ("amplitude-sign", "mp3", "320k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_320_DIR,
    ("amplitude-sign", "opus", "16k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_16_DIR,
    ("amplitude-sign", "opus", "48k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_48_DIR,
    ("amplitude-sign", "opus", "96k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_96_DIR,
    ("adaptive-orthomix", "mp3", "64k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_64_DIR,
    ("adaptive-orthomix", "mp3", "128k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_128_DIR,
    ("adaptive-orthomix", "mp3", "320k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_320_DIR,
    ("adaptive-orthomix", "opus", "16k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_16_DIR,
    ("adaptive-orthomix", "opus", "48k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_48_DIR,
    ("adaptive-orthomix", "opus", "96k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_96_DIR,
    ("rot-xor", "mp3", "64k"): DEC_AFTER_CODEC_ROT_XOR_MP3_64_DIR,
    ("rot-xor", "mp3", "128k"): DEC_AFTER_CODEC_ROT_XOR_MP3_128_DIR,
    ("rot-xor", "mp3", "320k"): DEC_AFTER_CODEC_ROT_XOR_MP3_320_DIR,
    ("rot-xor", "opus", "16k"): DEC_AFTER_CODEC_ROT_XOR_OPUS_16_DIR,
    ("rot-xor", "opus", "48k"): DEC_AFTER_CODEC_ROT_XOR_OPUS_48_DIR,
    ("rot-xor", "opus", "96k"): DEC_AFTER_CODEC_ROT_XOR_OPUS_96_DIR,
}

NEW_CODEC_WRONG_DIRS = {
    ("orthomix", "mp3", "64k"): DEC_AFTER_CODEC_ORTHOMIX_MP3_64_WK_DIR,
    ("orthomix", "mp3", "128k"): DEC_AFTER_CODEC_ORTHOMIX_MP3_128_WK_DIR,
    ("orthomix", "mp3", "320k"): DEC_AFTER_CODEC_ORTHOMIX_MP3_320_WK_DIR,
    ("orthomix", "opus", "16k"): DEC_AFTER_CODEC_ORTHOMIX_OPUS_16_WK_DIR,
    ("orthomix", "opus", "48k"): DEC_AFTER_CODEC_ORTHOMIX_OPUS_48_WK_DIR,
    ("orthomix", "opus", "96k"): DEC_AFTER_CODEC_ORTHOMIX_OPUS_96_WK_DIR,
    ("amplitude-sign", "mp3", "64k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_64_WK_DIR,
    ("amplitude-sign", "mp3", "128k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_128_WK_DIR,
    ("amplitude-sign", "mp3", "320k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_MP3_320_WK_DIR,
    ("amplitude-sign", "opus", "16k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_16_WK_DIR,
    ("amplitude-sign", "opus", "48k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_48_WK_DIR,
    ("amplitude-sign", "opus", "96k"): DEC_AFTER_CODEC_AMPLITUDE_SIGN_OPUS_96_WK_DIR,
    ("adaptive-orthomix", "mp3", "64k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_64_WK_DIR,
    ("adaptive-orthomix", "mp3", "128k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_128_WK_DIR,
    ("adaptive-orthomix", "mp3", "320k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_MP3_320_WK_DIR,
    ("adaptive-orthomix", "opus", "16k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_16_WK_DIR,
    ("adaptive-orthomix", "opus", "48k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_48_WK_DIR,
    ("adaptive-orthomix", "opus", "96k"): DEC_AFTER_CODEC_ADAPTIVE_ORTHOMIX_OPUS_96_WK_DIR,
    ("rot-xor", "mp3", "64k"): DEC_AFTER_CODEC_ROT_XOR_MP3_64_WK_DIR,
    ("rot-xor", "mp3", "128k"): DEC_AFTER_CODEC_ROT_XOR_MP3_128_WK_DIR,
    ("rot-xor", "mp3", "320k"): DEC_AFTER_CODEC_ROT_XOR_MP3_320_WK_DIR,
    ("rot-xor", "opus", "16k"): DEC_AFTER_CODEC_ROT_XOR_OPUS_16_WK_DIR,
    ("rot-xor", "opus", "48k"): DEC_AFTER_CODEC_ROT_XOR_OPUS_48_WK_DIR,
    ("rot-xor", "opus", "96k"): DEC_AFTER_CODEC_ROT_XOR_OPUS_96_WK_DIR,
}

FLOAT_CODEC_OPERATIONS = {
    "sbox-16-float-codec": "sbox-16",
    "rot-xor-16-float-codec": "rot-xor-16",
}
INT16_MDCT_OPERATIONS = ("sbox-16", "rot-xor-16", *FLOAT_CODEC_OPERATIONS)

for _operation in FLOAT_CODEC_OPERATIONS:
    NEW_OUTPUT_DIRS[_operation] = {
        "encrypted": os.path.join(NEW_ASSETS_DIR, "encrypted", _operation),
        "decrypted": os.path.join(NEW_ASSETS_DIR, "decrypted", _operation),
        "wrong": os.path.join(NEW_ASSETS_DIR, "decrypted_wrong_key", _operation),
    }

CODEC_VARIANTS = [
    ("mp3", "64k"),
    ("mp3", "128k"),
    ("mp3", "320k"),
    ("opus", "16k"),
    ("opus", "48k"),
    ("opus", "96k"),
]
for _operation in INT16_MDCT_OPERATIONS:
    for _codec, _bitrate in CODEC_VARIANTS:
        _suffix = f"{_operation}_{_codec}_{_bitrate}"
        NEW_CODEC_DIRS[(_operation, _codec, _bitrate)] = os.path.join(NEW_DEC_AFTER_CODEC_DIR, _suffix)
        NEW_CODEC_WRONG_DIRS[(_operation, _codec, _bitrate)] = os.path.join(NEW_DEC_AFTER_CODEC_WK_DIR, _suffix)

# Количество файлов и процессов оставлено таким же, как раньше.

# ===== НАСТРОЙКИ ЗАПУСКА НОВОЙ СЕРИИ =====
NEW_OPERATIONS = ("sbox-16-float-codec", "rot-xor-16-float-codec")
MAX_WORKERS = 8
RECALCULATE_EXISTING = False
SAVE_CODEC_WRONG_KEY = False

KEY_PATH = os.path.join(BASE_DIR, "key.bin")
# WRONG_KEY_PATH = os.path.join(BASE_DIR, "wrong_key.bin")  # не нужен: неправильный ключ задан ниже

# ===== СТАРОЕ СОЗДАНИЕ КАТАЛОГОВ: ОТКЛЮЧЕНО =====
# os.makedirs(OUT_ENC_INVERT_DIR, exist_ok=True)
# os.makedirs(OUT_DEC_INVERT_DIR, exist_ok=True)
# os.makedirs(OUT_ENC_SHUFFLE_DIR, exist_ok=True)
# os.makedirs(OUT_DEC_SHUFFLE_DIR, exist_ok=True)
# os.makedirs(OUT_ENC_SIGNFLIP_DIR, exist_ok=True)
# os.makedirs(OUT_DEC_SIGNFLIP_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_INVERT_MP3_64_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_INVERT_MP3_128_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_INVERT_MP3_320_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_INVERT_OPUS_16_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_INVERT_OPUS_48_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_INVERT_OPUS_96_DIR, exist_ok=True)
#
# os.makedirs(DEC_AFTER_CODEC_SHUFFLE_MP3_64_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SHUFFLE_MP3_128_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SHUFFLE_MP3_320_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SHUFFLE_OPUS_16_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SHUFFLE_OPUS_48_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SHUFFLE_OPUS_96_DIR, exist_ok=True)
#
# os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_MP3_64_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_MP3_128_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_MP3_320_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_OPUS_16_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_OPUS_48_DIR, exist_ok=True)
# os.makedirs(DEC_AFTER_CODEC_SIGNFLIP_OPUS_96_DIR, exist_ok=True)
#
# os.makedirs(OUT_DEC_SHUFFLE_WK_DIR, exist_ok=True)
# os.makedirs(OUT_DEC_SIGNFLIP_WK_DIR, exist_ok=True)

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
    from pydub import AudioSegment

    AudioSegment.converter = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
    AudioSegment.ffmpeg = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(FFMPEG_BIN_DIR, "ffprobe.exe")
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

    decoded_audio = decoded_audio.set_frame_rate(target_fs).set_channels(1).set_sample_width(2)
    decoded = np.array(decoded_audio.get_array_of_samples(), dtype=np.float64)

    return decoded


def compress_decompress_float(sig, fs, codec, bitrate, gain):
    """Прогон через MP3/Opus с подачей и чтением PCM Float32."""
    encoder, sample_format, extension = {
        "mp3": ("libmp3lame", "fltp", ".mp3"),
        "opus": ("libopus", "flt", ".ogg"),
    }[codec]
    configured_ffmpeg = os.path.join(FFMPEG_BIN_DIR, "ffmpeg.exe")
    ffmpeg = configured_ffmpeg if os.path.isfile(configured_ffmpeg) else shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg не найден в FFMPEG_BIN_DIR или PATH")

    # gain ограничивает пик 30000 в исходных единицах. Для Float PCM
    # 32768 соответствует единичной амплитуде. После декодирования убираем оба масштаба.
    pcm_scale = gain / 32768.0
    pcm = np.asarray(sig * pcm_scale, dtype="<f4")
    common = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    with tempfile.TemporaryDirectory(prefix="mdct_codec_") as directory:
        # MP3 записываем в файл, чтобы FFmpeg сохранил данные о задержке и padding.
        compressed = os.path.join(directory, "audio" + extension)
        commands = (
            ("encode", common + [
                "-f", "f32le", "-ar", str(fs), "-ac", "1", "-i", "pipe:0",
                "-c:a", encoder, "-sample_fmt", sample_format, "-b:a", bitrate, compressed,
            ], pcm.tobytes()),
            ("decode", common + [
                "-i", compressed, "-ar", str(fs), "-ac", "1",
                "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
            ], b""),
        )
        for stage, command, payload in commands:
            result = subprocess.run(command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode:
                error = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"{codec} {bitrate} {stage}: {error}")

    decoded = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    return decoded / pcm_scale


# ===== СТАРАЯ ОБРАБОТКА (АРХИВНЫЙ КОД, ОТКЛЮЧЕНА) =====
# def get_after_codec_dir(mode: str, codec: str, bitrate: str) -> str:
#     dirs = {
#         ("invert", "mp3", "64k"): DEC_AFTER_CODEC_INVERT_MP3_64_DIR,
#         ("invert", "mp3", "128k"): DEC_AFTER_CODEC_INVERT_MP3_128_DIR,
#         ("invert", "mp3", "320k"): DEC_AFTER_CODEC_INVERT_MP3_320_DIR,
#         ("invert", "opus", "16k"): DEC_AFTER_CODEC_INVERT_OPUS_16_DIR,
#         ("invert", "opus", "48k"): DEC_AFTER_CODEC_INVERT_OPUS_48_DIR,
#         ("invert", "opus", "96k"): DEC_AFTER_CODEC_INVERT_OPUS_96_DIR,
#
#         ("shuffle", "mp3", "64k"): DEC_AFTER_CODEC_SHUFFLE_MP3_64_DIR,
#         ("shuffle", "mp3", "128k"): DEC_AFTER_CODEC_SHUFFLE_MP3_128_DIR,
#         ("shuffle", "mp3", "320k"): DEC_AFTER_CODEC_SHUFFLE_MP3_320_DIR,
#         ("shuffle", "opus", "16k"): DEC_AFTER_CODEC_SHUFFLE_OPUS_16_DIR,
#         ("shuffle", "opus", "48k"): DEC_AFTER_CODEC_SHUFFLE_OPUS_48_DIR,
#         ("shuffle", "opus", "96k"): DEC_AFTER_CODEC_SHUFFLE_OPUS_96_DIR,
#
#         ("signflip", "mp3", "64k"): DEC_AFTER_CODEC_SIGNFLIP_MP3_64_DIR,
#         ("signflip", "mp3", "128k"): DEC_AFTER_CODEC_SIGNFLIP_MP3_128_DIR,
#         ("signflip", "mp3", "320k"): DEC_AFTER_CODEC_SIGNFLIP_MP3_320_DIR,
#         ("signflip", "opus", "16k"): DEC_AFTER_CODEC_SIGNFLIP_OPUS_16_DIR,
#         ("signflip", "opus", "48k"): DEC_AFTER_CODEC_SIGNFLIP_OPUS_48_DIR,
#         ("signflip", "opus", "96k"): DEC_AFTER_CODEC_SIGNFLIP_OPUS_96_DIR,
#     }
#     return dirs[(mode, codec, bitrate)]
#
# def process_one_file(i):
#     in_path = os.path.join(DATASET_DIR, f"test_sound{i}.wav")
#
#     fs, data = wavfile.read(in_path)
#
#     if data.ndim > 1:
#         data = data[:, 0]
#
#     x = data.astype(np.float64)
#
#     # invert
#     # inv_audio = process_mdct_blocks(x, invert)
#     # inv_dec = process_mdct_blocks(inv_audio, invert)
#
#     # write_wav(os.path.join(OUT_ENC_INVERT_DIR, f"test_sound{i}_invert_enc.wav"), fs, inv_audio)
#     # write_wav(os.path.join(OUT_DEC_INVERT_DIR, f"test_sound{i}_invert_dec.wav"), fs, inv_dec)
#
#     # shuffle
#     shuffle_nonce = f"test_sound{i}_shuffle"
#     shuf_audio = process_mdct_blocks(x, do_shuffle_enc, key_bytes, shuffle_nonce)
#     shuf_dec = process_mdct_blocks(shuf_audio, do_shuffle_dec, wrong_key_bytes, shuffle_nonce)
#
#     write_wav(os.path.join(OUT_ENC_SHUFFLE_DIR, f"test_sound{i}_shuffle_enc.wav"), fs, shuf_audio)
#     write_wav(os.path.join(OUT_DEC_SHUFFLE_WK_DIR, f"test_sound{i}_shuffle_dec.wav"), fs, shuf_dec)
#
#     # signflip
#     signflip_nonce = f"test_sound{i}_signflip"
#     sign_audio = process_mdct_blocks(x, do_signflip, key_bytes, signflip_nonce)
#     sign_dec = process_mdct_blocks(sign_audio, do_signflip, wrong_key_bytes, signflip_nonce)
#
#     write_wav(os.path.join(OUT_ENC_SIGNFLIP_DIR, f"test_sound{i}_signflip_enc.wav"), fs, sign_audio)
#     write_wav(os.path.join(OUT_DEC_SIGNFLIP_WK_DIR, f"test_sound{i}_signflip_dec.wav"), fs, sign_dec)
#
#     # ===== decrypted_after_codec datasets =====
#
#     # invert
#
#     # for codec, bitrate in CODEC_VARIANTS:
#     #     inv_audio_after_codec = compress_decompress_array(inv_audio, fs, codec=codec, bitrate=bitrate)
#     #     inv_dec_after_codec = process_mdct_blocks(inv_audio_after_codec, invert)
#     #     out_dir = get_after_codec_dir("invert", codec, bitrate)
#     #     out_name = f"test_sound{i}_invert_dec.wav"
#     #     write_wav(os.path.join(out_dir, out_name), fs, inv_dec_after_codec)
#
#     # # shuffle
#
#     # for codec, bitrate in CODEC_VARIANTS:
#     #     shuf_audio_after_codec = compress_decompress_array(shuf_audio, fs, codec=codec, bitrate=bitrate)
#     #     shuf_dec_after_codec = process_mdct_blocks(shuf_audio_after_codec, do_shuffle_dec, key_bytes, shuffle_nonce)
#     #     out_dir = get_after_codec_dir("shuffle", codec, bitrate)
#     #     out_name = f"test_sound{i}_shuffle_dec.wav"
#     #     write_wav(os.path.join(out_dir, out_name), fs, shuf_dec_after_codec)
#
#     # # signflip
#
#     # for codec, bitrate in CODEC_VARIANTS:
#     #     sign_audio_after_codec = compress_decompress_array(sign_audio, fs, codec=codec, bitrate=bitrate)
#     #     sign_dec_after_codec = process_mdct_blocks(sign_audio_after_codec, do_signflip, key_bytes, signflip_nonce)
#     #     out_dir = get_after_codec_dir("signflip", codec, bitrate)
#     #     out_name = f"test_sound{i}_signflip_dec.wav"
#     #     write_wav(os.path.join(out_dir, out_name), fs, sign_dec_after_codec)
#
#     return f"done: test_sound{i}.wav"

key_bytes = load_key_bytes(KEY_PATH)
wrong_key_bytes = b"wrong-key"



# ===========================================================================
# NEW EXPERIMENTS. Legacy code above is kept intact for comparison.
# Old invert / shuffle / signflip datasets are already calculated.
# Running THIS FILE without arguments computes ONLY new operations below.
# New outputs are ONLY in new_experiments/assets/. Old assets/ is never written.
# ===========================================================================
from pathlib import Path as _Path
import csv as _csv

# Правильный ключ загружается из KEY_PATH выше — НЕ МЕНЯЛИ.
# Неправильный ключ для контрольного эксперимента задан ниже в wrong_key_bytes.
# Все режимы шифруют только 2500 вещественных MDCT-коэффициентов.
# adaptive-orthomix — не байтовая подстановка; это модифицированный Гивенс.
# Внешний nirs_timepack_v2 / crypto._experimental_ops не требуется.
COEFFICIENT_SCALE = 32768.0
# Только старый rot-xor: 8-битное квантование с потерями.
# adaptive-orthomix НЕ использует этот предел и НЕ квантует MDCT-коэффициенты.
ROT_XOR_BYTE_LIMIT = 0.65

# Параметры адаптивного ортогонального вращения, r измеряется после /32768.
# AES S-box используется ТОЛЬКО для формирования угла, не для подстановки коэффициентов.
ADAPTIVE_RADIUS_STEP = 0.25
ADAPTIVE_MAX_ANGLE = 2.3


def _gf256_mul(a, b):
    """Умножение в GF(2^8) для построения стандартной AES S-box."""
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        a = ((a << 1) ^ (0x11B if a & 0x80 else 0)) & 255
        b >>= 1
    return result


def _gf256_pow(x, exponent):
    result = 1
    while exponent:
        if exponent & 1:
            result = _gf256_mul(result, x)
        x = _gf256_mul(x, x)
        exponent >>= 1
    return result


def _make_aes_sbox():
    table = np.empty(256, dtype=np.uint8)
    for x in range(256):
        inv = _gf256_pow(x, 254) if x else 0
        y = inv
        for shift in range(1, 5):
            y ^= ((inv << shift) | (inv >> (8 - shift))) & 255
        table[x] = y ^ 0x63
    return table


_AES_SBOX = _make_aes_sbox()
_AES_SBOX_INV = np.empty(256, dtype=np.uint8)
_AES_SBOX_INV[_AES_SBOX] = np.arange(256, dtype=np.uint8)


def _sbox16_one_block(values, rng, decrypt, quant_scale):
    """AES S-box на ДВУХ байтах знакового int16 MDCT-коэффициента.

    quant_scale (целых значений на единицу MDCT) известен из метаданных.
    Побайтовая подстановка обратима точно над int16, но после IMDCT+MDCT
    квантованные числа могут измениться. Это эксперимент, не обещание
    точного восстановления звука после перекрывающихся блоков.
    """
    if not np.isfinite(quant_scale) or quant_scale <= 0.0:
        raise ValueError(f'Некорректный sbox-16 quant_scale: {quant_scale}')
    scaled = np.asarray(values, dtype=np.float64) * quant_scale
    if not np.all(np.isfinite(scaled)):
        raise ValueError('sbox-16: нечисловые MDCT-коэффициенты')
    if decrypt:
        # После IMDCT -> MDCT коэффициенты могут отличаться и выходить из
        # int16. Жёстко ограничиваем лишь вход обратной S-box, чтобы эксперимент
        # не аварийно завершался; это НЕ возвращает потерянную информацию.
        scaled = np.clip(scaled, -32768.0, 32767.0)
    elif np.any(np.abs(scaled) > 32767.0):
        raise ValueError('sbox-16: int16 overflow при шифровании — увеличьте запас квантования')
    q = np.rint(scaled).astype(np.int16)
    # Используем uint16 как битовое представление int16 (без зависимости от
    # порядка байтов в операционной системе).
    bits = q.view(np.uint16).astype(np.uint32)
    lo = (bits & 255).astype(np.uint8)
    hi = ((bits >> 8) & 255).astype(np.uint8)
    n = len(q)
    k1 = rng.integers(0, 256, size=(2, n), dtype=np.uint8)
    k2 = rng.integers(0, 256, size=(2, n), dtype=np.uint8)
    if decrypt:
        lo2 = np.bitwise_xor(_AES_SBOX_INV[np.bitwise_xor(lo, k2[0])], k1[0])
        hi2 = np.bitwise_xor(_AES_SBOX_INV[np.bitwise_xor(hi, k2[1])], k1[1])
    else:
        lo2 = np.bitwise_xor(_AES_SBOX[np.bitwise_xor(lo, k1[0])], k2[0])
        hi2 = np.bitwise_xor(_AES_SBOX[np.bitwise_xor(hi, k1[1])], k2[1])
    combined = lo2.astype(np.uint16) | (hi2.astype(np.uint16) << 8)
    return combined.view(np.int16).astype(np.float64) / quant_scale


def _int16_scale_for_file(samples):
    """Скаляр масштаба для всего WAV, чтобы ни один исходный MDCT не обрезался.

    Значение записывается в JSON и используется одинаково при шифровании и
    расшифровании, включая ветви после MP3 и Opus.
    """
    samples = np.asarray(samples, dtype=np.float64)
    peak = 0.0
    for start in range(0, len(samples) - N + 1, M):
        peak = max(peak, float(np.max(np.abs(mdct4(samples[start:start+N] * MDCT_WINDOW)))))
    return 30000.0 / max(peak, 1.0)  # запас до 32767 от ошибок округления


def _rotate_left_16(values, amounts):
    """Циклическое вращение 16-битных слов влево; вычисления в uint32."""
    v = np.asarray(values, dtype=np.uint32)
    r = np.asarray(amounts, dtype=np.uint32)
    return (((v << r) | (v >> (16 - r))) & 65535).astype(np.uint16)


def _rotate_right_16(values, amounts):
    """Циклическое вращение 16-битных слов вправо."""
    v = np.asarray(values, dtype=np.uint32)
    r = np.asarray(amounts, dtype=np.uint32)
    return (((v >> r) | (v << (16 - r))) & 65535).astype(np.uint16)


def _rot_xor16_one_block(values, rng, decrypt, quant_scale):
    """Rot-XOR над int16 MDCT-коэффициентами: q -> XOR(key) -> ROTL16 -> XOR-chain.

    Вход/выход values в единицах MDCT. Шаг квантования = 1 / quant_scale.
    Обратная цепочка: c[i] XOR c[i-1], ROR16, XOR(key). iv для c[-1].
    Клиппинг допустим только при дешифровании, где он означает потерю данных.
    """
    if not np.isfinite(quant_scale) or quant_scale <= 0:
        raise ValueError(f'rot-xor-16: invalid quant_scale={quant_scale}')
    scaled = np.asarray(values, dtype=np.float64) * quant_scale
    if not np.all(np.isfinite(scaled)):
        raise ValueError('rot-xor-16: non-finite MDCT coefficients')
    if decrypt:
        scaled = np.clip(scaled, -32768, 32767)
    elif np.any((scaled < -32768.0) | (scaled > 32767.0)):
        raise ValueError('rot-xor-16: int16 overflow before encryption')

    q = np.rint(scaled).astype(np.int16)
    data = q.view(np.uint16)
    n = len(data)
    mask = rng.integers(0, 65536, size=n, dtype=np.uint16)
    shifts = rng.integers(1, 16, size=n, dtype=np.uint16)
    iv = np.uint16(rng.integers(0, 65536))

    if decrypt:
        prev = np.empty_like(data)
        prev[0] = iv
        prev[1:] = data[:-1]
        rotated = np.bitwise_xor(data, prev)
        out_bits = np.bitwise_xor(_rotate_right_16(rotated, shifts), mask)
    else:
        rotated = _rotate_left_16(np.bitwise_xor(data, mask), shifts)
        # c[i] = iv XOR rotated[0] XOR ... XOR rotated[i]
        out_bits = np.bitwise_xor(np.bitwise_xor.accumulate(rotated), iv)

    out_q = out_bits.astype(np.uint16).view(np.int16)
    return out_q.astype(np.float64) / quant_scale


def _coefficient_to_byte(values):
    """8-битное квантование только для rot-xor (с потерями)."""
    scaled = 128.0 * (1.0 + np.clip(values / ROT_XOR_BYTE_LIMIT, -1.0, 1.0))
    return np.clip(np.floor(scaled), 0, 255).astype(np.uint8)


def _byte_to_coefficient(values):
    return ROT_XOR_BYTE_LIMIT * ((values.astype(np.float64) + 0.5) / 128.0 - 1.0)


def _rotate_left_8(values, amounts):
    v, r = values.astype(np.uint16), amounts.astype(np.uint16)
    return (((v << r) | (v >> (8 - r))) & 255).astype(np.uint8)


def _rotate_right_8(values, amounts):
    v, r = values.astype(np.uint16), amounts.astype(np.uint16)
    return (((v >> r) | (v << (8 - r))) & 255).astype(np.uint8)


def _modify_new_mdct(coeff, block_index, operation, key, decrypt, file_number, sbox16_scale=None):
    """Встроенные операции: orthomix / amplitude-sign / adaptive-orthomix / sbox-16 / rot-xor / rot-xor-16.

    Генерация параметров детерминирована ключом, номером файла,
    названием шифра и индексом MDCT-блока (как nonce в старом скрипте).
    """
    values = np.asarray(coeff, dtype=np.float64) / COEFFICIENT_SCALE
    nonce = f"test_sound{file_number}_{operation}_block_{block_index}"
    rng = rng_from_key(key, nonce)
    count = len(values)

    if operation == 'orthomix':
        # Ортогональные вращения Гивенса между ключезависимыми парами.
        indices = rng.permutation(count)
        pairs = count // 2
        first, second = indices[:2*pairs:2], indices[1:2*pairs:2]
        theta = rng.uniform(-np.pi, np.pi, size=pairs)
        c, s = np.cos(theta), np.sin(theta)
        a, b = values[first].copy(), values[second].copy()
        output = values.copy()
        if decrypt:
            output[first] = c*a + s*b
            output[second] = -s*a + c*b
        else:
            output[first] = c*a - s*b
            output[second] = s*a + c*b

    elif operation == 'amplitude-sign':
        # Модули перемешиваются, знаки позиций изменяются независимо.
        indices = rng.permutation(count)
        inverse = np.argsort(indices)
        mask = rng.choice(np.array([-1.0, 1.0]), size=count)
        eps = 1e-8  # сохраняет знак даже когда переставляемый модуль == 0
        if decrypt:
            original_signs = np.where(values < 0, -1.0, 1.0) * mask
            magnitudes = np.maximum(np.abs(values)[inverse] - eps, 0.0)
            output = magnitudes * original_signs
        else:
            original_signs = np.where(values < 0, -1.0, 1.0)
            output = (np.abs(values)[indices] + eps) * original_signs * mask

    elif operation == 'adaptive-orthomix':
        # МОДИФИЦИРОВАННОЕ вращение Гивенса, а НЕ S-box-подстановка.
        # Пары коэффициентов выбираются по ключу. Угол зависит от ключа
        # и радиуса пары r = sqrt(a*a + b*b). Радиус сохраняется при вращении,
        # поэтому получатель вычисляет тот же угол для обратного поворота.
        # Здесь нет восьмибитного квантования коэффициентов MDCT.
        indices = rng.permutation(count)
        pairs = count // 2
        first = indices[:2 * pairs:2]
        second = indices[1:2 * pairs:2]
        a = values[first].copy()
        b = values[second].copy()
        radius = np.hypot(a, b)

        # S-box используется как нелинейная функция для вычисления УГЛА.
        # Плавное смешивание соседних значений уменьшает резкие скачки
        # при малых погрешностях после PCM16 и кодеков.
        offset = int(rng.integers(0, 256))
        position = np.mod(radius / ADAPTIVE_RADIUS_STEP + offset, 256.0)
        left = np.floor(position).astype(np.int64)
        frac = position - left
        blend = frac * frac * (3.0 - 2.0 * frac)  # smoothstep
        substituted = ((1.0 - blend) * _AES_SBOX[left].astype(np.float64)
                       + blend * _AES_SBOX[(left + 1) % 256].astype(np.float64))
        theta = ADAPTIVE_MAX_ANGLE * (2.0 * substituted / 255.0 - 1.0)
        c, t = np.cos(theta), np.sin(theta)
        output = values.copy()
        if decrypt:
            output[first] = c * a + t * b
            output[second] = -t * a + c * b
        else:
            output[first] = c * a - t * b
            output[second] = t * a + c * b

    elif operation == 'sbox-16':
        if sbox16_scale is None:
            raise ValueError('sbox-16: должен быть передан масштаб квантования файла')
        # values до вызова делятся на COEFFICIENT_SCALE, поэтому здесь
        # возвращаем коэффициенты в исходных единицах и нормируем результат.
        output = _sbox16_one_block(values * COEFFICIENT_SCALE, rng, decrypt, sbox16_scale)
        output /= COEFFICIENT_SCALE

    elif operation == 'rot-xor-16':
        if sbox16_scale is None:
            raise ValueError('rot-xor-16: не передан масштаб квантования')
        output = _rot_xor16_one_block(values * COEFFICIENT_SCALE,
                                      rng, decrypt, sbox16_scale)
        output /= COEFFICIENT_SCALE

    elif operation == 'rot-xor':
        # Rotation + XOR chain над 8-битными квантованными коэффициентами.
        k = rng.integers(0, 256, size=count, dtype=np.uint8)
        shifts = rng.integers(1, 8, size=count, dtype=np.uint8)
        iv = int(rng.integers(0, 256))
        if decrypt:
            data = _coefficient_to_byte(values)
            previous = np.empty_like(data)
            previous[0] = iv
            previous[1:] = data[:-1]
            rotated = np.bitwise_xor(data, previous)
            output = _byte_to_coefficient(np.bitwise_xor(_rotate_right_8(rotated, shifts), k))
        else:
            data = _coefficient_to_byte(values)
            rotated = _rotate_left_8(np.bitwise_xor(data, k), shifts)
            encrypted = np.empty_like(rotated)
            prev = iv
            for i in range(count):
                prev ^= int(rotated[i])
                encrypted[i] = prev
            output = _byte_to_coefficient(encrypted)
    else:
        raise ValueError(f'Неизвестная операция MDCT: {operation}')

    return np.asarray(output, dtype=np.float64) * COEFFICIENT_SCALE


def _apply_new_mdct(signal, operation, key, decrypt, file_number, sbox16_scale=None):
    # Исходные M=2500, N=5000, окно и MDCT/IMDCT НЕ МЕНЯЛИ.
    # Те же ключевые параметры, что у соответствующего режима с PCM16-каналом.
    mdct_operation = FLOAT_CODEC_OPERATIONS.get(operation, operation)

    def callback(coeff, block_index):
        return _modify_new_mdct(coeff, block_index, mdct_operation, key, decrypt, file_number, sbox16_scale)
    return process_mdct_blocks(signal, callback)


def _new_paths(operation, file_number):
    # Paths come from the explicit OUT_ENC_*/OUT_DEC_*/DEC_AFTER_CODEC_* constants above.
    d = NEW_OUTPUT_DIRS[operation]
    name = f'test_sound{file_number}_{operation}'
    return {
        'encrypted': _Path(d['encrypted']) / f'{name}_enc.wav',
        # PCM16 preview ONLY for listening. Never use this WAV for decryption or metrics.
        'preview': _Path(d['encrypted']) / f'{name}_enc_preview_pcm16.wav',
        'meta': _Path(d['encrypted']) / f'{name}_meta.json',
        'decrypted': _Path(d['decrypted']) / f'{name}_dec.wav',
        'wrong': _Path(d['wrong']) / f'{name}_dec.wav',
        'codec': lambda codec, bitrate: _Path(NEW_CODEC_DIRS[(operation, codec, bitrate)]) / f'{name}_dec.wav',
        'codec_wrong': lambda codec, bitrate: _Path(NEW_CODEC_WRONG_DIRS[(operation, codec, bitrate)]) / f'{name}_dec.wav',
    }


def _write_if_needed(path, fs, samples, float64=False):
    path = _Path(path)
    if not RECALCULATE_EXISTING and path.is_file():
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    if float64:
        wavfile.write(str(path), fs, np.asarray(samples, dtype=np.float64))
    else:
        write_wav(str(path), fs, samples)


def _case_finished(paths):
    if RECALCULATE_EXISTING:
        return False
    expected=[paths['encrypted'],paths['decrypted'],paths['wrong']]
    if paths['encrypted'].parent.name in INT16_MDCT_OPERATIONS:
        expected.extend((paths['meta'], paths['preview']))
    for codec, bitrate in CODEC_VARIANTS:
        expected.append(paths['codec'](codec,bitrate))
        if SAVE_CODEC_WRONG_KEY:
            expected.append(paths['codec_wrong'](codec,bitrate))
    return all(_Path(p).is_file() for p in expected)


def _prepare_new_output_dirs():
    # Создаём каталоги ТОЛЬКО выбранных NEW_OPERATIONS.
    for operation in NEW_OPERATIONS:
        dirs=NEW_OUTPUT_DIRS[operation]
        for path in dirs.values():
            os.makedirs(path,exist_ok=True)
        for codec,bitrate in CODEC_VARIANTS:
            os.makedirs(NEW_CODEC_DIRS[(operation,codec,bitrate)],exist_ok=True)
            if SAVE_CODEC_WRONG_KEY:
                os.makedirs(NEW_CODEC_WRONG_DIRS[(operation,codec,bitrate)],exist_ok=True)


def _process_new_one(index):
    source=_Path(DATASET_DIR)/f'test_sound{index}.wav'
    done=skipped=0;errors=[]
    if not source.is_file():
        return done,skipped,[('',f'missing input: {source}')]
    fs,samples=wavfile.read(str(source))
    if samples.ndim>1:
        samples=samples[:,0]
    samples=np.asarray(samples,dtype=np.float64)
    for operation in NEW_OPERATIONS:
        try:
            paths=_new_paths(operation,index)
            if _case_finished(paths):
                skipped+=1
                continue
            if operation in INT16_MDCT_OPERATIONS:
                float_codec = operation in FLOAT_CODEC_OPERATIONS
                # Для побитовых алгоритмов сохраняем Float64 и ВНОВЬ читаем WAV с диска.
                # Иначе мы тестируем только RAM-передачу, а не сохранённый шифротекст.
                import json as _json
                saved_cipher = paths['encrypted'].is_file()
                saved_meta = paths['meta'].is_file()
                if not RECALCULATE_EXISTING and saved_cipher != saved_meta:
                    raise ValueError(f'{operation}: остался неполный результат (encrypted WAV без meta JSON или наоборот); '
                                     'проверьте/удалите оба файла именно этого режима')
                if not RECALCULATE_EXISTING and saved_cipher and saved_meta:
                    info = _json.loads(paths['meta'].read_text(encoding='utf-8'))
                    scale = float(info['quant_scale'])
                    gain = float(info['codec_gain'])
                else:
                    scale = _int16_scale_for_file(samples)
                    encrypted = _apply_new_mdct(samples, operation, key_bytes, False, index, scale)
                    max_peak = float(np.max(np.abs(encrypted))) if len(encrypted) else 0.0
                    gain = min(1.0, 30000.0/max_peak) if max_peak > 0 else 1.0
                    _write_if_needed(paths['encrypted'], fs, encrypted, float64=True)
                    paths['meta'].parent.mkdir(parents=True,exist_ok=True)
                    paths['meta'].write_text(_json.dumps({
                        'mode': operation, 'quant_scale': scale, 'codec_gain': gain,
                        'ciphertext_peak': max_peak, 'encrypted_format': 'IEEE float64 WAV',
                        'note': (
                            'No clipping on encrypted WAV. Codec uses normalized Float32 PCM.'
                            if float_codec else
                            'No clipping on encrypted WAV. Codec uses normalized PCM16.'
                        ),
                    }, indent=2), encoding='utf-8')

                fs_cipher, ciphertext = wavfile.read(str(paths['encrypted']))
                if fs_cipher != fs or ciphertext.dtype != np.float64 or len(ciphertext) != len(samples):
                    raise ValueError(f'{operation}: saved encrypted WAV must be mono Float64, same rate and length')
                if not (np.isfinite(scale) and scale > 0 and np.isfinite(gain) and 0 < gain <= 1):
                    raise ValueError(f'{operation}: invalid scale/gain in metadata')
                ciphertext = ciphertext.astype(np.float64, copy=False)
                # A second, Windows-compatible PCM16 WAV for listening only.
                # 'gain' was calculated from the full float64 ciphertext peak,
                # so this has no clipping and does NOT change encrypted Float64 WAV.
                # If all other results already exist, only the missing preview is written.
                _write_if_needed(paths['preview'], fs, ciphertext * gain)
                # Без кодека не округляем зашифрованный WAV до PCM16.
                # Только после обратного MDCT + выбранной операции пишем PCM16 через write_wav.
                if RECALCULATE_EXISTING or not paths['decrypted'].is_file():
                    clear = _apply_new_mdct(ciphertext, operation, key_bytes, True, index, scale)
                    if len(samples) > 2*N:
                        sl = slice(N, len(samples)-N)
                        corr = float(np.corrcoef(samples[sl], clear[sl])[0,1])
                        if not np.isfinite(corr) or corr < 0.9:
                            print(f'[{operation.upper()} WARNING] test_sound{index}: no-codec correlation={corr:.5f}', flush=True)
                    _write_if_needed(paths['decrypted'], fs, clear)
                if RECALCULATE_EXISTING or not paths['wrong'].is_file():
                    wrong = _apply_new_mdct(ciphertext, operation, wrong_key_bytes, True, index, scale)
                    _write_if_needed(paths['wrong'], fs, wrong)
                for codec,bitrate in CODEC_VARIANTS:
                    correct_path=paths['codec'](codec,bitrate)
                    wrong_path=paths['codec_wrong'](codec,bitrate)
                    if not RECALCULATE_EXISTING and correct_path.is_file() and (not SAVE_CODEC_WRONG_KEY or wrong_path.is_file()):
                        continue
                    if float_codec:
                        wire = compress_decompress_float(ciphertext, fs, codec, bitrate, gain)
                        output_fs = fs
                    else:
                        wire = compress_decompress_array(ciphertext * gain, fs, codec=codec,
                                                         bitrate=bitrate, target_fs=16000) / gain
                        output_fs = 16000
                    recovered = _apply_new_mdct(wire, operation, key_bytes, True, index, scale)
                    _write_if_needed(correct_path, output_fs, recovered)
                    if SAVE_CODEC_WRONG_KEY:
                        recovered_wrong = _apply_new_mdct(wire, operation, wrong_key_bytes, True, index, scale)
                        _write_if_needed(wrong_path, output_fs, recovered_wrong)
            else:
                encrypted=_apply_new_mdct(samples,operation,key_bytes,False,index)
                if operation == 'adaptive-orthomix':
                    overflow = int(np.count_nonzero((encrypted < -32768) | (encrypted > 32767)))
                    if overflow:
                        print(f'[ADAPTIVE-ORTHOMIX PCM CLIP] test_sound{index}: '
                              f'{overflow}/{len(encrypted)} samples; '
                              f'peak={np.max(np.abs(encrypted)):.1f}', flush=True)
                ciphertext=clip_to_int16(encrypted).astype(np.float64)
                _write_if_needed(paths['encrypted'],fs,ciphertext)
                clear=_apply_new_mdct(ciphertext,operation,key_bytes,True,index)
                _write_if_needed(paths['decrypted'],fs,clear)
                wrong=_apply_new_mdct(ciphertext,operation,wrong_key_bytes,True,index)
                _write_if_needed(paths['wrong'],fs,wrong)
                for codec,bitrate in CODEC_VARIANTS:
                    correct_path=paths['codec'](codec,bitrate)
                    wrong_path=paths['codec_wrong'](codec,bitrate)
                    if not RECALCULATE_EXISTING and correct_path.is_file() and (not SAVE_CODEC_WRONG_KEY or wrong_path.is_file()):
                        continue
                    wire=compress_decompress_array(ciphertext,fs,codec=codec,bitrate=bitrate,target_fs=16000)
                    recovered=_apply_new_mdct(wire,operation,key_bytes,True,index)
                    _write_if_needed(correct_path,16000,recovered)
                    if SAVE_CODEC_WRONG_KEY:
                        recovered_wrong=_apply_new_mdct(wire,operation,wrong_key_bytes,True,index)
                        _write_if_needed(wrong_path,16000,recovered_wrong)
            done+=1
        except Exception as error:
            errors.append((operation,f'{type(error).__name__}: {error}'))
    return done,skipped,errors



def _run_new_measurements():
    if not _Path(DATASET_DIR).is_dir():
        raise FileNotFoundError(f'Датасет отсутствует: {DATASET_DIR}')
    if not _Path(KEY_PATH).is_file():
        raise FileNotFoundError(f'Ключ отсутствует: {KEY_PATH}')
    if key_bytes == wrong_key_bytes:
        raise ValueError('key.bin and hardcoded wrong key must differ')
    _prepare_new_output_dirs()
    print(f'START MDCT: {N_FILES} WAV, {MAX_WORKERS} процессов, '
          f'{len(NEW_OPERATIONS)} новых операций, {len(CODEC_VARIANTS)} кодеков',flush=True)
    print(f'OLD METHODS DISABLED; output ONLY to: {NEW_ASSETS_DIR}',flush=True)
    print(f'DATASET: {DATASET_DIR}',flush=True)
    print(f'KEY: {KEY_PATH}',flush=True)
    fails=[];complete=skipped=0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures={pool.submit(_process_new_one,i):i for i in range(1,N_FILES+1)}
        for fut in as_completed(futures):
            number=futures[fut]
            try:
                n,k,errors=fut.result()
                complete+=n;skipped+=k
                for operation,error in errors:
                    fails.append((number,operation,error))
            except Exception as e:
                fails.append((number,'',f'{type(e).__name__}: {e}'))
            if (complete+skipped+len(fails))%25==0:
                print(f'processed={complete}, skipped={skipped}, errors={len(fails)}',flush=True)
    if fails:
        error_file=_Path(NEW_RESULTS_DIR)/'new_mdct_errors.csv'
        with error_file.open('w',newline='',encoding='utf-8-sig') as f:
            w=_csv.writer(f,delimiter=';')
            w.writerow(('file','operation','error'))
            w.writerows(fails)
        print(f'ERROR LIST: {error_file}',flush=True)
    print(f'FINISH MDCT: done={complete}; skipped={skipped}; failed={len(fails)}',flush=True)
    if fails:
        raise SystemExit(1)


if __name__=='__main__':
    _run_new_measurements()
