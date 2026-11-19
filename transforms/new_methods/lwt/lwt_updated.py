#!/usr/bin/env python3
"""Эксперименты с lifting wavelet transform (LWT).

Один блок равен 320 отсчётам при 16 кГц. Прямое и обратное преобразования реализованы
локально на основе lifting CDF 5/3 (симметричные границы). Для запуска нужны NumPy, SciPy и,
при включённых MP3/Opus, FFmpeg. Шифротекст хранится в Float64 WAV.
Рядом в assets/diagnostics сохраняются масштаб и длина для восстановления.
По умолчанию запускаются семь методов шифрования LWT. Три дополнительных
операции над парами коэффициентов доступны через --operations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import multiprocessing
import os
import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np
from scipy import signal
from scipy.io import wavfile

TRANSFORM = 'lwt'
ENGINE_VERSION = 'lwt-cdf53-lifting-f64-v1'
BASIS_NAME = 'cdf53-symmetric-multilevel-lifting'
OPERATIONS = ('invert', 'signflip', 'shuffle', 'orthomix', 'amplitude-sign',
              'sbox', 'rot-xor')
PAIRED_OPERATIONS = ('phase-offset', 'amplitude-permutation', 'amplitude-phase')
SUPPORTED_OPERATIONS = OPERATIONS + PAIRED_OPERATIONS
BIT_OPERATIONS = frozenset(('sbox', 'rot-xor'))

# ========== НАСТРОЙКИ ЭКСПЕРИМЕНТА ==========
FRAME = 320                    # 20 мс при 16 кГц; шаг тоже 320, перекрытия нет
VOICE_RATE = 16000
CODECS = (("mp3", "64k"), ("mp3", "128k"), ("mp3", "320k"),
          ("opus", "16k"), ("opus", "48k"), ("opus", "96k"))
# Расположение скрипта: nirs/total/transforms/new_methods/lwt/lwt.py
BASE_DIR = Path(__file__).resolve().parent
TRANSFORMS_DIR = BASE_DIR.parent.parent
ROOT_DIR = TRANSFORMS_DIR.parent.parent

# Сначала структура со скриншота, затем расположения из прежних скриптов.
DATASET_PATHS = (
    TRANSFORMS_DIR / "dataset" / "test_sounds3",
    (BASE_DIR / "../../../dataset/test_sounds3").resolve(),
    ROOT_DIR / "dataset" / "test_sounds3",
)
DATASET_DIR = next((p for p in DATASET_PATHS if p.is_dir()), DATASET_PATHS[0])
KEY_PATH = TRANSFORMS_DIR / "mdct" / "key.bin"
WRONG_KEY = b"wrong-key"  # задан прямо здесь; wrong_key.bin не читается
FFMPEG_BIN = (ROOT_DIR / "ffmpeg" / "bin" / "ffmpeg.exe"
              if os.name == "nt" else Path("ffmpeg"))
OUT_DIR = BASE_DIR  # результаты в assets/ рядом с этим скриптом
N_FILES = 17
MAX_WORKERS = 8
OVERWRITE = False
WRITE_WRONG_KEY = True
INPUT_GAIN = 1.0
MAX_LAG_MS = 250.0
WIRE_PEAK = 0.95                # запас до полного уровня, БЕЗ обрезания отсчётов
QUANT_TARGET = 30000.0          # int16, один масштаб на файл, запас до 32767
SIGN_EPS = 1e-8                 # сохраняет знак, когда на его позицию переносится нулевой модуль
PHASE_EPS = 1e-4                # для аргумента комплексной пары при переносе нулевого модуля
# ===========================================


def split_comma(text):
    return tuple(x.strip().lower() for x in text.split(',') if x.strip())


def to_float(wav):
    """PCM -> float64. Без нормировки по пику и без клиппинга."""
    a = np.asarray(wav)
    if a.dtype == np.uint8:
        a = (a.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(a.dtype, np.signedinteger):
        a = a.astype(np.float64) / float(-int(np.iinfo(a.dtype).min))
    elif np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    else:
        raise ValueError(f'Unsupported WAV dtype: {a.dtype}')
    if a.ndim == 2:
        a = a.mean(axis=1)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError('Нужен непустой конечный моно/стерео WAV')
    return a


def load_mono_16k(path):
    rate, raw = wavfile.read(path)
    audio = to_float(raw)
    if rate != VOICE_RATE:
        common = gcd(int(rate), VOICE_RATE)
        audio = signal.resample_poly(audio, VOICE_RATE // common, int(rate) // common)
    return np.asarray(audio, dtype=np.float64)


def pcm16(x):
    """Только конечный WAV для прослушивания. Единый масштаб 32768 туда/обратно."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError('Нечисловые отсчёты при сохранении PCM16')
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype('<i2')


def save_wav(path, data, *, floating=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError(f'Нечисловые отсчёты: {path}')
    wavfile.write(path, VOICE_RATE, data if floating else pcm16(data))


# Точный обратимый lifting CDF 5/3 на вещественных отсчётах.
# Никакой матрицы LWT, FFT или пересохранения коэффициентов в PCM16.
# Это конкретная реализация LWT; без crypto/adapter.py нельзя подтвердить,
# что его исходная lifting-схема/вейвлет совпадают с этой.
LWT_LEVELS = 6  # 320 -> 160 -> 80 -> 40 -> 20 -> 10 -> 5


def _lifting_forward(values):
    values = np.asarray(values, dtype=np.float64)
    even = values[..., ::2].copy()
    odd = values[..., 1::2].copy()
    # Predict: odd_i -= (even_i + even_{i+1})/2.
    right = np.concatenate((even[..., 1:], even[..., -1:]), axis=-1)
    detail = odd - 0.5 * (even + right)
    # Update: even_i += (detail_{i-1} + detail_i)/4.
    left = np.concatenate((detail[..., :1], detail[..., :-1]), axis=-1)
    approx = even + 0.25 * (left + detail)
    return approx, detail


def _lifting_inverse(approx, detail):
    approx = np.asarray(approx, dtype=np.float64)
    detail = np.asarray(detail, dtype=np.float64)
    left = np.concatenate((detail[..., :1], detail[..., :-1]), axis=-1)
    even = approx - 0.25 * (left + detail)
    right = np.concatenate((even[..., 1:], even[..., -1:]), axis=-1)
    odd = detail + 0.5 * (even + right)
    out = np.empty(approx.shape[:-1] + (approx.shape[-1] * 2,), dtype=np.float64)
    out[..., ::2], out[..., 1::2] = even, odd
    return out


def lwt(blocks):
    """6 уровней lifting CDF 5/3, упаковка [a6,d6,d5,...,d1]."""
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.shape[-1] != FRAME:
        raise ValueError(f'Для LWT требуется блок из {FRAME} отсчётов')
    approx = blocks
    details = []
    for _ in range(LWT_LEVELS):
        approx, detail = _lifting_forward(approx)
        details.append(detail)
    return np.concatenate([approx] + details[::-1], axis=-1)


def ilwt(coefficients):
    """Обратный CDF 5/3 lifting: восстановление всех 320 отсчётов."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape[-1] != FRAME:
        raise ValueError(f'Для ILWT требуется {FRAME} коэффициентов')
    size = FRAME // (2 ** LWT_LEVELS)
    approx = coefficients[..., :size]
    offset = size
    for _ in range(LWT_LEVELS):
        detail = coefficients[..., offset:offset + size]
        approx = _lifting_inverse(approx, detail)
        offset += size
        size *= 2
    return approx


def frame_rng(key, file_number, operation, frame_index):
    """Воспроизводимые параметры эксперимента, разделённые по файлу/методу/блоку.

    PCG64 здесь используется как генератор параметров исследования.
    Это не реализация стандартного криптографического протокола.
    """
    context = f'{ENGINE_VERSION}|{file_number}|{operation}|{frame_index}'.encode('utf-8')
    seed = int.from_bytes(hmac.new(key, context, hashlib.sha256).digest(), 'big')
    return np.random.Generator(np.random.PCG64(seed))


def _gf256_mul(a, b):
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
        value = inv
        for shift in range(1, 5):
            value ^= ((inv << shift) | (inv >> (8 - shift))) & 255
        table[x] = value ^ 0x63
    return table


AES_SBOX = _make_aes_sbox()
AES_SBOX_INV = np.empty(256, dtype=np.uint8)
AES_SBOX_INV[AES_SBOX] = np.arange(256, dtype=np.uint8)


def rotate_left16(words, shifts):
    values = np.asarray(words, dtype=np.uint32)
    shifts = np.asarray(shifts, dtype=np.uint32)
    return (((values << shifts) | (values >> (16 - shifts))) & 65535).astype(np.uint16)


def rotate_right16(words, shifts):
    values = np.asarray(words, dtype=np.uint32)
    shifts = np.asarray(shifts, dtype=np.uint32)
    return (((values >> shifts) | (values << (16 - shifts))) & 65535).astype(np.uint16)


def modify_words(words, rng, operation, decrypt=False):
    """Точные перестановки 16-битных слов. Без float, FFT и квантования."""
    words = np.asarray(words, dtype=np.uint16)
    size = len(words)
    if operation == 'rot-xor':
        mask = rng.integers(0, 65536, size=size, dtype=np.uint16)
        shifts = rng.integers(1, 16, size=size, dtype=np.uint16)
        iv = np.uint16(rng.integers(0, 65536))
        if decrypt:
            previous = np.empty_like(words)
            previous[0] = iv
            previous[1:] = words[:-1]
            return rotate_right16(words ^ previous, shifts) ^ mask
        rotated = rotate_left16(words ^ mask, shifts)
        return np.bitwise_xor.accumulate(rotated) ^ iv
    if operation == 'sbox':
        first = rng.integers(0, 256, size=(2, size), dtype=np.uint8)
        second = rng.integers(0, 256, size=(2, size), dtype=np.uint8)
        low = (words & 255).astype(np.uint8)
        high = (words >> 8).astype(np.uint8)
        if decrypt:
            low = AES_SBOX_INV[low ^ second[0]] ^ first[0]
            high = AES_SBOX_INV[high ^ second[1]] ^ first[1]
        else:
            low = AES_SBOX[low ^ first[0]] ^ second[0]
            high = AES_SBOX[high ^ first[1]] ^ second[1]
        return low.astype(np.uint16) | (high.astype(np.uint16) << 8)
    raise ValueError(f'Не битовая операция: {operation}')


def native_rng(key, frame_index, domain):
    """Параметры прежних методов из nirs_timepack_v2, включая период 64 блока."""
    material = (b'NIRS_EXPERIMENTAL_V1\0' + key + b'\0LWTCipher\0'
                + domain.encode('ascii') + b'\0' + (int(frame_index) % 64).to_bytes(8, 'big'))
    digest = hashlib.sha512(material).digest()
    seed = np.random.SeedSequence(np.frombuffer(digest, dtype='<u4'), pool_size=8)
    return np.random.Generator(np.random.PCG64(seed))


def sign_mask(key, frame_index, count):
    return native_rng(key, frame_index, 'signflip').choice([-1.0, 1.0], size=count)


def paired_operation(values, operation, rng, decrypt, phase_eps):
    """z[j] = c[2j] + i*c[2j+1]. Угол относится к этой паре коэффициентов."""
    if values.size % 2:
        raise ValueError('Для комплексных пар требуется чётное число коэффициентов')
    pairs = values[::2] + 1j * values[1::2]
    if operation == 'phase-offset':
        angles = rng.uniform(-np.pi, np.pi, size=pairs.size)
        changed = pairs * np.exp(1j * (-angles if decrypt else angles))
    else:
        permutation = rng.permutation(pairs.size)
        phase = np.angle(pairs)
        if operation == 'amplitude-phase':
            angles = rng.uniform(-np.pi, np.pi, size=pairs.size)
            phase = phase - angles if decrypt else phase + angles
        if decrypt:
            radius = np.maximum(np.abs(pairs)[np.argsort(permutation)] - phase_eps, 0.0)
        else:
            radius = np.abs(pairs)[permutation] + phase_eps
        changed = radius * np.exp(1j * phase)
    result = np.empty_like(values)
    result[::2], result[1::2] = changed.real, changed.imag
    return result


def bit_operation(values, operation, rng, decrypt, scale, allow_corrupt, counters):
    """Равномерное квантование коэффициентов и обратимая операция на 16-битных словах."""
    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Нужен положительный масштаб квантования из JSON')
    rounded = np.rint(values * scale)
    if not np.all(np.isfinite(rounded)):
        raise ValueError('Переполнение при квантовании коэффициентов LWT')
    outside = (rounded < -32768) | (rounded > 32767)
    count = int(np.count_nonzero(outside))
    if count:
        if not (decrypt and allow_corrupt):
            raise ValueError(f'{operation}: {count} слов вне диапазона int16')
        # Принимаем и учитываем выход за диапазон после кодека с потерями.
        rounded = np.clip(rounded, -32768, 32767)
    if counters is not None:
        counters['out_of_range_words'] += count
    words = rounded.astype(np.int16).view(np.uint16)
    changed = modify_words(words, rng, operation, decrypt)
    return changed.view(np.int16).astype(np.float64) / scale


def modify_coefficients(values, operation, key, frame_index, *, file_number=0,
                        decrypt=False, quant_scale=None, sign_eps=SIGN_EPS,
                        phase_eps=PHASE_EPS, allow_corrupt=False, counters=None):
    """Прямая или обратная операция над вектором вещественных коэффициентов блока."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError('Нужен вектор конечных вещественных коэффициентов')
    count = values.size
    if operation == 'invert':
        return values[::-1].copy()
    if operation == 'signflip':
        return values * sign_mask(key, frame_index, count)
    if operation == 'shuffle':
        permutation = native_rng(key, frame_index, 'shuffle').permutation(count)
        return values[np.argsort(permutation) if decrypt else permutation]
    if operation == 'orthomix':
        rng = native_rng(key, frame_index, 'orthomix')
        permutation = rng.permutation(count)
        pairs = permutation[:2 * (count // 2)].reshape(-1, 2)
        left, right = pairs.T
        angles = rng.uniform(-np.pi, np.pi, size=len(pairs))
        cosine = np.cos(angles)
        sine = np.sin(-angles if decrypt else angles)
        result = values.copy()
        result[left] = cosine * values[left] - sine * values[right]
        result[right] = sine * values[left] + cosine * values[right]
        return result
    if operation == 'amplitude-sign':
        if not np.isfinite(sign_eps) or sign_eps <= 0:
            raise ValueError('sign_eps должен быть положительным конечным')
        permutation = native_rng(key, frame_index, 'magnitude').permutation(count)
        mask = sign_mask(key, frame_index, count)
        # Нулю назначаем знак +1: иначе sign(0) стёр бы чужой модуль.
        signs = np.where(values < 0, -1.0, 1.0)
        if decrypt:
            radius = np.maximum(np.abs(values)[np.argsort(permutation)] - sign_eps, 0.0)
        else:
            radius = np.abs(values)[permutation] + sign_eps
        return signs * mask * radius
    rng = frame_rng(key, file_number, operation, frame_index)
    if operation in BIT_OPERATIONS:
        return bit_operation(values, operation, rng, decrypt, quant_scale, allow_corrupt, counters)
    if operation in PAIRED_OPERATIONS:
        if not np.isfinite(phase_eps) or phase_eps <= 0:
            raise ValueError('phase_eps должен быть положительным конечным')
        return paired_operation(values, operation, rng, decrypt, phase_eps)
    raise ValueError(f'Неизвестная операция: {operation}')


def apply_to_frames(moments, operation, key, file_number, **options):
    """Модифицирует матрицу коэффициентов на месте, сохраняя исходные индексы кадров."""
    for index, row in enumerate(moments):
        moments[index] = modify_coefficients(row, operation, key, index,
                                             file_number=file_number, **options)


def layout_for(operation):
    return 'adjacent-real-pairs' if operation in PAIRED_OPERATIONS else 'real-wavelet-coefficients'


def encrypt_audio(samples, operation, key, file_number=0):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not samples.size or not np.all(np.isfinite(samples)):
        raise ValueError('Нужен непустой конечный одномерный сигнал')
    padded = np.pad(samples, (0, -samples.size % FRAME))
    moments = lwt(padded.reshape(-1, FRAME))
    scale = None
    if operation in BIT_OPERATIONS:
        scale = QUANT_TARGET / max(1.0, float(np.max(np.abs(moments))))
    apply_to_frames(moments, operation, key, file_number, quant_scale=scale)
    raw = ilwt(moments).ravel()
    peak = float(np.max(np.abs(raw)))
    wire_gain = min(1.0, WIRE_PEAK / peak) if peak else 1.0
    metadata = {
        'engine': ENGINE_VERSION, 'basis': BASIS_NAME, 'transform': TRANSFORM,
        'index': int(file_number), 'operation': operation, 'coefficient_layout': layout_for(operation),
        'frame': FRAME, 'hop': FRAME, 'overlap': 0, 'sample_rate': VOICE_RATE,
        'samples': samples.size, 'padded_samples': raw.size, 'raw_cipher_peak': peak,
        'quant_bits': 16 if operation in BIT_OPERATIONS else None,
        'quant_scale': scale, 'sign_eps': SIGN_EPS, 'phase_eps': PHASE_EPS,
        'wire_gain': wire_gain, 'wire_format': 'float64', 'lwt_levels': LWT_LEVELS,
    }
    return raw * wire_gain, metadata


def decrypt_audio(wire, metadata, key, *, allow_corrupt=False, return_counters=False):
    """Восстановление по шифротексту, параметрам JSON и ключу."""
    operation = metadata.get('operation')
    if (metadata.get('engine') != ENGINE_VERSION or metadata.get('transform') != TRANSFORM
            or metadata.get('basis') != BASIS_NAME or operation not in SUPPORTED_OPERATIONS
            or metadata.get('coefficient_layout') != layout_for(operation)
            or metadata.get('frame') != FRAME or metadata.get('hop') != FRAME
            or metadata.get('overlap') != 0 or metadata.get('sample_rate') != VOICE_RATE
            or metadata.get('lwt_levels') != LWT_LEVELS):
        raise ValueError('JSON не соответствует этой версии LWT')
    wire = np.asarray(wire, dtype=np.float64)
    size = metadata['samples']
    if (not isinstance(size, (int, np.integer)) or size < 1 or wire.ndim != 1
            or wire.size != metadata['padded_samples'] or wire.size != size + (-size) % FRAME
            or not np.all(np.isfinite(wire))):
        raise ValueError('Длина или содержимое WAV расходятся с полными блоками из JSON')
    gain = metadata['wire_gain']
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Некорректный wire_gain')
    moments = lwt((wire / gain).reshape(-1, FRAME))
    counters = {'out_of_range_words': 0}
    apply_to_frames(moments, operation, key, metadata['index'], decrypt=True,
                    quant_scale=metadata['quant_scale'], sign_eps=metadata['sign_eps'],
                    phase_eps=metadata['phase_eps'], allow_corrupt=allow_corrupt, counters=counters)
    result = ilwt(moments).ravel()[:size]
    return (result, counters) if return_counters else result


def decrypt_saved(encrypted_path, metadata_path, key, output_path=None):
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    rate, raw = wavfile.read(encrypted_path)
    if rate != metadata['sample_rate'] or raw.dtype != np.float64:
        raise ValueError('Для точного восстановления нужен исходный Float64 encrypted WAV')
    result = decrypt_audio(raw, metadata, key)
    if output_path is not None:
        save_wav(output_path, result)
    return result


def choose_ffmpeg(arg=None):
    if arg:
        path = Path(arg)
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(str(arg))
        if found:
            return found
        raise FileNotFoundError(f'FFmpeg не найден: {arg}')
    if FFMPEG_BIN.is_file():
        return str(FFMPEG_BIN.resolve())
    return shutil.which('ffmpeg')


def ffmpeg_call(command):
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=180)
    if completed.returncode:
        raise RuntimeError(f'FFmpeg failed ({completed.returncode}):\n{completed.stderr[-2500:]}')


def compress_and_decode(wave_data, codec, bitrate, ffmpeg_bin):
    """Float64 -> реальный кодек -> Float64. Предварительного PCM16/clip нет."""
    with tempfile.TemporaryDirectory(prefix='nirs_lwt_codec_') as td:
        tmp = Path(td)
        input_wav, output_wav = tmp / 'input.wav', tmp / 'decoded.wav'
        save_wav(input_wav, wave_data, floating=True)
        if codec == 'mp3':
            encoded = tmp / 'encoded.mp3'
            options = ['-ar', '44100', '-ac', '1', '-c:a', 'libmp3lame', '-b:a', bitrate]
        elif codec == 'opus':
            encoded = tmp / 'encoded.ogg'
            options = ['-ar', '48000', '-ac', '1', '-c:a', 'libopus', '-b:a', bitrate,
                       '-application', 'audio']
        else:
            raise ValueError(f'Unsupported codec: {codec}')
        base = [str(ffmpeg_bin), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y']
        ffmpeg_call(base + ['-i', str(input_wav)] + options + [str(encoded)])
        ffmpeg_call(base + ['-i', str(encoded), '-ar', str(VOICE_RATE), '-ac', '1',
                           '-c:a', 'pcm_f64le', str(output_wav)])
        rate, data = wavfile.read(output_wav)
        if rate != VOICE_RATE or data.ndim != 1 or not np.all(np.isfinite(data)):
            raise ValueError('Кодек вернул некорректный WAV')
        return data.astype(np.float64)


def align_wire(decoded, reference, max_lag_ms=MAX_LAG_MS, confidence_floor=0.12):
    """Выравниваем шифротекст, не расшифрованную речь. Это офлайн-оценка задержки."""
    m = int(max_lag_ms * VOICE_RATE / 1000)
    n = min(len(reference), len(decoded), 4 * VOICE_RATE)
    lag, confidence, state = 0, 0.0, 'short'
    if n >= 2 * FRAME:
        a = np.array(reference[:n], dtype=np.float64, copy=True)
        b = np.array(decoded[:min(len(decoded), n+m)], dtype=np.float64, copy=True)
        a -= a.mean()
        b -= b.mean()
        state = 'low_confidence'
        if np.max(np.abs(a)) > 1e-10 and np.max(np.abs(b)) > 1e-10:
            corr = signal.correlate(b, a, mode='full', method='fft')
            lags = signal.correlation_lags(len(b), len(a), mode='full')
            keep = (lags >= -m) & (lags <= m)
            best = int(np.argmax(corr[keep]))
            confidence = float(corr[keep][best] / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
            confidence = float(np.clip(confidence, 0, 1))
            if confidence >= confidence_floor:
                lag = int(lags[keep][best])
                state = 'ok'
    result = decoded[lag:] if lag >= 0 else np.pad(decoded, (-lag, 0))
    if len(result) < len(reference):
        result = np.pad(result, (0, len(reference)-len(result)))
    return result[:len(reference)], lag, confidence, state


def quality_metrics(reference, recovered):
    reference = np.asarray(reference, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)
    if recovered.shape != reference.shape or not np.all(np.isfinite(recovered)):
        raise ValueError('Некорректная длина или нечисловой восстановленный сигнал')
    error = recovered - reference
    mse = float(np.mean(error**2))
    power = float(np.mean(reference**2))
    return {
        'max_abs_error': float(np.max(np.abs(error))),
        'rmse': float(np.sqrt(mse)),
        'snr_db': float(10*np.log10(power/mse)) if power > 0 and mse > 0 else None,
        'reference_rms': float(np.sqrt(power)),
        'recovered_rms': float(np.sqrt(np.mean(recovered**2))),
        'first_frame_max_error': float(np.max(np.abs(error[:FRAME]))),
        'last_frame_max_error': float(np.max(np.abs(error[-FRAME:]))),
        'pcm16_saturated_samples': int(np.count_nonzero(
            (np.rint(recovered*32768) < -32768) | (np.rint(recovered*32768) > 32767))),
    }


def output_paths(out_root, transform, operation, idx):
    assets = Path(out_root) / 'assets'
    fn = f'test_sound{idx}_{operation}'
    return {
        'encrypted': assets / 'encrypted' / operation / f'{fn}_enc.wav',
        'preview': assets / 'encrypted' / operation / f'{fn}_enc_preview_pcm16.wav',
        'decrypted': assets / 'decrypted' / operation / f'{fn}_dec.wav',
        'decrypted_wrong_key': assets / 'decrypted_wrong_key' / operation / f'{fn}_dec.wav',
        'codec': lambda codec, rate: assets / 'decrypted_after_codec' /
                 f'{operation}_{codec}_{rate}' / f'{fn}_dec.wav',
        'codec_wrong': lambda codec, rate: assets / 'decrypted_after_codec_wrong_key' /
                 f'{operation}_{codec}_{rate}' / f'{fn}_dec.wav',
        'metadata': assets / 'diagnostics' / operation / f'{fn}.json',
    }


def case_fingerprint(idx, source_digest, operation, key, wrong_key, codecs, want_wrong, gain):
    settings = {
        'engine': ENGINE_VERSION, 'source_sha256': source_digest, 'index': idx,
        'operation': operation, 'key_sha256': hashlib.sha256(key).hexdigest(),
        'wrong_key_sha256': hashlib.sha256(wrong_key).hexdigest() if want_wrong else None,
        'codecs': codecs, 'want_wrong': want_wrong, 'input_gain': gain,
        'frame': FRAME, 'voice_rate': VOICE_RATE, 'wire_peak': WIRE_PEAK,
        'quant_target': QUANT_TARGET, 'sign_eps': SIGN_EPS, 'phase_eps': PHASE_EPS, 'lwt_levels': LWT_LEVELS,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def process_case(idx, input_wav, key, wrong_key, operation, codecs, ffmpeg_bin,
                 output_root, want_wrong=WRITE_WRONG_KEY, overwrite=OVERWRITE,
                 max_lag_ms=MAX_LAG_MS, gain=INPUT_GAIN, source_audio=None,
                 source_digest=None):
    paths = output_paths(output_root, TRANSFORM, operation, idx)
    want_wrong = bool(want_wrong and operation != 'invert')
    source_digest = source_digest or hashlib.sha256(Path(input_wav).read_bytes()).hexdigest()
    fingerprint = case_fingerprint(idx, source_digest, operation, key, wrong_key,
                                   codecs, want_wrong, gain)
    # Задержка тоже влияет на результат, поэтому участвует в проверке актуальности.
    fingerprint = hashlib.sha256(f'{fingerprint}|{max_lag_ms}'.encode()).hexdigest()
    required = [paths['encrypted'], paths['preview'], paths['decrypted'], paths['metadata']]
    if want_wrong:
        required.append(paths['decrypted_wrong_key'])
    for codec, bitrate in codecs:
        required.append(paths['codec'](codec, bitrate))
        if want_wrong:
            required.append(paths['codec_wrong'](codec, bitrate))
    if not overwrite and all(p.is_file() and p.stat().st_size > 44 for p in required):
        try:
            old = json.loads(paths['metadata'].read_text(encoding='utf-8'))
            if old.get('fingerprint') == fingerprint and old.get('status') == 'complete':
                return 'skip', old
        except (ValueError, OSError):
            pass

    audio = (load_mono_16k(input_wav) if source_audio is None else np.asarray(source_audio)) * gain
    encrypted, diagnostics = encrypt_audio(audio, operation, key, idx)
    # Завершённый JSON относится только к полностью записанному набору WAV.
    paths['metadata'].unlink(missing_ok=True)
    save_wav(paths['encrypted'], encrypted, floating=True)
    save_wav(paths['preview'], encrypted)
    # Проверяется фактический путь через сохранённый WAV, не обход в памяти.
    rate, encrypted = wavfile.read(paths['encrypted'])
    if rate != VOICE_RATE or encrypted.dtype != np.float64:
        raise ValueError('Шифротекст должен оставаться Float64')
    recovered = decrypt_audio(encrypted, diagnostics, key)
    save_wav(paths['decrypted'], recovered)
    diagnostics.update({
        'status': 'complete', 'fingerprint': fingerprint, 'input_gain': gain,
        'source_sha256': source_digest, 'frames': len(encrypted) // FRAME,
        'cipher_peak': float(np.max(np.abs(encrypted))),
        'cipher_rms': float(np.sqrt(np.mean(encrypted**2))),
        'cipher_clipped_samples': int(np.count_nonzero(np.abs(encrypted) > 1.0)),
        'preview_nonzero_samples': int(np.count_nonzero(pcm16(encrypted))),
        'no_codec': quality_metrics(audio, recovered),
        'saved_decrypted_pcm16': quality_metrics(audio, load_mono_16k(paths['decrypted'])),
        'codecs': {},
    })
    if want_wrong:
        wrong = decrypt_audio(encrypted, diagnostics, wrong_key)
        save_wav(paths['decrypted_wrong_key'], wrong)
        diagnostics['wrong_key'] = quality_metrics(audio, wrong)
    for codec, bitrate in codecs:
        decoded = compress_and_decode(encrypted, codec, bitrate, ffmpeg_bin)
        aligned, lag, confidence, state = align_wire(decoded, encrypted, max_lag_ms)
        recovered, counters = decrypt_audio(aligned, diagnostics, key,
                                            allow_corrupt=True, return_counters=True)
        save_wav(paths['codec'](codec, bitrate), recovered)
        codec_info = {
            'lag_samples': lag, 'correlation': confidence, 'alignment': state,
            'decoder_samples': len(decoded), 'aligned_samples': len(aligned),
            **quality_metrics(audio, recovered), **counters,
        }
        if want_wrong:
            wrong = decrypt_audio(aligned, diagnostics, wrong_key, allow_corrupt=True)
            save_wav(paths['codec_wrong'](codec, bitrate), wrong)
            codec_info['wrong_key'] = quality_metrics(audio, wrong)
        diagnostics['codecs'][f'{codec}_{bitrate}'] = codec_info
    paths['metadata'].parent.mkdir(parents=True, exist_ok=True)
    # JSON — маркер полного результата. Старые результаты без него не пропускаются.
    temporary = paths['metadata'].with_suffix('.json.tmp')
    temporary.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(paths['metadata'])
    return 'ok', diagnostics


_worker_config = None


def _init_worker(config):
    global _worker_config
    _worker_config = config


def _process_one_index(idx):
    config = _worker_config
    source = Path(config['dataset']) / f'test_sound{idx}.wav'
    completed = skipped = 0
    failures = []
    if not source.is_file():
        return idx, 0, 0, [('', f'Input WAV does not exist: {source}')]
    try:
        source_audio = load_mono_16k(source)
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except Exception as error:
        return idx, 0, 0, [('', f'{type(error).__name__}: {error}')]
    for operation in config['operations']:
        try:
            status, _ = process_case(
                idx, source, config['key'], config['wrong_key'], operation,
                config['codecs'], config['ffmpeg'], config['out'],
                config['want_wrong'], config['overwrite'], config['max_lag_ms'],
                config['gain'], source_audio, source_digest)
            completed += status == 'ok'
            skipped += status == 'skip'
        except Exception as error:
            failures.append((operation, f'{type(error).__name__}: {error}'))
    return idx, completed, skipped, failures


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET_DIR)
    parser.add_argument('--key', type=Path, help='Путь к ключу (по умолчанию transforms/mdct/key.bin)')
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    parser.add_argument('--ffmpeg')
    parser.add_argument('--n-files', type=int, default=N_FILES)
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--operations', type=split_comma, default=OPERATIONS,
                        help='Список через запятую. LWT: ' + ', '.join(OPERATIONS)
                             + '. На комплексных парах: ' + ', '.join(PAIRED_OPERATIONS))
    parser.add_argument('--no-codecs', action='store_true')
    parser.add_argument('--no-wrong-key', action='store_true', default=not WRITE_WRONG_KEY)
    parser.add_argument('--overwrite', action='store_true', default=OVERWRITE)
    return parser


def main(argv=None):
    from concurrent.futures import ProcessPoolExecutor

    args = build_parser().parse_args(argv)
    if not args.dataset.is_dir():
        raise FileNotFoundError(f'Не найден датасет: {args.dataset}')
    key_path = args.key if args.key is not None else KEY_PATH
    if not key_path.is_file():
        raise FileNotFoundError(f'Не найден ключ: {key_path}. Исправь KEY_PATH или укажи --key.')
    key = key_path.read_bytes()
    if len(key) < 16:
        raise ValueError('Key file must contain >= 16 bytes')
    wrong_key = WRONG_KEY
    if wrong_key == key:
        raise ValueError('Неправильный ключ совпадает с правильным')
    if not np.isfinite(INPUT_GAIN) or INPUT_GAIN <= 0:
        raise ValueError('INPUT_GAIN must be positive finite')
    if not 0 < WIRE_PEAK < 1 or not 0 < QUANT_TARGET <= 32767:
        raise ValueError('Некорректный WIRE_PEAK или QUANT_TARGET')
    if not args.operations or set(args.operations) - set(SUPPORTED_OPERATIONS):
        raise ValueError(f'Допустимые операции: {SUPPORTED_OPERATIONS}')
    if len(set(args.operations)) != len(args.operations):
        raise ValueError('Операции не должны повторяться')
    if args.workers < 1 or args.n_files < 1:
        raise ValueError('workers и n-files должны быть >= 1')
    codecs = () if args.no_codecs else CODECS
    ffmpeg = choose_ffmpeg(args.ffmpeg) if codecs else None
    if codecs and not ffmpeg:
        raise FileNotFoundError('Не найден FFmpeg. Укажи --ffmpeg или FFMPEG_BIN.')
    args.out.mkdir(parents=True, exist_ok=True)
    config = {
        'dataset': args.dataset, 'key': key, 'wrong_key': wrong_key,
        'operations': args.operations, 'codecs': codecs, 'ffmpeg': ffmpeg,
        'out': args.out, 'want_wrong': not args.no_wrong_key,
        'overwrite': args.overwrite, 'max_lag_ms': MAX_LAG_MS, 'gain': INPUT_GAIN,
    }
    print(f'НАЧАЛО: LWT CDF 5/3, блок/шаг={FRAME}/{FRAME}, без перекрытия; WAV={args.n_files}', flush=True)
    print('Операции:', ', '.join(args.operations), flush=True)
    print('Кодеки:', ', '.join(f'{c}_{br}' for c, br in codecs) or 'выключены', flush=True)
    print('Датасет:', args.dataset, flush=True)
    print('Ключ:', key_path, '| Выход:', args.out / 'assets', flush=True)
    total_ok = total_skipped = 0
    errors = []
    # Явная передача config работает и под Windows spawn, и при параметрах CLI.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(config,), mp_context=multiprocessing.get_context('spawn')) as executor:
        for idx, ok, skipped, failures in executor.map(
                _process_one_index, range(1, args.n_files+1), chunksize=1):
            total_ok += ok
            total_skipped += skipped
            for operation, message in failures:
                errors.append({'file': idx, 'operation': operation, 'error': message})
            if idx % 25 == 0 or idx == args.n_files or failures:
                print(f'[{idx}/{args.n_files}] готово={total_ok}, пропущено={total_skipped}, '
                      f'ошибок={len(errors)}', flush=True)
    # Переписываем журнал и при успехе, чтобы не оставался старый список ошибок.
    error_file = args.out / 'generation_errors.csv'
    with error_file.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('file', 'operation', 'error'), delimiter=';')
        writer.writeheader()
        writer.writerows(errors)
    print(f'ГОТОВО: выполнено={total_ok}, пропущено={total_skipped}, ошибок={len(errors)}', flush=True)
    if errors:
        print('ОШИБКИ:', error_file, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()