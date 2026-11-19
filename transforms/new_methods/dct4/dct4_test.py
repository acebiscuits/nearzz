#!/usr/bin/env python3
"""DCT-IV, независимые блоки по 320 отсчётов, семь операций.

Самостоятельный файл: crypto/adapter.py и nirs_timepack_v2 не нужны.
Зависимости: NumPy, SciPy; FFmpeg нужен только для MP3/Opus.
Encrypted WAV: Float64 с обратимым общим масштабом в diagnostics/*.json.
PCM16 preview нужен только для прослушивания, не для точного дешифрования.
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
from scipy.fft import dct
from scipy.io import wavfile

TRANSFORM = 'dct4'
ENGINE_VERSION = 'dct4-seven-ops-f64-v1'
OPERATIONS = ('invert', 'shuffle', 'signflip', 'orthomix', 'rot-xor',
              'sbox', 'amplitude-phase')
SUPPORTED_OPERATIONS = OPERATIONS
BIT_OPERATIONS = frozenset(('sbox', 'rot-xor'))

# ========== НАСТРОЙКИ ЭКСПЕРИМЕНТА ==========
FRAME = 320                    # 20 мс при 16 кГц; шаг тоже 320, перекрытия нет
VOICE_RATE = 16000
CODECS = (("mp3", "64k"), ("mp3", "128k"), ("mp3", "320k"),
          ("opus", "16k"), ("opus", "48k"), ("opus", "96k"))
# Расположение скрипта: nirs/total/transforms/new_methods/dct4/dct4.py
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
N_FILES = 1722
MAX_WORKERS = 8
OVERWRITE = False
WRITE_WRONG_KEY = True
INPUT_GAIN = 1.0
MAX_LAG_MS = 250.0
WIRE_PEAK = 0.95                # запас до полного уровня, БЕЗ обрезания отсчётов
QUANT_TARGET = 30000.0          # int16, один масштаб на файл, запас до 32767
PHASE_EPS = 1e-4                # единицы ортонормированных DCT-коэффициентов
# В OPERATIONS включены только семь перечисленных выше способов.
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


def dct4(frame):
    """Ортонормированная DCT-IV. Эта же функция выполняет обратное преобразование."""
    return dct(np.asarray(frame, dtype=np.float64), type=4, norm='ortho')


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
    """Точные перестановки 16-битных слов. Без float, DCT и квантования."""
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


def modify_coefficients(values, operation, key, frame_index, *, decrypt=False,
                        file_number=0, quant_scale=None, phase_eps=PHASE_EPS,
                        allow_corrupt=False, counters=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError('Некорректные DCT-коэффициенты')
    if operation == 'invert':
        # Как в прежнем MDCT: разворот порядка, а НЕ смена знака всех значений.
        return values[::-1].copy()
    rng = frame_rng(key, file_number, operation, frame_index)
    size = len(values)
    if operation == 'shuffle':
        permutation = rng.permutation(size)
        return values[np.argsort(permutation) if decrypt else permutation]
    if operation == 'signflip':
        signs = 2.0 * rng.integers(0, 2, size=size) - 1.0
        return values * signs
    if operation == 'orthomix':
        indices = rng.permutation(size)
        pairs = size // 2
        first, second = indices[:2*pairs:2], indices[1:2*pairs:2]
        angles = rng.uniform(-np.pi, np.pi, size=pairs)
        if decrypt:
            angles = -angles
        c, s = np.cos(angles), np.sin(angles)
        a, b = values[first], values[second]
        result = values.copy()
        result[first] = c*a - s*b
        result[second] = s*a + c*b
        return result
    if operation == 'amplitude-phase':
        if size % 2:
            raise ValueError('amplitude-phase требует чётного размера блока')
        if not np.isfinite(phase_eps) or phase_eps <= 0:
            raise ValueError('phase_eps должен быть положительным конечным')
        # Это комплексные ПАРЫ вещественных DCT, а не собственные фазы DCT.
        z = values[0::2] + 1j * values[1::2]
        permutation = rng.permutation(len(z))
        angles = rng.uniform(-np.pi, np.pi, size=len(z))
        if decrypt:
            radius = np.maximum(np.abs(z)[np.argsort(permutation)] - phase_eps, 0.0)
            phase = np.angle(z) - angles
        else:
            radius = np.abs(z)[permutation] + phase_eps
            phase = np.angle(z) + angles
        result = np.empty_like(values)
        result[0::2] = radius * np.cos(phase)
        result[1::2] = radius * np.sin(phase)
        return result
    if operation in BIT_OPERATIONS:
        if quant_scale is None or not np.isfinite(quant_scale) or quant_scale <= 0:
            raise ValueError('Нужен положительный quant_scale из метаданных файла')
        scaled = values * quant_scale
        if not np.all(np.isfinite(scaled)):
            raise ValueError('Переполнение при квантовании DCT-коэффициентов')
        rounded = np.rint(scaled)
        outside = (rounded < -32768) | (rounded > 32767)
        count = int(np.count_nonzero(outside))
        if count:
            if not (decrypt and allow_corrupt):
                raise ValueError(f'{operation}: {count} коэффициентов вне int16')
            # Только повреждённый канал: насыщение для продолжения эксперимента.
            # В JSON видно число таких слов; это не восстановление информации.
            rounded = np.clip(rounded, -32768, 32767)
        if counters is not None:
            counters['out_of_range_words'] += count
        words = rounded.astype(np.int16).view(np.uint16)
        modified = modify_words(words, rng, operation, decrypt)
        return modified.view(np.int16).astype(np.float64) / quant_scale
    raise ValueError(f'Неизвестная операция: {operation}')


class DCT4Cipher:
    def __init__(self, operation, key, *, decrypt=False, file_number=0,
                 quant_scale=None, phase_eps=PHASE_EPS, frame_size=FRAME,
                 allow_corrupt=False):
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f'Неизвестная операция: {operation}')
        if frame_size <= 0 or frame_size % 2:
            raise ValueError('frame_size должен быть положительным чётным')
        self.operation, self.key, self.decrypt = operation, key, decrypt
        self.file_number, self.quant_scale = file_number, quant_scale
        self.phase_eps, self.frame_size = phase_eps, frame_size
        self.allow_corrupt = allow_corrupt
        self.counters = {'out_of_range_words': 0}

    def process(self, frame_index, samples):
        if np.shape(samples) != (self.frame_size,):
            raise ValueError(f'Ожидался блок из {self.frame_size} отсчётов')
        coeff = dct4(samples)
        changed = modify_coefficients(
            coeff, self.operation, self.key, frame_index, decrypt=self.decrypt,
            file_number=self.file_number, quant_scale=self.quant_scale,
            phase_eps=self.phase_eps, allow_corrupt=self.allow_corrupt,
            counters=self.counters)
        return dct4(changed)


def pad_frames(samples, frame_size=FRAME):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not samples.size or not np.all(np.isfinite(samples)):
        raise ValueError('Нужен непустой конечный одномерный сигнал')
    return np.pad(samples, (0, (-len(samples)) % frame_size))


def run_frames(cipher, samples):
    """Полные неперекрывающиеся блоки. Шифротекст НЕ обрезается по исходной длине."""
    if cipher.decrypt and len(samples) % cipher.frame_size:
        raise ValueError('Усечён зашифрованный последний блок')
    source = pad_frames(samples, cipher.frame_size)
    output = np.empty_like(source)
    for i, start in enumerate(range(0, len(source), cipher.frame_size)):
        out = cipher.process(i, source[start:start + cipher.frame_size])
        if out.shape != (cipher.frame_size,) or not np.all(np.isfinite(out)):
            raise ValueError(f'Некорректный выход блока #{i}')
        output[start:start + cipher.frame_size] = out
    return output


def quant_scale_for_file(samples):
    blocks = pad_frames(samples).reshape(-1, FRAME)
    peak = float(np.max(np.abs(dct(blocks, type=4, norm='ortho', axis=1))))
    return QUANT_TARGET / max(peak, 1.0)


def encrypt_audio(samples, operation, key, file_number=0):
    samples = np.asarray(samples, dtype=np.float64)
    quant_scale = quant_scale_for_file(samples) if operation in BIT_OPERATIONS else None
    cipher = DCT4Cipher(operation, key, file_number=file_number, quant_scale=quant_scale)
    raw = run_frames(cipher, samples)
    peak = float(np.max(np.abs(raw)))
    wire_gain = min(1.0, WIRE_PEAK / peak) if peak else 1.0
    metadata = {
        'engine': ENGINE_VERSION, 'transform': TRANSFORM, 'operation': operation,
        'index': int(file_number), 'frame': FRAME, 'hop': FRAME, 'overlap': 0,
        'sample_rate': VOICE_RATE, 'samples': len(samples), 'padded_samples': len(raw),
        'quant_bits': 16 if operation in BIT_OPERATIONS else None,
        'quant_scale': quant_scale, 'phase_eps': PHASE_EPS,
        'wire_gain': wire_gain, 'wire_format': 'float64', 'raw_cipher_peak': peak,
    }
    return raw * wire_gain, metadata


def decrypt_audio(wire, metadata, key, *, allow_corrupt=False, return_counters=False):
    """Дешифрование только из WAV, JSON и ключа, без исходного аудио."""
    if metadata.get('engine') != ENGINE_VERSION or metadata.get('transform') != TRANSFORM:
        raise ValueError('Несовместимая версия метаданных')
    wire = np.asarray(wire, dtype=np.float64)
    if wire.ndim != 1 or len(wire) != metadata['padded_samples']:
        raise ValueError('Длина WAV не соответствует полным блокам из JSON')
    gain = metadata['wire_gain']
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Некорректный wire_gain')
    if not 0 < metadata['samples'] <= len(wire):
        raise ValueError('Некорректная исходная длина')
    cipher = DCT4Cipher(
        metadata['operation'], key, decrypt=True, file_number=metadata['index'],
        quant_scale=metadata['quant_scale'], phase_eps=metadata['phase_eps'],
        frame_size=metadata['frame'], allow_corrupt=allow_corrupt)
    result = run_frames(cipher, wire / gain)[:metadata['samples']]
    return (result, cipher.counters) if return_counters else result


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
    with tempfile.TemporaryDirectory(prefix='nirs_dct4_codec_') as td:
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
        'quant_target': QUANT_TARGET, 'phase_eps': PHASE_EPS,
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
        'cipher_clipped_samples': 0,
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
    parser.add_argument('--operations', type=split_comma, default=OPERATIONS)
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
    print(f'НАЧАЛО: DCT-IV, блок/шаг={FRAME}/{FRAME}, без перекрытия; WAV={args.n_files}', flush=True)
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
