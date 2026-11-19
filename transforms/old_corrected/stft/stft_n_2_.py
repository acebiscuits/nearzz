import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import hann


FFMPEG_BIN_DIR = r"D:\5_year\nirs\ffmpeg\bin"
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = (BASE_DIR / "../../../dataset/test_sounds3").resolve()
KEY_PATH = BASE_DIR / "key.bin"
WRONG_KEY = b"wrong-key"
RESULTS_DIR = BASE_DIR / "new_experiments_hann_fixed"

FLOAT_CODEC_OPERATIONS = {
    "sbox-16-float-codec": "sbox-16",
    "rot-xor-16-float-codec": "rot-xor-16",
}
INT16_STFT_OPERATIONS = ("sbox-16", "rot-xor-16", *FLOAT_CODEC_OPERATIONS)
POLAR_OPERATIONS = ("phase-offset", "amplitude-permutation", "amplitude-phase")
NEW_OPERATIONS = (
    "orthomix",
    *POLAR_OPERATIONS,
    "amplitude-sign",
    "adaptive-orthomix",
    "rot-xor",
    *INT16_STFT_OPERATIONS,
)
MAX_WORKERS = 8
RECALCULATE_EXISTING = False
SAVE_CODEC_WRONG_KEY = False


CODEC_VARIANTS = [
    ("mp3", "64k"),
    ("mp3", "128k"),
    ("mp3", "320k"),
    ("opus", "16k"),
    ("opus", "48k"),
    ("opus", "96k"),
]


def to_int16_safe(samples):
    samples = np.asarray(samples, dtype=np.float64)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(samples, -32768, 32767).astype(np.int16)


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
N_FILES = 15
WINDOW_VECTOR = window_nonzero(window_function, segment_length)
# Масштаб вещественных компонент rFFT для операций, рассчитанных на MDCT.
STFT_COMPONENT_SCALE = np.sqrt(2.0 / segment_length_padded)


def rng_from_key(key_bytes, nonce):
    h = hashlib.sha512()
    h.update(key_bytes)
    h.update(str(nonce).encode("utf-8"))
    digest = h.digest()
    seed_words = np.frombuffer(digest, dtype=np.uint32)
    seed_seq = np.random.SeedSequence(seed_words, pool_size=8)
    return np.random.default_rng(seed_seq)


COEFFICIENT_SCALE = 32768.0
# Только старый rot-xor: 8-битное квантование с потерями.
# adaptive-orthomix НЕ использует этот предел и НЕ квантует STFT-коэффициенты.
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
    """AES S-box на ДВУХ байтах знакового int16 STFT-коэффициента.

    quant_scale (целых значений на единицу STFT) известен из метаданных.
    Побайтовая подстановка обратима точно над int16, но после iSTFT+STFT
    квантованные числа могут измениться. Это эксперимент, не обещание
    точного восстановления звука после перекрывающихся блоков.
    """
    if not np.isfinite(quant_scale) or quant_scale <= 0.0:
        raise ValueError(f'Некорректный sbox-16 quant_scale: {quant_scale}')
    scaled = np.asarray(values, dtype=np.float64) * quant_scale
    if not np.all(np.isfinite(scaled)):
        raise ValueError('sbox-16: нечисловые STFT-коэффициенты')
    if decrypt:
        # После iSTFT -> STFT коэффициенты могут отличаться и выходить из
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
    """Rot-XOR над int16 STFT-коэффициентами: q -> XOR(key) -> ROTL16 -> XOR-chain.

    Вход/выход values в единицах STFT. Шаг квантования = 1 / quant_scale.
    Обратная цепочка: c[i] XOR c[i-1], ROR16, XOR(key). iv для c[-1].
    Клиппинг допустим только при дешифровании, где он означает потерю данных.
    """
    if not np.isfinite(quant_scale) or quant_scale <= 0:
        raise ValueError(f'rot-xor-16: invalid quant_scale={quant_scale}')
    scaled = np.asarray(values, dtype=np.float64) * quant_scale
    if not np.all(np.isfinite(scaled)):
        raise ValueError('rot-xor-16: non-finite STFT coefficients')
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


def _frame_rng(key, file_number, operation, frame):
    return rng_from_key(key, f"stft_test_sound{file_number}_{operation}_block_{frame}")


def _modify_coefficients(coeff, block_index, operation, key, decrypt, file_number, sbox16_scale=None):
    """Встроенные операции: orthomix / amplitude-sign / adaptive-orthomix / sbox-16 / rot-xor / rot-xor-16.

    Генерация параметров детерминирована ключом, номером файла,
    названием шифра и индексом STFT-блока (как nonce в старом скрипте).
    """
    values = np.asarray(coeff, dtype=np.float64) / COEFFICIENT_SCALE
    rng = _frame_rng(key, file_number, operation, block_index)
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
        # Здесь нет восьмибитного квантования коэффициентов.
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
        raise ValueError(f'Неизвестная операция STFT: {operation}')

    return np.asarray(output, dtype=np.float64) * COEFFICIENT_SCALE


def _int16_scale_for_spectrum(spectrum):
    bins = spectrum[1:-1]
    peak = max(float(np.max(np.abs(bins.real))), float(np.max(np.abs(bins.imag))))
    return 30000.0 / max(peak * STFT_COMPONENT_SCALE, 1.0)


def _modify_polar_coefficients(bins, rng, operation, decrypt):
    if operation == "phase-offset":
        shifts = rng.uniform(-np.pi, np.pi, size=bins.size)
        return bins * np.exp(1j * (-shifts if decrypt else shifts))

    permutation = rng.permutation(bins.size)
    magnitudes = np.abs(bins)
    phases = np.angle(bins)
    # Малая добавка в единицах rFFT сохраняет фазу на позиции,
    # даже когда перестановка подставила туда нулевой модуль.
    eps = 1e-8 * COEFFICIENT_SCALE / STFT_COMPONENT_SCALE
    if operation == "amplitude-phase":
        shifts = rng.uniform(-np.pi, np.pi, size=bins.size)
        phases = phases - shifts if decrypt else phases + shifts
    if decrypt:
        inverse_permutation = np.argsort(permutation)
        magnitudes = np.maximum(magnitudes[inverse_permutation] - eps, 0.0)
    else:
        magnitudes = magnitudes[permutation] + eps
    return magnitudes * np.exp(1j * phases)


def _modify_stft(spectrum, operation, key, decrypt, file_number, quant_scale):
    operation = FLOAT_CODEC_OPERATIONS.get(operation, operation)
    result = spectrum.copy()
    # DC и Найквист остаются вещественными и сохраняются без изменений.
    bins = spectrum[1:-1]
    for frame in range(bins.shape[1]):
        # Первый кадр начинается на один шаг левее сигнала.
        # Кадр, начинающийся с нулевого отсчёта, сохраняет прежний индекс 0.
        frame_index = frame - 1
        if operation in POLAR_OPERATIONS:
            rng = _frame_rng(key, file_number, operation, frame_index)
            result[1:-1, frame] = _modify_polar_coefficients(bins[:, frame], rng, operation, decrypt)
            continue

        values = np.empty(2 * bins.shape[0], dtype=np.float64)
        values[0::2] = bins[:, frame].real * STFT_COMPONENT_SCALE
        values[1::2] = bins[:, frame].imag * STFT_COMPONENT_SCALE
        modified = _modify_coefficients(values, frame_index, operation, key, decrypt, file_number, quant_scale)
        result[1:-1, frame] = (modified[0::2] + 1j * modified[1::2]) / STFT_COMPONENT_SCALE
    return result


def _analyse(samples):
    # Центрируем первый кадр на начале сигнала. Оригинал целиком лежит
    # там, где два Hann-окна перекрываются, включая короткие файлы.
    right_padding = shift_length + (-len(samples)) % shift_length
    padded = np.pad(samples, (shift_length, right_padding))
    frames = np.lib.stride_tricks.sliding_window_view(padded, segment_length)[::shift_length]
    starts = np.arange(len(frames)) * shift_length
    spectrum = np.fft.rfft(frames * WINDOW_VECTOR, n=segment_length_padded, axis=1)
    return spectrum.T, starts, starts + segment_length


def _synthesize(spectrum, starts, stops, length):
    frames = np.fft.irfft(spectrum, n=segment_length_padded, axis=0)[:segment_length]
    wave = np.zeros(stops[-1], dtype=np.float64)
    weight = np.zeros_like(wave)
    for frame, (start, stop) in enumerate(zip(starts, stops)):
        wave[start:stop] += frames[:, frame]
        weight[start:stop] += WINDOW_VECTOR
    # Делим только на исходном участке. Края padding с малым весом
    # не попадают ни в WAV, ни в расчёт gain.
    original = slice(shift_length, shift_length + length)
    return wave[original] / weight[original]


def _codec_roundtrip(samples, fs, codec, bitrate, gain, float_channel):
    encoder, extension = {
        "mp3": ("libmp3lame", ".mp3"),
        "opus": ("libopus", ".ogg"),
    }[codec]
    configured = Path(FFMPEG_BIN_DIR) / "ffmpeg.exe"
    ffmpeg = str(configured) if configured.is_file() else shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg не найден в FFMPEG_BIN_DIR или PATH")

    if float_channel:
        wire_format, dtype, target_fs = "f32le", "<f4", fs
        encoder_format = "fltp" if codec == "mp3" else "flt"
        pcm_scale = gain / 32768.0
        pcm = np.asarray(samples * pcm_scale, dtype=dtype)
    else:
        wire_format, dtype, target_fs = "s16le", "<i2", 16000
        encoder_format = "s16p" if codec == "mp3" else "s16"
        pcm_scale = gain
        pcm = to_int16_safe(samples * pcm_scale).astype(dtype, copy=False)

    expected_length = (len(samples) * target_fs + fs // 2) // fs
    # У очень короткого MP3 может не хватить кадров для чтения FFmpeg.
    # Дополняем вход тишиной и после декодирования возвращаем исходную длительность.
    minimum_input_length = (4096 * fs + target_fs - 1) // target_fs
    if pcm.size < minimum_input_length:
        pcm = np.pad(pcm, (0, minimum_input_length - pcm.size))

    common = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    with tempfile.TemporaryDirectory(prefix="stft_codec_") as directory:
        compressed = str(Path(directory) / ("audio" + extension))
        # Для MP3 нужен seekable-файл, чтобы записать данные о задержке и padding.
        # В Float-режиме FFmpeg при необходимости подберёт частоту кодека сам.
        # На выходе декодера восстанавливаем исходную частоту для STFT.
        encode_rate = [] if float_channel else ["-ar", str(target_fs)]
        encode = common + [
            "-f", wire_format, "-ar", str(fs), "-ac", "1", "-i", "pipe:0",
            *encode_rate, "-c:a", encoder, "-sample_fmt", encoder_format,
            "-b:a", bitrate, compressed,
        ]
        decode = common + [
            "-i", compressed, "-ar", str(target_fs), "-ac", "1",
            "-c:a", "pcm_" + wire_format, "-f", wire_format, "pipe:1",
        ]
        for stage, command, payload in (
            ("encode", encode, pcm.tobytes()),
            ("decode", decode, b""),
        ):
            result = subprocess.run(command, input=payload, capture_output=True)
            if result.returncode:
                error = result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"{codec} {bitrate} {stage}: {error}")

    decoded = np.frombuffer(result.stdout, dtype=dtype).astype(np.float64)
    if not decoded.size:
        raise RuntimeError(f"{codec} {bitrate}: декодер вернул пустой сигнал")
    if decoded.size < expected_length:
        decoded = np.pad(decoded, (0, expected_length - decoded.size))
    return target_fs, decoded[:expected_length] / pcm_scale


def _new_paths(operation, file_number):
    assets = RESULTS_DIR / "assets"
    name = f"test_sound{file_number}_{operation}"
    return {
        "encrypted": assets / "encrypted" / operation / f"{name}_enc.wav",
        "preview": assets / "encrypted" / operation / f"{name}_enc_preview_pcm16.wav",
        "meta": assets / "encrypted" / operation / f"{name}_meta.json",
        "decrypted": assets / "decrypted" / operation / f"{name}_dec.wav",
        "wrong": assets / "decrypted_wrong_key" / operation / f"{name}_dec.wav",
        "codec": lambda codec, bitrate: (
            assets / "decrypted_after_codec" / f"{operation}_{codec}_{bitrate}" / f"{name}_dec.wav"
        ),
        "codec_wrong": lambda codec, bitrate: (
            assets / "decrypted_after_codec_wrong_key" / f"{operation}_{codec}_{bitrate}" / f"{name}_dec.wav"
        ),
    }


def _needs_write(path):
    return RECALCULATE_EXISTING or not path.is_file()


def _write_audio(path, fs, samples, float64=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(samples, dtype=np.float64) if float64 else to_int16_safe(samples)
    wavfile.write(str(path), fs, audio)


def _case_finished(paths, int16_coefficients):
    expected = [paths["encrypted"], paths["decrypted"], paths["wrong"]]
    if int16_coefficients:
        expected.extend((paths["meta"], paths["preview"]))
    for codec, bitrate in CODEC_VARIANTS:
        expected.append(paths["codec"](codec, bitrate))
        if SAVE_CODEC_WRONG_KEY:
            expected.append(paths["codec_wrong"](codec, bitrate))
    return not any(_needs_write(path) for path in expected)


def _load_or_encrypt(paths, fs, samples, spectrum, starts, stops, operation, key, file_number):
    int16_coefficients = operation in INT16_STFT_OPERATIONS
    quant_scale, gain = None, 1.0
    write_ciphertext = _needs_write(paths["encrypted"])
    if int16_coefficients and not RECALCULATE_EXISTING:
        if paths["encrypted"].is_file() != paths["meta"].is_file():
            raise ValueError(f"{operation}: зашифрованный WAV и meta JSON должны сохраняться парой")

    if write_ciphertext:
        if int16_coefficients:
            quant_scale = _int16_scale_for_spectrum(spectrum)
        encrypted_spec = _modify_stft(spectrum, operation, key, False, file_number, quant_scale)
        encrypted = _synthesize(encrypted_spec, starts, stops, len(samples))
        if int16_coefficients:
            peak = float(np.max(np.abs(encrypted)))
            gain = 30000.0 / peak if peak else 1.0
            # Float WAV использует единичную амплитуду. Gain компенсируется
            # при чтении, а для PCM16-превью остаётся уровень до 30000.
            _write_audio(paths["encrypted"], fs, encrypted * (gain / 32768.0), float64=True)
            paths["meta"].write_text(json.dumps({
                "mode": operation,
                "quant_scale": quant_scale,
                "codec_gain": gain,
                "ciphertext_peak": peak,
                "encrypted_format": "normalized IEEE float64 WAV",
            }, indent=2), encoding="utf-8")
        else:
            _write_audio(paths["encrypted"], fs, encrypted)

    saved_fs, ciphertext = wavfile.read(str(paths["encrypted"]))
    expected_dtype = np.float64 if int16_coefficients else np.int16
    if saved_fs != fs or ciphertext.shape != samples.shape or ciphertext.dtype != expected_dtype:
        raise ValueError(f"Некорректный формат зашифрованного WAV: {paths['encrypted']}")
    if int16_coefficients:
        if not write_ciphertext:
            metadata = json.loads(paths["meta"].read_text(encoding="utf-8"))
            quant_scale = float(metadata["quant_scale"])
            gain = float(metadata["codec_gain"])
            if not (np.isfinite(quant_scale) and quant_scale > 0 and np.isfinite(gain) and gain > 0):
                raise ValueError(f"Некорректный масштаб: {paths['meta']}")
        if _needs_write(paths["preview"]):
            _write_audio(paths["preview"], fs, np.rint(ciphertext * 32768.0))
        ciphertext = ciphertext * (32768.0 / gain)
    return ciphertext.astype(np.float64, copy=False), quant_scale, gain


def _save_recovered(paths, fs, samples, operation, key, wrong_key, file_number, quant_scale):
    spectrum, starts, stops = _analyse(samples)
    for path, decode_key in ((paths[0], key), (paths[1], wrong_key)):
        if path is not None and _needs_write(path):
            decoded = _modify_stft(spectrum, operation, decode_key, True, file_number, quant_scale)
            _write_audio(path, fs, _synthesize(decoded, starts, stops, len(samples)))


def _process_new_one(index, key, wrong_key):
    source = DATASET_DIR / f"test_sound{index}.wav"
    done = skipped = 0
    errors = []
    if not source.is_file():
        return done, skipped, [("", f"missing input: {source}")]
    fs, samples = wavfile.read(str(source))
    if samples.ndim > 1:
        samples = samples[:, 0]
    samples = np.asarray(samples, dtype=np.float64)
    if not samples.size:
        return done, skipped, [("", f"empty input: {source}")]

    spectrum, starts, stops = _analyse(samples)
    for operation in NEW_OPERATIONS:
        try:
            paths = _new_paths(operation, index)
            if _case_finished(paths, operation in INT16_STFT_OPERATIONS):
                skipped += 1
                continue
            ciphertext, quant_scale, gain = _load_or_encrypt(
                paths, fs, samples, spectrum, starts, stops, operation, key, index,
            )
            _save_recovered(
                (paths["decrypted"], paths["wrong"]), fs, ciphertext, operation,
                key, wrong_key, index, quant_scale,
            )
            for codec, bitrate in CODEC_VARIANTS:
                correct_path = paths["codec"](codec, bitrate)
                wrong_path = paths["codec_wrong"](codec, bitrate) if SAVE_CODEC_WRONG_KEY else None
                if not _needs_write(correct_path) and (wrong_path is None or not _needs_write(wrong_path)):
                    continue
                fs_dec, wire = _codec_roundtrip(
                    ciphertext, fs, codec, bitrate, gain, operation in FLOAT_CODEC_OPERATIONS,
                )
                _save_recovered(
                    (correct_path, wrong_path), fs_dec, wire, operation,
                    key, wrong_key, index, quant_scale,
                )
            done += 1
        except Exception as error:
            errors.append((operation, f"{type(error).__name__}: {error}"))
    return done, skipped, errors


def _prepare_new_output_dirs():
    for operation in NEW_OPERATIONS:
        paths = _new_paths(operation, 1)
        for name in ("encrypted", "decrypted", "wrong"):
            paths[name].parent.mkdir(parents=True, exist_ok=True)
        for codec, bitrate in CODEC_VARIANTS:
            paths["codec"](codec, bitrate).parent.mkdir(parents=True, exist_ok=True)
            if SAVE_CODEC_WRONG_KEY:
                paths["codec_wrong"](codec, bitrate).parent.mkdir(parents=True, exist_ok=True)


def _run_new_measurements():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Датасет отсутствует: {DATASET_DIR}")
    if not KEY_PATH.is_file():
        raise FileNotFoundError(f"Ключ отсутствует: {KEY_PATH}")
    key = KEY_PATH.read_bytes()
    wrong_key = WRONG_KEY
    if key == wrong_key:
        raise ValueError("key.bin and built-in wrong key must differ")
    _prepare_new_output_dirs()

    print(f"START STFT: {N_FILES} WAV, {MAX_WORKERS} процессов, "
          f"{len(NEW_OPERATIONS)} режимов, {len(CODEC_VARIANTS)} вариантов кодека", flush=True)
    print(f"DATASET: {DATASET_DIR}", flush=True)
    print(f"OUTPUT: {RESULTS_DIR / 'assets'}", flush=True)
    failures = []
    complete = skipped = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_process_new_one, index, key, wrong_key): index
            for index in range(1, N_FILES + 1)
        }
        for processed, future in enumerate(as_completed(futures), 1):
            number = futures[future]
            try:
                done, passed, errors = future.result()
                complete += done
                skipped += passed
                failures.extend((number, operation, error) for operation, error in errors)
            except Exception as error:
                failures.append((number, "", f"{type(error).__name__}: {error}"))
            if processed % 25 == 0:
                print(f"processed={processed}, done={complete}, skipped={skipped}, errors={len(failures)}", flush=True)

    if failures:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        error_file = RESULTS_DIR / "new_stft_errors.csv"
        with error_file.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("file", "operation", "error"))
            writer.writerows(failures)
        print(f"ERROR LIST: {error_file}", flush=True)
    print(f"FINISH STFT: done={complete}; skipped={skipped}; failed={len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_new_measurements()
