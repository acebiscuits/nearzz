import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from pesq import pesq

INDICES = range(1, 21)

ORIGINAL_DATASET = "../dataset/test_sounds2"

DATASETS = {
   "cos_simple_64K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/simple_mp3_64k/test_sound{i}_simple_dec.wav",
    "cos_simple_128K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/simple_mp3_128k/test_sound{i}_simple_dec.wav",
    "cos_simple_320K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/simple_mp3_320k/test_sound{i}_simple_dec.wav",
    "cos_simple_32K_opus": "../transforms/cos_inv/assets/decrypted_after_codec/simple_opus_32k/test_sound{i}_simple_dec.wav",
    "cos_simple_96K_opus": "../transforms/cos_inv/assets/decrypted_after_codec/simple_opus_96k/test_sound{i}_simple_dec.wav",

    "cos_simple_sign_64K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/simple_sign_mp3_64k/test_sound{i}_simple_sign_dec.wav",
    "cos_simple_sign_128K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/simple_sign_mp3_128k/test_sound{i}_simple_sign_dec.wav",
    "cos_simple_sign_320K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/simple_sign_mp3_320k/test_sound{i}_simple_sign_dec.wav",
    "cos_simple_sign_32K_opus": "../transforms/cos_inv/assets/decrypted_after_codec/simple_sign_opus_32k/test_sound{i}_simple_sign_dec.wav",
    "cos_simple_sign_96K_opus": "../transforms/cos_inv/assets/decrypted_after_codec/simple_sign_opus_96k/test_sound{i}_simple_sign_dec.wav",

    "cos_split_64K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/split_mp3_64k/test_sound{i}_split_dec.wav",
    "cos_split_128K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/split_mp3_128k/test_sound{i}_split_dec.wav",
    "cos_split_320K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/split_mp3_320k/test_sound{i}_split_dec.wav",
    "cos_split_32K_opus": "../transforms/cos_inv/assets/decrypted_after_codec/split_opus_32k/test_sound{i}_split_dec.wav",
    "cos_split_96K_opus": "../transforms/cos_inv/assets/decrypted_after_codec/split_opus_96k/test_sound{i}_split_dec.wav",

    "cos_split_sign_64K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/split_sign_mp3_64k/test_sound{i}_split_sign_dec.wav",
    "cos_split_sign_128K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/split_sign_mp3_128k/test_sound{i}_split_sign_dec.wav",
    "cos_split_sign_320K_mp3": "../transforms/cos_inv/assets/decrypted_after_codec/split_sign_mp3_320k/test_sound{i}_split_sign_dec.wav",
    "cos_split_sign_32K_opus": "../transforms/cos_inv/assets/decrypted_after_codec/split_sign_opus_32k/test_sound{i}_split_sign_dec.wav",
    "cos_split_sign_96K_opus": "../transforms/cos_inv/assets/decrypted_after_codec/split_sign_opus_96k/test_sound{i}_split_sign_dec.wav",

    "mdct_invert_64K_mp3": "../transforms/mdct/assets/decrypted_after_codec/invert_mp3_64k/test_sound{i}_invert_dec.wav",
    "mdct_invert_128K_mp3": "../transforms/mdct/assets/decrypted_after_codec/invert_mp3_128k/test_sound{i}_invert_dec.wav",
    "mdct_invert_320K_mp3": "../transforms/mdct/assets/decrypted_after_codec/invert_mp3_320k/test_sound{i}_invert_dec.wav",
    "mdct_invert_32K_opus": "../transforms/mdct/assets/decrypted_after_codec/invert_opus_32k/test_sound{i}_invert_dec.wav",
    "mdct_invert_96K_opus": "../transforms/mdct/assets/decrypted_after_codec/invert_opus_96k/test_sound{i}_invert_dec.wav",

    "mdct_shuffle_64K_mp3": "../transforms/mdct/assets/decrypted_after_codec/shuffle_mp3_64k/test_sound{i}_shuffle_dec.wav",
    "mdct_shuffle_128K_mp3": "../transforms/mdct/assets/decrypted_after_codec/shuffle_mp3_128k/test_sound{i}_shuffle_dec.wav",
    "mdct_shuffle_320K_mp3": "../transforms/mdct/assets/decrypted_after_codec/shuffle_mp3_320k/test_sound{i}_shuffle_dec.wav",
    "mdct_shuffle_32K_opus": "../transforms/mdct/assets/decrypted_after_codec/shuffle_opus_32k/test_sound{i}_shuffle_dec.wav",
    "mdct_shuffle_96K_opus": "../transforms/mdct/assets/decrypted_after_codec/shuffle_opus_96k/test_sound{i}_shuffle_dec.wav",

    "mdct_signflip_64K_mp3": "../transforms/mdct/assets/decrypted_after_codec/signflip_mp3_64k/test_sound{i}_signflip_dec.wav",
    "mdct_signflip_128K_mp3": "../transforms/mdct/assets/decrypted_after_codec/signflip_mp3_128k/test_sound{i}_signflip_dec.wav",
    "mdct_signflip_320K_mp3": "../transforms/mdct/assets/decrypted_after_codec/signflip_mp3_320k/test_sound{i}_signflip_dec.wav",
    "mdct_signflip_32K_opus": "../transforms/mdct/assets/decrypted_after_codec/signflip_opus_32k/test_sound{i}_signflip_dec.wav",
    "mdct_signflip_96K_opus": "../transforms/mdct/assets/decrypted_after_codec/signflip_opus_96k/test_sound{i}_signflip_dec.wav",

    "stft_invert_64K_mp3": "../transforms/stft/assets/decrypted_after_codec/invert_mp3_64k/test_sound{i}_invert_dec.wav",
    "stft_invert_128K_mp3": "../transforms/stft/assets/decrypted_after_codec/invert_mp3_128k/test_sound{i}_invert_dec.wav",
    "stft_invert_320K_mp3": "../transforms/stft/assets/decrypted_after_codec/invert_mp3_320k/test_sound{i}_invert_dec.wav",
    "stft_invert_32K_opus": "../transforms/stft/assets/decrypted_after_codec/invert_opus_32k/test_sound{i}_invert_dec.wav",
    "stft_invert_96K_opus": "../transforms/stft/assets/decrypted_after_codec/invert_opus_96k/test_sound{i}_invert_dec.wav",

    "stft_shuffle_64K_mp3": "../transforms/stft/assets/decrypted_after_codec/shuffle_mp3_64k/test_sound{i}_shuffle_dec.wav",
    "stft_shuffle_128K_mp3": "../transforms/stft/assets/decrypted_after_codec/shuffle_mp3_128k/test_sound{i}_shuffle_dec.wav",
    "stft_shuffle_320K_mp3": "../transforms/stft/assets/decrypted_after_codec/shuffle_mp3_320k/test_sound{i}_shuffle_dec.wav",
    "stft_shuffle_32K_opus": "../transforms/stft/assets/decrypted_after_codec/shuffle_opus_32k/test_sound{i}_shuffle_dec.wav",
    "stft_shuffle_96K_opus": "../transforms/stft/assets/decrypted_after_codec/shuffle_opus_96k/test_sound{i}_shuffle_dec.wav",

    "stft_signflip_64K_mp3": "../transforms/stft/assets/decrypted_after_codec/signflip_mp3_64k/test_sound{i}_signflip_dec.wav",
    "stft_signflip_128K_mp3": "../transforms/stft/assets/decrypted_after_codec/signflip_mp3_128k/test_sound{i}_signflip_dec.wav",
    "stft_signflip_320K_mp3": "../transforms/stft/assets/decrypted_after_codec/signflip_mp3_320k/test_sound{i}_signflip_dec.wav",
    "stft_signflip_32K_opus": "../transforms/stft/assets/decrypted_after_codec/signflip_opus_32k/test_sound{i}_signflip_dec.wav",
    "stft_signflip_96K_opus": "../transforms/stft/assets/decrypted_after_codec/signflip_opus_96k/test_sound{i}_signflip_dec.wav",
}

FS_TARGET = 16000
PESQ_MODE = "nb"
DO_ALIGN = False
MAX_LAG_MS = 150.0

OUT_FILES_CSV = "./pesq_files.csv"
OUT_MEAN_CSV = "./pesq_mean.csv"


def to_mono_float(data):
    if data.dtype.kind in ("i", "u"):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)

    if data.ndim > 1:
        data = data[:, 0]

    return data


def load_wav(path):
    fs, data = wavfile.read(path)
    return fs, to_mono_float(data)


def resample_to(x, fs_from, fs_to):
    if fs_from == fs_to:
        return x
    g = np.gcd(fs_from, fs_to)
    return signal.resample_poly(x, fs_to // g, fs_from // g)


def align_by_xcorr(ref, test, fs, max_lag_ms=80.0):
    ref0 = ref - np.mean(ref)
    test0 = test - np.mean(test)

    max_lag = int(fs * max_lag_ms / 1000.0)
    c = signal.correlate(test0, ref0, mode="full")
    mid = len(c) // 2
    lo, hi = mid - max_lag, mid + max_lag + 1
    lag = np.argmax(c[lo:hi]) + lo - mid

    if lag > 0:
        test = test[lag:]
        ref = ref[:len(test)]
    elif lag < 0:
        ref = ref[-lag:]
        test = test[:len(ref)]

    n = min(len(ref), len(test))
    return ref[:n], test[:n], int(lag)


def pesq_pair(ref, test, fs_in, do_align):

    if do_align:
        ref, test, lag = align_by_xcorr(ref, test, FS_TARGET, MAX_LAG_MS)
    else:
        lag = 0

    score = float(pesq(FS_TARGET, ref, test, PESQ_MODE))
    return score, lag


def aggregate_pesq(orig_path, datasets, indices):
    rows = []

    for name, path in datasets.items():

        for i in indices:
            orig_file = os.path.join(orig_path, f"test_sound{i}.wav")
            proc_file = path.format(i=i)

            row = {
                "dataset": name,
                "file": i,
                "orig_path": orig_file,
                "proc_path": proc_file,
                "pesq": np.nan,
                "lag_samples_16k": np.nan,
                "status": "ok",
                "error": None,
            }

            try:
                if not os.path.isfile(orig_file):
                    raise FileNotFoundError(f"orig not found: {orig_file}")

                if not os.path.isfile(proc_file):
                    raise FileNotFoundError(f"proc not found: {proc_file}")

                fs_o, ref = load_wav(orig_file)
                fs_p, test = load_wav(proc_file)

                use_xcorr_align = "_dec" in name
                score, lag = pesq_pair(ref, test, fs_o, use_xcorr_align)
                row["pesq"] = score

                row["lag_samples_16k"] = lag

            except Exception as e:
                row["status"] = "fail"
                row["error"] = str(e)

            rows.append(row)

    df_files = pd.DataFrame(rows)

    df_mean = (
        df_files[df_files["status"] == "ok"]
        .groupby("dataset", as_index=False)
        .agg(
            count=("file", "count"),
            pesq_mean=("pesq", "mean"),
            pesq_std=("pesq", lambda x: float(np.std(x, ddof=0))),
            lag_samples_16k_mean=("lag_samples_16k", "mean"),
        )
        .sort_values("dataset")
        .reset_index(drop=True)
    )

    return df_files, df_mean


def main():
    df_files, df_mean = aggregate_pesq(
        ORIGINAL_DATASET,
        DATASETS,
        INDICES
    )

    os.makedirs(os.path.dirname(OUT_FILES_CSV) or ".", exist_ok=True)
    df_files.to_csv(OUT_FILES_CSV, index=False, encoding="utf-8", sep=";")
    df_mean.to_csv(OUT_MEAN_CSV, index=False, encoding="utf-8", sep=";")

    print(df_files.to_string(index=False))
    print()
    print(df_mean.to_string(index=False))
    print()
    print(f"Saved: {OUT_FILES_CSV}")
    print(f"Saved: {OUT_MEAN_CSV}")


if __name__ == "__main__":
    main()