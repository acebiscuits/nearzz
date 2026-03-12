import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from pesq import pesq

INDICES = range(1, 24)

ORIGINAL_DATASET = "../dataset/test_sounds"

DATASETS = {
    "stft_invert_enc": "../transforms/stft/assets/encrypted/invert/test_sound{i}_invert_enc.wav",
    "stft_shuffle_enc": "../transforms/stft/assets/encrypted/shuffle/test_sound{i}_shuffle_enc.wav",

    "stft_invert_dec": "../transforms/stft/assets/decrypted/invert/test_sound{i}_invert_dec.wav",
    "stft_shuffle_dec": "../transforms/stft/assets/decrypted/shuffle/test_sound{i}_shuffle_dec.wav",
}

FS_TARGET = 16000
DO_ALIGN = True
MAX_LAG_MS = 80.0

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


def pesq_wb_pair(ref, test, fs_in):
    ref16 = resample_to(ref, fs_in, FS_TARGET)
    test16 = resample_to(test, fs_in, FS_TARGET)

    if DO_ALIGN:
        ref16, test16, lag = align_by_xcorr(ref16, test16, FS_TARGET, MAX_LAG_MS)
    else:
        lag = 0

    score = float(pesq(FS_TARGET, ref16, test16, "wb"))
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
                "pesq_wb": np.nan,
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

                if fs_p != fs_o:
                    test = resample_to(test, fs_p, fs_o)

                score, lag = pesq_wb_pair(ref, test, fs_o)

                row["pesq_wb"] = score
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
            pesq_wb_mean=("pesq_wb", "mean"),
            pesq_wb_std=("pesq_wb", lambda x: float(np.std(x, ddof=0))),
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
    df_files.to_csv(OUT_FILES_CSV, index=False, encoding="utf-8")
    df_mean.to_csv(OUT_MEAN_CSV, index=False, encoding="utf-8")

    print(df_files.to_string(index=False))
    print()
    print(df_mean.to_string(index=False))
    print()
    print(f"Saved: {OUT_FILES_CSV}")
    print(f"Saved: {OUT_MEAN_CSV}")


if __name__ == "__main__":
    main()