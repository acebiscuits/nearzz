#!/usr/bin/env python3

import subprocess
import threading
import signal
import numpy as np


RATE = 44100
CHANNELS = 1
BLOCK = 1024
GAIN = 3.0

MIC = "alsa_input.pci-0000_04_00.6.analog-stereo"
TX_SINK = "telemost_enc_sink"

RX_SOURCE = "telemost_rx_sink.monitor"
SPEAKERS = "alsa_output.pci-0000_04_00.6.analog-stereo"

BLOCK_BYTES = BLOCK * CHANNELS * 2

running = True


def stop_handler(sig, frame):
    global running
    running = False


signal.signal(signal.SIGINT, stop_handler)


def process(data):
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

    samples *= GAIN
    samples = np.clip(samples, -32768, 32767)

    return samples.astype(np.int16).tobytes()


def pipe(source, sink, name):
    reader = subprocess.Popen(
        [
            "parec",
            "--device=" + source,
            "--format=s16le",
            "--rate=" + str(RATE),
            "--channels=" + str(CHANNELS),
            "--latency-msec=20",
            "--process-time-msec=10",
            "--raw",
        ],
        stdout=subprocess.PIPE,
        bufsize=0,
    )

    writer = subprocess.Popen(
        [
            "pacat",
            "--playback",
            "--device=" + sink,
            "--format=s16le",
            "--rate=" + str(RATE),
            "--channels=" + str(CHANNELS),
            "--latency-msec=20",
            "--process-time-msec=10",
            "--raw",
        ],
        stdin=subprocess.PIPE,
        bufsize=0,
    )

    print(name, "started")

    try:
        while running:
            data = reader.stdout.read(BLOCK_BYTES)

            if not data:
                break

            writer.stdin.write(process(data))
            writer.stdin.flush()

    finally:
        reader.kill()
        writer.kill()
        print(name, "stopped")


tx = threading.Thread(
    target=pipe,
    args=(MIC, TX_SINK, "TX"),
    daemon=True,
)

rx = threading.Thread(
    target=pipe,
    args=(RX_SOURCE, SPEAKERS, "RX"),
    daemon=True,
)


tx.start()
rx.start()

print("Audio processor started")
print("Ctrl+C = stop")

try:
    while running:
        tx.join(0.5)
        rx.join(0.5)
except KeyboardInterrupt:
    running = False