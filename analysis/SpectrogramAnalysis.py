import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

FILE_1   = "analysis/wind-record.mp3"
FILE_2   = "analysis/wind.wav"
TITLE_1  = "Field-recorded Wind Spectrogram"
TITLE_2  = "Synthesized Wind Spectrogram"
OUTPUT_1 = "analysis/WindRecordPlot.png"
OUTPUT_2 = "analysis/WindSynthPlot.png"

# FILE_1   = "analysis/thunder-record.mp3"
# FILE_2   = "analysis/thunder.wav"
# TITLE_1  = "Field-recorded Thunder Spectrogram"
# TITLE_2  = "Synthesized Thunder Spectrogram"
# OUTPUT_1 = "analysis/ThunderRecordPlot.png"
# OUTPUT_2 = "analysis/ThunderSynthPlot.png"

# FILE_1   = "analysis/crickets-record.mp3"
# FILE_2   = "analysis/crickets.wav"
# TITLE_1  = "Field-recorded Crickets Spectrogram"
# TITLE_2  = "Synthesized Crickets Spectrogram"
# OUTPUT_1 = "analysis/CricketsRecordPlot.png"
# OUTPUT_2 = "analysis/CricketsSynthPlot.png"

SAMPLE_RATE = None
DB_MIN      = -80
CMAP        = "magma"
DPI         = 150
MAX_SECONDS = 20


def load_centre_clip(path, target_dur):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    total_dur = librosa.get_duration(y=y, sr=sr)
    if total_dur <= target_dur:
        return y, sr, total_dur
    centre = total_dur / 2
    start_s = centre - target_dur / 2
    start_sample = int(start_s * sr)
    end_sample = start_sample + int(target_dur * sr)
    return y[start_sample:end_sample], sr, target_dur


def make_spectrogram(y, sr, title, output_path, target_dur):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    times = librosa.times_like(S, sr=sr, hop_length=512)
    freqs = librosa.mel_frequencies(n_mels=128, fmin=20, fmax=sr / 2)

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    img = ax.pcolormesh(times, freqs, S_db, shading="auto", cmap=CMAP, vmin=DB_MIN, vmax=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_ylim(20, sr / 2)
    ax.set_xlim(0, target_dur)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    plt.colorbar(img, ax=ax, label="dB")
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def get_duration(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return librosa.get_duration(y=y, sr=sr)


dur1 = get_duration(FILE_1)
dur2 = get_duration(FILE_2)
target_dur = min(dur1, dur2, MAX_SECONDS)

y1, sr1, _ = load_centre_clip(FILE_1, target_dur)
y2, sr2, _ = load_centre_clip(FILE_2, target_dur)

make_spectrogram(y1, sr1, TITLE_1, OUTPUT_1, target_dur)
make_spectrogram(y2, sr2, TITLE_2, OUTPUT_2, target_dur)
