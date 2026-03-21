import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
from scipy.signal import resample
from scipy.fft import rfft, irfft, rfftfreq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import io
import sounddevice as sd
import soundfile as sf
import subprocess

# Parameters
sample_rate = 44100 # Hz
duration = 10.0 # seconds
current_sound = np.zeros(int(sample_rate * duration)) # Basic sound array with no components

def lowpass(sound, cutoff=200, order=5): # Trivial implementation so using butter
    b, a = butter(order, cutoff/(sample_rate/2), btype='low')
    return lfilter(b, a, sound)

def highpass(sound, cutoff=2000, order=5): # Trivial implementation so using butter
    b, a = butter(order, cutoff/(sample_rate/2), btype='high')
    return lfilter(b, a, sound)

def white_noise(d=duration): # Low initial amplitude because it will be layered in most sounds
    noise = 0.3 * np.random.randn(int(sample_rate * d))
    return noise

def sine(frequency=400.0, d=duration):
    t = np.linspace(0, d, int(sample_rate * d), endpoint=False)
    return 0.3 * np.sin(2 * np.pi * frequency * t)

def triangle(frequency=400.0, d=duration):
    t = np.linspace(0, d, int(sample_rate * d), endpoint=False)
    return 0.3 * (2 * np.abs(2 * (frequency * t - np.floor(frequency * t + 0.5))) - 1)

def sawtooth(frequency=400.0, d=duration):
    t = np.linspace(0, d, int(sample_rate * d), endpoint=False)
    return 0.3 * (2 * (frequency * t - np.floor(frequency * t)) - 1)

def sawtooth_vibrato(frequency=220.0, d=duration, vibrato_rate=25.0, vibrato_depth=15.0):
    t = np.linspace(0, d, int(sample_rate * d), endpoint=False)
    vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    f = frequency + vibrato
    phase = np.cumsum(f) / sample_rate
    return 0.3 * (2 * (phase - np.floor(phase)) - 1)

def harmonic_sine(frequency=400.0, d=duration, depth=5):
    t = np.linspace(0, d, int(sample_rate * d), endpoint=False)
    sound = np.zeros_like(t)
    for d in range(1, depth + 1):
        sound += (d / (depth + 1)) * np.sin(2 * np.pi * (frequency * d) * t)
    return sound

def flute_like_sine(frequency=400.0, depth=7, d=duration):
    t = np.linspace(0, d, int(sample_rate * d), endpoint=False)
    sound = np.zeros_like(t)
    for d in range(1, depth + 1, 2):
        phase = np.random.uniform(0, 2 * np.pi)
        amp = 1.0 / (d ** 1.4)
        sound += amp * np.sin(2 * np.pi * frequency * d * t + phase)
    return sound

def band_EQ(sound, affect_array=[1,1,1,0.6,0.3,0.1,0,0.1,0,0]): # Legacy, no longer used
    f_min = 20
    f_max = 20 * 10**3
    num_samples = int(sample_rate * duration)
    controls = np.array(affect_array)
    band_freqs = np.logspace(np.log10(f_min), np.log10(f_max), len(controls))
    X = rfft(sound)
    freqs = rfftfreq(num_samples, 1/sample_rate)
    interp_fn = interp1d(band_freqs, controls, kind='linear', bounds_error=False, fill_value=(controls[0], controls[-1]))
    scales = interp_fn(freqs)
    Y = X * scales
    return irfft(Y, n=len(sound))

def sin_LFO(sound, affect, freq):
    t = np.arange(len(sound)) / sample_rate
    lfo = 0.5 * (1 + affect * np.sin(2 * np.pi * freq * t))
    return sound * lfo

def LFO(sound, freq=1.0, waveform='sine', depth=0.5, d=duration):
    t = np.arange(len(sound)) / sample_rate
    if waveform == 'sine':
        lfo = np.sin(2 * np.pi * freq * t)
    elif waveform == 'triangle':
        lfo = 2 * np.abs(2 * (freq * t % 1) - 1) - 1
    elif waveform == 'square':
        lfo = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == 'saw':
        lfo = 2 * (freq * t % 1) - 1
    else:
        raise ValueError("Unsupported waveform")
    return (1 + lfo * depth) * sound

def LFO_random_smooth(sound, freq=1.0, depth=0.5, d=duration):
    t = np.arange(len(sound)) / sample_rate
    num_points = int(freq * d) + 2
    control_times = np.linspace(0, d, num_points)
    control_values = np.random.uniform(-1, 1, num_points)
    lfo = np.interp(t, control_times, control_values)
    return (1 + lfo * depth) * sound

def smooth_random_amplitude_modulation(sound,mod_freq=1.0,min_gain=0.5,max_gain=1.5):
    n = len(sound)
    noise = np.random.randn(n)
    kernel_size = int(sample_rate / mod_freq)
    kernel_size = max(1, kernel_size)
    kernel = np.ones(kernel_size) / kernel_size
    smooth_noise = np.convolve(noise, kernel, mode="same")
    smooth_noise -= smooth_noise.min()
    smooth_noise /= smooth_noise.max()
    gain = min_gain + smooth_noise * (max_gain - min_gain)
    return sound * gain

def reverb(sound, gain=0.5, depth=1): # Does not make a noticeable effect on sounds with broad frequency ranges.
    shift_size = int(sample_rate/30)
    r = np.zeros_like(sound)
    for d in range(1, depth+1):
        r += np.roll(sound, shift=shift_size, axis=0) * gain / d
    return r

def slow_noise(change_rate=0.1, seed=None, d=duration):
    rng = np.random.default_rng(seed)
    n = int(d * sample_rate)
    t = np.arange(n) / sample_rate
    x = t * change_rate
    i0 = np.floor(x).astype(int)
    i1 = i0 + 1
    values = rng.random(i1.max() + 1)
    f = x - i0
    f = f * f * (3 - 2 * f)
    return values[i0] * (1 - f) + values[i1] * f

def adsr_envelope(waveform, attack, decay, sustain_level, release): # Uses linear interpolation between phases. Could implement easings with a pre-made library to smooth attack for better modulation.
    total_samples = len(waveform)
    a = int(attack  * sample_rate)
    d = int(decay   * sample_rate)
    r = int(release * sample_rate)
    s = max(0, total_samples - a - d - r)
    attack_env  = np.linspace(0.0, 1.0, a, endpoint=False)
    decay_env   = np.linspace(1.0, sustain_level, d, endpoint=False)
    sustain_env = np.full(s, sustain_level)
    release_env = np.linspace(sustain_level, 0.0, r)
    envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])
    if len(envelope) < total_samples:
        envelope = np.pad(envelope, (0, total_samples - len(envelope)))
    else:
        envelope = envelope[:total_samples]
    return waveform * envelope

def loop_finite(enveloped_wave, d=duration, n=1): # Loops a finite amount of sounds, with n per second, for d seconds
    total_samples = int(d * sample_rate)
    output = np.zeros(total_samples)
    pos = 0
    while pos < total_samples:
        end = min(pos + len(enveloped_wave), total_samples)
        chunk = enveloped_wave[:end - pos]
        output[pos:end] += chunk
        pos += int(sample_rate/n)
    return output

def loop_finite_random(enveloped_wave, d=duration, n=1, amplitude_depth=0.0): # The same as the above function, but sounds are randomly distributed with a mean of n per second. The relative amplitude of each sound is
    total_samples = int(d * sample_rate)
    output = np.zeros(total_samples)
    wave = enveloped_wave[:total_samples]
    n = int(n * d)
    for _ in range(n):
        pos = np.random.randint(0, total_samples)
        amp = np.random.uniform(1 - amplitude_depth, 1)
        output = output + amp * offset(
            np.pad(wave, (0, total_samples - len(wave))),
            -pos / sample_rate
        )
    return output

def triangular_envelope(audio, d=duration):
    total_samples = int(d * sample_rate)
    if total_samples > len(audio):
        total_samples = len(audio)
    half = total_samples // 2
    env = np.linspace(0, 1, half)
    env = np.concatenate([env, env[::-1]])
    if len(env) < total_samples:
        env = np.append(env, 0)
    audio[:total_samples] *= env
    return audio

def pitch_glide(sound, scale=0.5):
    total_samples = len(sound)
    semitone_envelope = np.linspace(0.0, scale, total_samples)
    pitch_ratios = 2 ** (semitone_envelope / 12)
    read_positions = np.cumsum(pitch_ratios)
    read_positions = read_positions / read_positions[-1] * (total_samples - 1)
    original_positions = np.arange(total_samples)
    return np.interp(read_positions, original_positions, sound)

def offset(array, t):
    samples = int(t * sample_rate)
    return np.roll(array, samples)


### RENDER AND EXPORTING ###

def export_to_bytes(sound, format='wav'):
    # Render sound and return (bytes, mime_type) for HTTP streaming download
    # Does not work when running this file on its own, needs app.py to govern this function
    audio_data = render(sound)
    buf = io.BytesIO()

    if format == 'wav':
        write(buf, sample_rate, audio_data)
        buf.seek(0)
        return buf.read(), 'audio/wav'

    elif format == 'ogg':
        sf.write(buf, audio_data, sample_rate, format='OGG', subtype='VORBIS')
        buf.seek(0)
        return buf.read(), 'audio/ogg'

    elif format == 'opus':
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_data, sample_rate, format='WAV')
        wav_buf.seek(0)
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', 'pipe:0', '-c:a', 'libopus', '-f', 'opus', 'pipe:1'],
            input=wav_buf.read(), check=True, capture_output=True
        )
        return result.stdout, 'audio/ogg; codecs=opus'

    elif format == 'mp3':
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_data, sample_rate, format='WAV')
        wav_buf.seek(0)
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', 'pipe:0', '-c:a', 'libmp3lame', '-f', 'mp3', 'pipe:1'],
            input=wav_buf.read(), check=True, capture_output=True
        )
        return result.stdout, 'audio/mpeg'

    else:
        raise ValueError(f"Unsupported format: '{format}'")

def export(sound, filename, format='wav'):
    # Write audio file to disk (used when running module directly)
    audio_data = render(sound)
    if format == 'wav':
        write(filename + ".wav", sample_rate, audio_data)
    elif format == 'ogg':
        sf.write(filename + ".ogg", audio_data, sample_rate, format='OGG', subtype='VORBIS')
    elif format == 'opus':
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        subprocess.run(
            ['ffmpeg', '-y', '-i', 'pipe:0', '-c:a', 'libopus', filename + '.opus'],
            input=buffer.read(), check=True, capture_output=True
        )
    elif format == 'mp3':
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        subprocess.run(
            ['ffmpeg', '-y', '-i', 'pipe:0', '-c:a', 'libmp3lame', filename + '.mp3'],
            input=buffer.read(), check=True, capture_output=True
        )
    else:
        raise ValueError(f"Unsupported format: '{format}'")

def crossfade_loop(sound, fade_duration=duration/5.0):
    fade_samples = int(fade_duration * sample_rate)
    total_samples = int(duration*sample_rate)
    ramp_in = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_samples)))
    ramp_out = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_samples)))

    start = sound[0:fade_samples].copy()
    end = sound[(total_samples-fade_samples):total_samples].copy()
    out = sound.copy()

    out[0:fade_samples]  = (start * ramp_in) + (end * ramp_out) # Beginning of loop
    out[(total_samples-fade_samples):total_samples] = (end * ramp_out) + (start * ramp_in) # End of loop

    return out

def render(sound):
    global current_sound
    #sound = crossfade_loop(sound)
    maxval = np.max(np.abs(sound))
    if maxval == 0:
        sound_int16 = np.int16(sound)
    else:
        sound_int16 = np.int16(sound / maxval * 32767)
    reduced = (sound_int16 * 0.05).astype(np.int16)
    current_sound = reduced
    return reduced


### SOUNDS ###

def wind(width=0.1, intensity=0.5, volume=1.0): # Intensity = volume in this simple scenario
    w1 = LFO((highpass(lowpass(white_noise(duration), 1000), 800)) * slow_noise(10.0, None, duration) * 0.4, 1.0, 'triangle', 0.7)
    w2 = LFO(highpass(lowpass(white_noise(duration), 600), 200), 0.5, 'sine', 0.3)
    w3 = LFO(((highpass(lowpass(white_noise(duration), 1500), 1100)) + 0.05*flute_like_sine(320-width*100, 7, duration)) * slow_noise(20.0, None, duration) * 0.1, 0.2, 'triangle', 0.9)
    f = w1*(1-width*0.5) + w2*width + w3*(1-width)
    return f * volume * intensity

def thunder(volume=1.0): # Improvement experimented on here that failed: since kick drums sound like thunder and are made of sine waves, I could add a drum-like punch sound layered on top of each noise burst
    # Another possible improvement experimented on here: packing the finite loops of the bursts into literal bursts OF bursts, and then loop those bursts of bursts.
    w = lowpass(white_noise(duration), 200) * 0.5
    burst1 = lowpass(white_noise(5), 200) #+ sine(50, 5) * 0.2
    burst1 = adsr_envelope(burst1, attack=0.1, decay=3, sustain_level=0.0, release=0.00)
    burst2 = lowpass(white_noise(5), 500) #+ sine(100, 5) * 0.2
    burst2 = adsr_envelope(burst2, attack=0.01, decay=2, sustain_level=0.0, release=0.00) * 0.8
    burst3 = lowpass(white_noise(5), 50) #+ sine(20, 5) * 0.2
    burst3 = LFO(adsr_envelope(burst3, attack=0.002, decay=4, sustain_level=0.0, release=0.00) * 2, 10, 'sine', 0.5, 5)
    burst4 = highpass(lowpass(white_noise(2), 1000), 300)
    burst4 = adsr_envelope(burst3, attack=0.002, decay=0.5, sustain_level=0.0, release=0.00)
    bob = loop_finite_random(burst1, 2, 4, 0.7) + loop_finite_random(burst3, 2, 4, 0.7)
    bursts = (loop_finite_random(burst1, duration, 0.8, 0.5)
              + loop_finite_random(burst2, duration, 0.7, 0.5)
              + loop_finite_random(burst3, duration, 0.7, 0.5)
              + loop_finite_random(burst4, duration, 0.6, 0.5)
              + loop_finite_random(bob, duration, 0.4, 0.5))
    combined = w + bursts
    combined = LFO_random_smooth(combined, 2, 0.9, duration)

    # There seems to be clipping/crackling happening to this sound in particular. It isn't harsh clipping but just soft clipping-- maybe I need to investigate the render function?
    return combined * volume


# Old, but best I was able to get
def rain(width=1.0, intensity=0.5, volume=1.0): # intensity is from 0 to 1, with 0 being sprinkling, and 1 being pouring
    w = LFO(lowpass(white_noise(), 700), 0.32, depth=0.22)
    w1 = highpass(lowpass(white_noise(), 9000), 5000)
    w2 = smooth_random_amplitude_modulation(w1, 100, 0.15, 2.2) # Similar result to granular synthesis, if not basically the same thing
    w4 = highpass(lowpass(white_noise(), 2200), 900)
    w5 = highpass(smooth_random_amplitude_modulation(w4, 28, 0.0, 2.0), 450)
    w7 = smooth_random_amplitude_modulation(w, 12.0)
    combined = (w * 0.9 + w2 * 1.1 + w5 * 3.0 + w7 * 1.6) * (0.9 + 0.3 * slow_noise(0.08))
    combined = band_EQ(combined, [0.7, 0.9, 1.0, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15])
    combined = combined * intensity
    return combined * volume


def rain_legacy(width=0.1, intensity=0.6, volume=1.0): # A failed attempt at granular synthesis to improve upon the previous rain sound, which sadly ended up sounding better than this one. The older and better sounding rain sound is consequently the one that will be left in the production build for right now.
    center = highpass(lowpass(white_noise(duration), 6000), 2000) * 0.7

    close_drop = (
        sine(320, 0.5) * 0.2
        + highpass(lowpass(white_noise(0.5), 4000), 200) * 0.05
    )
    close_drop = adsr_envelope(close_drop, attack=0.01, decay=0.2, sustain_level=0.0, release=0.00)
    close_drop = pitch_glide(close_drop, 2.0)

    close_drop2 = sine(240, 0.5) * 0.2
    close_drop2 = adsr_envelope(close_drop, attack=0.05, decay=0.4, sustain_level=0.0, release=0.00)
    close_drop2 = pitch_glide(close_drop, 2.0)

    medium_drop = (
        sine(100, 0.4) * 0.3
        + highpass(lowpass(white_noise(0.4), 6000), 1500) * 0.2
    )
    medium_drop = adsr_envelope(medium_drop, attack=0.01, decay=0.05, sustain_level=0.5, release=0.2)

    far_drop = highpass(lowpass(white_noise(0.4), 6000), 1000) * 0.4
    far_drop = adsr_envelope(far_drop, attack=0.008, decay=0.02, sustain_level=0.0, release=0.0)

    farthest_drop = highpass(lowpass(white_noise(0.1), 11000), 9000) * 0.4
    farthest_drop = adsr_envelope(far_drop, attack=0.003, decay=0.01, sustain_level=0.0, release=0.0)

    combined = (
        LFO(loop_finite_random(medium_drop, duration, 20, 0.4), 0.4, 'sine', 0.3)
        + LFO(loop_finite_random(far_drop, duration, 50, 0.9) * 0.6, 0.23, 'sine', 0.1)
        + LFO(loop_finite_random(farthest_drop, duration, 200, 0.9) * 0.9, 0.5, 'sine', 0.2)
        + loop_finite_random(close_drop, duration, 1, 0.9) * 0.05
        + loop_finite_random(close_drop2, duration, 1, 0.9) * 0.04
        + center
    )
    return combined * volume

def ocean(intensity=0.5, volume=1.0): # Intensity is fixed because I wanted to separate the shore audio, but never got around to making a lighter sounding shore due to the constraints on my FM synthesis functions
    w2 = LFO(highpass(lowpass(white_noise(duration), 500), 100), 0.05, 'sine', 0.99)
    w3 = LFO(highpass(lowpass(white_noise(duration), 300), 10), 0.12, 'triangle', 0.99) * 0.5
    w4 = LFO(highpass(white_noise(duration), 300), 0.15, 'triangle', 0.8) * 0.5
    return (w2 + w3 + w4) * volume

def leaves(volume=1.0):
    w = LFO(lowpass(white_noise(duration), 8000), 0.2, 'sine', 0.5) * 0.1
    combined = w
    for i in range(1, 4):
        rustle = adsr_envelope(highpass(white_noise(0.3), 9000 + 200*i), 0.01, 0.03, 0.0, 0.0) / 20
        rustle = loop_finite_random(rustle, duration, 3.0, 0.9)
        combined += rustle
    combined = LFO(combined, 0.2, 'sine', 0.3)
    return combined * volume

def buzzing_electronics(width=0.0, volume=1.0):
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    hum = 0.07 * np.sin(2 * np.pi * (120.0 + (1-width)*60.0) * t)
    hf = 0.05 * np.sin(2 * np.pi * (1800.0 + (1-width)*400.0) * t
                        + 30.0 * np.sin(2 * np.pi * (80.0 + (1-width)*50.0) * t))
    noise_layer = highpass(lowpass(white_noise(duration), 8000), 2000) * 0.05
    w = hum + hf + noise_layer
    w = LFO(w, freq=120.0, waveform='sine', depth=0.12)
    w = w + 0.25 * reverb(w, gain=0.4, depth=2)
    return w * volume

def grumbling_machinery(depth=1.0, intensity=0.1, volume=1.0):
    base = (lowpass(highpass(sawtooth_vibrato(30*(0.5+intensity*0.5), duration, 120*(1-intensity/0.5), 100), 20), 2000)
            + highpass(lowpass(white_noise(duration), 1000), 100) * 0.8) * (0.2+intensity*0.8)
    base = lowpass(base, 3000-depth*2500)
    return base * volume

def faraway_cars(volume=1.0):
    low = highpass(lowpass(LFO(white_noise(duration), freq=0.25, waveform='sine', depth=0.1) * 0.1, 400), 200)
    return low * volume

def crickets(intensity=0.5, volume=1.0):
    chirp_duration = 0.22
    t = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration))
    carrier = sine(2300, chirp_duration) + sine(3800, chirp_duration) * 0.5 + sine(5000, chirp_duration) * 0.05 + lowpass(highpass(white_noise(chirp_duration), 2000), 2500) * 0.3
    noise = np.random.normal(0, 0.08, len(t)) * 0.02
    chirp = LFO(carrier, 70, 'sine', 0.9, chirp_duration) + noise
    chirp = adsr_envelope(chirp, attack=0.12, decay=0.06, sustain_level=0.8, release=0.04)
    gap = np.zeros(int(sample_rate * 0.07))
    single_chirp = np.concatenate([chirp*0.4, gap, chirp])

    default1 = loop_finite(single_chirp, duration, 1.0) * 0.8
    default2 = loop_finite_random(single_chirp, duration, 10, 0.1) * 0.01
    output = default1 + default2 + + offset(default2*2, 2.134) + lowpass(white_noise(duration), 400) * 0.15
    return output * volume

def bees(volume=1.0): # Unfinished
    base = (lowpass(highpass(sawtooth(240, 3), 200), 3000)
            + highpass(lowpass(white_noise(3), 1000), 100) * 0.5)
    base = LFO_random_smooth(base, 5, 0.7, 3)
    buzz = triangular_envelope(base, 3)
    buzzes = loop_finite_random(buzz, duration, 0.3, 0.3)
    buzzes = LFO_random_smooth(buzzes, 5, 0.7, duration)
    return buzzes * volume


if __name__ == "__main__": # When running this file on its own, you can test exports locally. Right here, the crickets export is tested locally.
    #export(crickets(), 'crickets')
    export(rain(), 'test_sounds/rain')
    #export(thunder(), 'test_sounds/thunder')
