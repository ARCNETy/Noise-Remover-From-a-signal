"""
Signal Noise Removal Tool
==========================
Interactive noise removal / signal denoising tool that preserves the original
signal as closely as possible while suppressing noise.

Algorithms implemented:
  1. Spectral Subtraction    — subtract estimated noise spectrum from signal
  2. Wiener Filter           — optimal MMSE estimator using noise PSD estimate
  3. Wavelet Denoising       — threshold wavelet coefficients (soft/hard)
  4. Savitzky-Golay          — polynomial smoothing (preserves peak shape)
  5. Median Filter           — remove impulse / spike noise
  6. Kalman Filter           — recursive state estimator (tracks signal)
  7. Moving Average          — simple low-complexity smoother

Key design principles:
  • Original signal is NEVER modified — all processing on a copy
  • Side-by-side original vs denoised comparison at all times
  • SNR improvement, distortion, and spectral fidelity metrics shown live
  • Difference (residual) plot shows exactly what was removed

Requirements:
  pip install scipy matplotlib numpy pywavelets
  (pywavelets optional — wavelet method gracefully falls back if absent)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
import warnings
warnings.filterwarnings('ignore')

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

# ─── Colour palette ───────────────────────────────────────────────────────────

BG_DARK  = '#0d0d1a'
BG_PANEL = '#1a1a2e'
BG_CTRL  = '#0d0d1a'
GRID_C   = '#2a2a3e'
TEXT_PRI = '#B4B2A9'
TEXT_SEC = '#888780'
SPINE_C  = '#2a2a3e'

C_ORIG    = '#378ADD'   # blue    — original signal
C_NOISY   = '#E24B4A'   # red     — noisy signal
C_CLEAN   = '#4CAF82'   # green   — denoised output
C_DIFF    = '#F5A623'   # amber   — removed noise (residual)
C_SPEC    = '#C77DFF'   # violet  — spectrum
C_METRIC  = '#1D9E75'   # teal    — metric lines


# ─── Signal generators ────────────────────────────────────────────────────────

def generate_test_signal(kind, N, fs):
    """Generate a clean test signal."""
    t = np.arange(N) / fs
    if kind == 'Sine 440 Hz':
        return np.sin(2 * np.pi * 440 * t)

    elif kind == 'Multi-tone':
        return (0.6 * np.sin(2 * np.pi * 200  * t) +
                0.4 * np.sin(2 * np.pi * 800  * t) +
                0.2 * np.sin(2 * np.pi * 1500 * t))

    elif kind == 'Chirp':
        return signal.chirp(t, f0=50, f1=fs / 4, t1=t[-1], method='linear')

    elif kind == 'ECG-like':
        # Synthetic ECG: narrow QRS + P + T waves at ~72 BPM
        bpm  = 72
        rr   = fs * 60 / bpm
        sig  = np.zeros(N)
        for beat in np.arange(0, N, rr):
            b = int(beat)
            # QRS complex
            for i, (amp, width) in enumerate([(0.1, 8), (1.0, 4), (-0.3, 6)]):
                offset = b + i * 5 - 5
                if 0 <= offset < N:
                    win = signal.windows.gaussian(width * 6, std=width)
                    end = min(offset + len(win), N)
                    sig[offset:end] += amp * win[:end - offset]
            # T wave
            t_offset = b + int(rr * 0.35)
            if 0 <= t_offset < N:
                tw = signal.windows.gaussian(60, std=15)
                end = min(t_offset + len(tw), N)
                sig[t_offset:end] += 0.3 * tw[:end - t_offset]
        return sig

    elif kind == 'Step signal':
        sig = np.zeros(N)
        for pos in [N // 5, 2 * N // 5, 3 * N // 5, 4 * N // 5]:
            sig[pos:] += 0.5 * (1 if (pos // (N // 5)) % 2 == 0 else -1)
        return sig

    elif kind == 'Speech-like':
        env = np.abs(np.sin(2 * np.pi * 2 * np.arange(N) / fs))
        return env * (0.5 * np.sin(2 * np.pi * 120 * np.arange(N) / fs) +
                      0.3 * np.sin(2 * np.pi * 240 * np.arange(N) / fs) +
                      0.15 * np.sin(2 * np.pi * 480 * np.arange(N) / fs))
    return np.random.randn(N)


def add_noise(clean, noise_type, snr_db, fs):
    """Add noise to a clean signal. Returns noisy signal."""
    sig_power = np.mean(clean ** 2) + 1e-15
    noise_var = sig_power / (10 ** (snr_db / 10))

    N = len(clean)
    if noise_type == 'White Gaussian':
        noise = np.sqrt(noise_var) * np.random.randn(N)

    elif noise_type == 'Pink (1/f)':
        # Pink noise via FFT shaping
        f    = np.fft.rfftfreq(N)
        f[0] = 1e-10
        S    = 1.0 / np.sqrt(f)
        phase = np.random.uniform(0, 2 * np.pi, len(f))
        X     = S * np.exp(1j * phase)
        pink  = np.fft.irfft(X, n=N)
        pink /= (np.std(pink) + 1e-15)
        noise = np.sqrt(noise_var) * pink

    elif noise_type == 'Impulse spikes':
        noise = np.zeros(N)
        n_spikes = max(1, int(N * 0.01))
        idx = np.random.choice(N, n_spikes, replace=False)
        noise[idx] = np.sqrt(noise_var) * 10 * np.random.randn(n_spikes)

    elif noise_type == 'Narrowband':
        # Sinusoidal interference at 50 Hz (power-line hum)
        t     = np.arange(N) / fs
        noise = np.sqrt(2 * noise_var) * np.sin(2 * np.pi * 50 * t)

    elif noise_type == 'Band-limited':
        wn = np.random.randn(N)
        b, a = signal.butter(4, [0.4, 0.49], btype='band')
        noise = signal.filtfilt(b, a, wn)
        noise *= np.sqrt(noise_var) / (np.std(noise) + 1e-15)

    else:
        noise = np.sqrt(noise_var) * np.random.randn(N)

    return clean + noise, noise


# ─── Denoising algorithms ─────────────────────────────────────────────────────

def denoise_spectral_subtraction(noisy, fs, alpha=2.0, beta=0.01,
                                  noise_frames=10):
    """
    Spectral subtraction — estimates noise PSD from first N frames,
    subtracts it from signal spectrum.
    alpha : over-subtraction factor (1–3)
    beta  : spectral floor (prevents musical noise)
    """
    N        = len(noisy)
    frame_sz = min(256, N // noise_frames)
    if frame_sz < 4:
        return noisy.copy()

    # Estimate noise PSD from first few frames
    noise_est = noisy[:frame_sz * noise_frames]
    noise_psd = np.mean(np.abs(fft(
        noise_est.reshape(noise_frames, frame_sz), axis=1)) ** 2, axis=0)
    noise_psd = np.concatenate([noise_psd, noise_psd[::-1]])[:N]

    # Process full signal
    X    = fft(noisy)
    Xmag = np.abs(X)
    Xpha = np.angle(X)

    # Subtract noise estimate
    noise_full = np.tile(noise_psd, int(np.ceil(N / len(noise_psd))))[:N]
    mag_clean  = np.maximum(
        np.sqrt(np.maximum(Xmag ** 2 - alpha * noise_full, 0)),
        beta * Xmag)

    # Reconstruct
    X_clean = mag_clean * np.exp(1j * Xpha)
    return np.real(ifft(X_clean))


def denoise_wiener(noisy, fs, noise_frames=8):
    """
    Wiener filter — computes optimal gain H(f) = Psignal / (Psignal + Pnoise).
    Estimates noise PSD from first frames; signal PSD from full spectrum.
    """
    N        = len(noisy)
    frame_sz = min(512, N // max(noise_frames, 2))
    if frame_sz < 4:
        return noisy.copy()

    # Noise PSD estimate
    noise_chunk = noisy[:frame_sz * noise_frames]
    frames      = noise_chunk[:frame_sz * noise_frames].reshape(noise_frames, frame_sz)
    noise_psd_f = np.mean(np.abs(fft(frames, axis=1)) ** 2, axis=0)

    X        = fft(noisy)
    Px       = np.abs(X) ** 2          # noisy signal PSD

    # Tile noise PSD to match signal length
    noise_full = np.tile(noise_psd_f, int(np.ceil(N / frame_sz)))[:N]

    # Wiener gain (floored at 0.01 to avoid division artefacts)
    Psig = np.maximum(Px - noise_full, 0)
    H    = Psig / (Px + 1e-15)
    H    = np.maximum(H, 0.01)

    X_clean = H * X
    return np.real(ifft(X_clean))


def denoise_wavelet(noisy, level=4, wavelet='db4', mode='soft',
                     threshold_rule='universal'):
    """
    Wavelet denoising — DWT, threshold coefficients, IDWT.
    mode  : 'soft' (smooth) or 'hard' (preserve edges)
    rule  : 'universal' (√(2 ln N) · σ) or 'bayes' (signal-adaptive)
    """
    if not HAS_PYWT:
        # Graceful fallback: Savitzky-Golay
        return denoise_savgol(noisy, window=11, poly=3)

    coeffs = pywt.wavedec(noisy, wavelet, level=level)
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745     # MAD noise estimate

    if threshold_rule == 'universal':
        thr = sigma * np.sqrt(2 * np.log(len(noisy)))
    else:  # bayes
        sig2  = max(np.var(noisy) - sigma ** 2, 1e-15)
        thr   = sigma ** 2 / np.sqrt(sig2)

    coeffs_thr = [coeffs[0]]   # keep approximation coefficients unchanged
    for c in coeffs[1:]:
        coeffs_thr.append(pywt.threshold(c, thr, mode=mode))

    return pywt.waverec(coeffs_thr, wavelet)[:len(noisy)]


def denoise_savgol(noisy, window=21, poly=3):
    """
    Savitzky-Golay filter — fits local polynomials, preserves peak shape.
    window : must be odd and > poly
    """
    w = window if window % 2 == 1 else window + 1
    w = max(w, poly + 2)
    return signal.savgol_filter(noisy, window_length=w, polyorder=poly)


def denoise_median(noisy, kernel=5):
    """Median filter — excellent for impulse/spike noise."""
    k = kernel if kernel % 2 == 1 else kernel + 1
    return signal.medfilt(noisy, kernel_size=k)


def denoise_kalman(noisy, process_var=1e-5, obs_var=0.01):
    """
    1-D Kalman filter for signal tracking.
    process_var : Q — how fast the true signal can change
    obs_var     : R — measurement noise variance
    Models: x[n] = x[n-1] + w (random walk) + v (observation noise)
    """
    N   = len(noisy)
    out = np.zeros(N)
    x   = noisy[0]    # initial state estimate
    P   = 1.0         # initial error covariance

    Q = process_var
    R = obs_var

    for n in range(N):
        # Predict
        x_pred = x
        P_pred = P + Q

        # Update (Kalman gain)
        K = P_pred / (P_pred + R)
        x = x_pred + K * (noisy[n] - x_pred)
        P = (1 - K) * P_pred

        out[n] = x

    return out


def denoise_moving_average(noisy, window=11):
    """Simple moving average — fastest, most basic smoother."""
    w = max(window, 1)
    kernel = np.ones(w) / w
    # Use 'same' mode and reflect-pad to avoid edge distortion
    padded = np.pad(noisy, w // 2, mode='reflect')
    return np.convolve(padded, kernel, mode='valid')[:len(noisy)]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(original, noisy, denoised):
    """
    Returns dict of signal quality metrics.
    All metrics compare against the ORIGINAL clean signal.
    """
    eps = 1e-15
    N   = min(len(original), len(denoised))
    orig = original[:N]
    nois = noisy[:N]
    den  = denoised[:N]

    # SNR improvement
    noise_in   = nois - orig
    noise_out  = den  - orig
    snr_in     = 10 * np.log10(np.mean(orig**2) / (np.mean(noise_in**2)  + eps))
    snr_out    = 10 * np.log10(np.mean(orig**2) / (np.mean(noise_out**2) + eps))
    snr_gain   = snr_out - snr_in

    # RMSE vs original
    rmse_noisy = np.sqrt(np.mean(noise_in ** 2))
    rmse_clean = np.sqrt(np.mean(noise_out ** 2))

    # Pearson correlation with original
    corr = np.corrcoef(orig, den)[0, 1]

    # Spectral distortion (log-spectral distance)
    S_orig = np.abs(fft(orig)) + eps
    S_den  = np.abs(fft(den))  + eps
    lsd    = np.sqrt(np.mean((20 * np.log10(S_orig / S_den)) ** 2))

    return dict(
        snr_in=snr_in, snr_out=snr_out, snr_gain=snr_gain,
        rmse_noisy=rmse_noisy, rmse_clean=rmse_clean,
        corr=corr, lsd=lsd
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_hz(f):
    return f'{f/1000:.1f} kHz' if f >= 1000 else f'{f:.0f} Hz'

def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TEXT_SEC, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    ax.grid(True, color=GRID_C, linewidth=0.5)
    if title:  ax.set_title(title,  color=TEXT_PRI, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_SEC, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_SEC, fontsize=8)


# ─── Main GUI ─────────────────────────────────────────────────────────────────

ALGO_NAMES  = ['Spectral Sub', 'Wiener', 'Wavelet',
               'Savitzky-Golay', 'Median', 'Kalman', 'Moving Avg']
SIG_NAMES   = ['Sine 440 Hz', 'Multi-tone', 'Chirp',
               'ECG-like', 'Step signal', 'Speech-like']
NOISE_NAMES = ['White Gaussian', 'Pink (1/f)', 'Impulse spikes',
               'Narrowband', 'Band-limited']


class NoiseRemovalTool:
    """
    Interactive noise removal GUI.
    The original clean signal is stored separately and NEVER modified.
    All denoising operates only on the noisy copy.
    """

    def __init__(self):
        # Signal parameters
        self.fs       = 4000.0
        self.N        = 2000
        self.sig_kind = 'ECG-like'
        self.noise_t  = 'White Gaussian'
        self.snr_db   = 10.0
        self.algo     = 'Wiener'

        # Algorithm parameters
        self.alpha    = 2.0     # spectral subtraction over-subtraction
        self.beta     = 0.01    # spectral floor
        self.wv_level = 4       # wavelet decomposition levels
        self.sg_win   = 21      # Savitzky-Golay window
        self.sg_poly  = 3       # Savitzky-Golay poly order
        self.med_k    = 5       # median filter kernel
        self.kal_q    = 1e-5    # Kalman process variance
        self.kal_r    = 0.01    # Kalman observation variance
        self.ma_win   = 11      # moving average window

        # Stored signals — originals are sacred
        self._original = None
        self._noisy    = None
        self._noise    = None
        self._denoised = None

        self._build_gui()
        self._generate_and_process()
        plt.show()

    # ── GUI ───────────────────────────────────────────────────────────────

    def _build_gui(self):
        self.fig = plt.figure(figsize=(16, 10), facecolor=BG_PANEL)
        self.fig.canvas.manager.set_window_title(
            'Signal Noise Removal Tool — Original Preserved')

        outer = gridspec.GridSpec(
            1, 2, width_ratios=[1, 3.2], wspace=0.04,
            left=0.01, right=0.99, top=0.97, bottom=0.04)

        # ── Left controls ─────────────────────────────────────────────────
        cgs = gridspec.GridSpecFromSubplotSpec(
            14, 1, subplot_spec=outer[0], hspace=0.52)

        ax_algo  = self.fig.add_subplot(cgs[0:4])
        ax_sig   = self.fig.add_subplot(cgs[4:7])
        ax_noise = self.fig.add_subplot(cgs[7:10])
        ax_snr   = self.fig.add_subplot(cgs[10])
        ax_p1    = self.fig.add_subplot(cgs[11])
        ax_p2    = self.fig.add_subplot(cgs[12])
        ax_reset = self.fig.add_subplot(cgs[13])

        for ax in [ax_algo, ax_sig, ax_noise, ax_snr,
                   ax_p1, ax_p2, ax_reset]:
            ax.set_facecolor(BG_CTRL)

        # Algorithm radio
        self.radio_algo = RadioButtons(
            ax_algo, ALGO_NAMES, active=1, activecolor=C_CLEAN)
        ax_algo.set_title('Denoising algorithm', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.radio_algo.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(8)
        self.radio_algo.on_clicked(self._on_algo)

        # Signal radio
        self.radio_sig = RadioButtons(
            ax_sig, SIG_NAMES, active=3, activecolor=C_ORIG)
        ax_sig.set_title('Test signal', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.radio_sig.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(8)
        self.radio_sig.on_clicked(self._on_sig)

        # Noise radio
        self.radio_noise = RadioButtons(
            ax_noise, NOISE_NAMES, active=0, activecolor=C_NOISY)
        ax_noise.set_title('Noise type', color=TEXT_PRI, fontsize=9, pad=2)
        for lbl in self.radio_noise.labels:
            lbl.set_color(TEXT_PRI); lbl.set_fontsize(8)
        self.radio_noise.on_clicked(self._on_noise)

        # SNR slider
        sc = BG_CTRL
        self.sl_snr = Slider(ax_snr, 'Input SNR\n(dB)',
                             0, 30, valinit=self.snr_db, valstep=1,
                             color=C_NOISY, initcolor='none', facecolor=sc)
        self.sl_snr.label.set_color(TEXT_PRI);  self.sl_snr.label.set_fontsize(8)
        self.sl_snr.valtext.set_color('#ffffff'); self.sl_snr.valtext.set_fontsize(8)
        self.sl_snr.on_changed(self._on_param)

        # Param slider 1 (context-sensitive)
        self.sl_p1 = Slider(ax_p1, 'Param 1',
                            0.01, 10, valinit=self.alpha,
                            color=C_SPEC, initcolor='none', facecolor=sc)
        self.sl_p1.label.set_color(TEXT_PRI);  self.sl_p1.label.set_fontsize(8)
        self.sl_p1.valtext.set_color('#ffffff'); self.sl_p1.valtext.set_fontsize(8)
        self.sl_p1.on_changed(self._on_param)

        # Param slider 2 (context-sensitive)
        self.sl_p2 = Slider(ax_p2, 'Param 2',
                            1, 50, valinit=self.sg_win, valstep=2,
                            color=C_DIFF, initcolor='none', facecolor=sc)
        self.sl_p2.label.set_color(TEXT_PRI);  self.sl_p2.label.set_fontsize(8)
        self.sl_p2.valtext.set_color('#ffffff'); self.sl_p2.valtext.set_fontsize(8)
        self.sl_p2.on_changed(self._on_param)

        # Reset / regenerate noise button
        self.btn_reset = Button(ax_reset, 'New noise realisation',
                                color=BG_CTRL, hovercolor=BG_PANEL)
        self.btn_reset.label.set_color(TEXT_PRI)
        self.btn_reset.label.set_fontsize(9)
        self.btn_reset.on_clicked(self._on_reset)

        # ── Right: 3×2 plot grid ──────────────────────────────────────────
        pgs = gridspec.GridSpecFromSubplotSpec(
            3, 2, subplot_spec=outer[1], hspace=0.48, wspace=0.3)

        self.ax_time   = self.fig.add_subplot(pgs[0, 0])  # time domain overlay
        self.ax_diff   = self.fig.add_subplot(pgs[0, 1])  # removed noise
        self.ax_spec   = self.fig.add_subplot(pgs[1, 0])  # spectra comparison
        self.ax_spect  = self.fig.add_subplot(pgs[1, 1])  # spectrogram
        self.ax_metric = self.fig.add_subplot(pgs[2, 0])  # metric bars
        self.ax_resid  = self.fig.add_subplot(pgs[2, 1])  # residual histogram

        for ax in [self.ax_time, self.ax_diff, self.ax_spec,
                   self.ax_spect, self.ax_metric, self.ax_resid]:
            _style_ax(ax)

        # Legend patch (top right)
        self.fig.text(0.36, 0.965, '─ Original',
                      color=C_ORIG, fontsize=8, va='top')
        self.fig.text(0.44, 0.965, '─ Noisy',
                      color=C_NOISY, fontsize=8, va='top')
        self.fig.text(0.52, 0.965, '─ Denoised',
                      color=C_CLEAN, fontsize=8, va='top')
        self.fig.text(0.61, 0.965, '─ Removed',
                      color=C_DIFF, fontsize=8, va='top')

        # Metrics bar
        self.met_ax = self.fig.add_axes([0.34, 0.005, 0.65, 0.028])
        self.met_ax.axis('off')
        self.met_txt = self.met_ax.text(
            0.5, 0.5, '', ha='center', va='center',
            color=TEXT_PRI, fontsize=8, transform=self.met_ax.transAxes)

    # ── Slider context update ─────────────────────────────────────────────

    def _update_slider_labels(self):
        """Set slider labels and ranges appropriate for the current algorithm."""
        algo = self.algo
        if algo == 'Spectral Sub':
            self.sl_p1.label.set_text('Over-sub α\n(1–3)')
            self.sl_p1.valmin, self.sl_p1.valmax = 1.0, 5.0
            self.sl_p1.set_val(np.clip(self.alpha, 1, 5))
            self.sl_p2.label.set_text('Floor β\n(0.001–0.1)')
            self.sl_p2.valmin, self.sl_p2.valmax = 1, 20
            self.sl_p2.set_val(5)

        elif algo == 'Wiener':
            self.sl_p1.label.set_text('Noise frames\n(estimate)')
            self.sl_p1.valmin, self.sl_p1.valmax = 1, 10
            self.sl_p1.set_val(5)
            self.sl_p2.label.set_text('(unused)')
            self.sl_p2.set_val(1)

        elif algo == 'Wavelet':
            self.sl_p1.label.set_text('Decomp levels\n(1–8)')
            self.sl_p1.valmin, self.sl_p1.valmax = 1, 8
            self.sl_p1.set_val(self.wv_level)
            self.sl_p2.label.set_text('(unused)')
            self.sl_p2.set_val(1)

        elif algo == 'Savitzky-Golay':
            self.sl_p1.label.set_text('Window length\n(odd, ≥5)')
            self.sl_p1.valmin, self.sl_p1.valmax = 5, 101
            self.sl_p1.set_val(self.sg_win)
            self.sl_p2.label.set_text('Poly order\n(1–5)')
            self.sl_p2.valmin, self.sl_p2.valmax = 1, 5
            self.sl_p2.set_val(self.sg_poly)

        elif algo == 'Median':
            self.sl_p1.label.set_text('Kernel size\n(odd)')
            self.sl_p1.valmin, self.sl_p1.valmax = 3, 51
            self.sl_p1.set_val(self.med_k)
            self.sl_p2.label.set_text('(unused)')
            self.sl_p2.set_val(1)

        elif algo == 'Kalman':
            self.sl_p1.label.set_text('Process var Q\n(×1e-5)')
            self.sl_p1.valmin, self.sl_p1.valmax = 0.01, 10
            self.sl_p1.set_val(self.kal_q * 1e5)
            self.sl_p2.label.set_text('Obs var R\n(×0.01)')
            self.sl_p2.valmin, self.sl_p2.valmax = 1, 50
            self.sl_p2.set_val(self.kal_r * 100)

        elif algo == 'Moving Avg':
            self.sl_p1.label.set_text('Window size\n(samples)')
            self.sl_p1.valmin, self.sl_p1.valmax = 1, 101
            self.sl_p1.set_val(self.ma_win)
            self.sl_p2.label.set_text('(unused)')
            self.sl_p2.set_val(1)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_algo(self, label):
        self.algo = label
        self._update_slider_labels()
        self._process_only()

    def _on_sig(self, label):
        self.sig_kind = label
        self._generate_and_process()

    def _on_noise(self, label):
        self.noise_t = label
        self._generate_and_process()

    def _on_param(self, _):
        self.snr_db = float(self.sl_snr.val)
        self._generate_and_process()

    def _on_reset(self, _):
        """Regenerate noise (new random seed) — original untouched."""
        self._generate_and_process()

    # ── Signal generation ─────────────────────────────────────────────────

    def _generate_and_process(self):
        """Generate original + noisy signals, then denoise."""
        self.snr_db = float(self.sl_snr.val)

        # Store original — this is the ground truth, NEVER modified
        self._original = generate_test_signal(self.sig_kind, self.N, self.fs)

        # Add noise to a copy
        self._noisy, self._noise = add_noise(
            self._original, self.noise_t, self.snr_db, self.fs)

        self._process_only()

    def _process_only(self):
        """Run the selected denoising algorithm on the noisy signal."""
        if self._noisy is None:
            return

        p1 = float(self.sl_p1.val)
        p2 = float(self.sl_p2.val)

        algo = self.algo
        try:
            if algo == 'Spectral Sub':
                self._denoised = denoise_spectral_subtraction(
                    self._noisy, self.fs,
                    alpha=np.clip(p1, 1, 5),
                    beta=np.clip(p2 / 1000, 0.001, 0.1),
                    noise_frames=8)

            elif algo == 'Wiener':
                self._denoised = denoise_wiener(
                    self._noisy, self.fs,
                    noise_frames=max(2, int(p1)))

            elif algo == 'Wavelet':
                self._denoised = denoise_wavelet(
                    self._noisy,
                    level=max(1, min(int(p1), 8)),
                    wavelet='db4', mode='soft')

            elif algo == 'Savitzky-Golay':
                win = max(5, int(p1))
                win = win if win % 2 == 1 else win + 1
                poly = max(1, min(int(p2), win - 1))
                self._denoised = denoise_savgol(self._noisy, win, poly)

            elif algo == 'Median':
                k = max(3, int(p1))
                k = k if k % 2 == 1 else k + 1
                self._denoised = denoise_median(self._noisy, k)

            elif algo == 'Kalman':
                self._denoised = denoise_kalman(
                    self._noisy,
                    process_var=np.clip(p1 * 1e-5, 1e-8, 1e-2),
                    obs_var=np.clip(p2 * 0.01, 1e-4, 1.0))

            elif algo == 'Moving Avg':
                self._denoised = denoise_moving_average(
                    self._noisy, window=max(1, int(p1)))

            # Safety: ensure same length as original
            N = len(self._original)
            if len(self._denoised) > N:
                self._denoised = self._denoised[:N]
            elif len(self._denoised) < N:
                self._denoised = np.pad(
                    self._denoised, (0, N - len(self._denoised)), mode='edge')

        except Exception as ex:
            print(f'[Denoising error] {ex}')
            self._denoised = self._noisy.copy()

        self._draw_all()

    # ── Plots ─────────────────────────────────────────────────────────────

    def _draw_all(self):
        self._plot_time()
        self._plot_diff()
        self._plot_spectra()
        self._plot_spectrogram()
        self._plot_metrics()
        self._plot_residual()
        self._update_metrics_bar()
        self.fig.canvas.draw_idle()

    def _plot_time(self):
        ax = self.ax_time
        ax.cla()
        _style_ax(ax, 'Time domain — Original vs Noisy vs Denoised',
                  'Sample (n)', 'Amplitude')

        N   = len(self._original)
        n   = np.arange(N)
        # Downsample for display
        step = max(1, N // 2000)

        ax.plot(n[::step], self._noisy[::step],
                color=C_NOISY, linewidth=0.6, alpha=0.55, label='Noisy')
        ax.plot(n[::step], self._denoised[::step],
                color=C_CLEAN, linewidth=1.2, alpha=0.9, label='Denoised')
        ax.plot(n[::step], self._original[::step],
                color=C_ORIG, linewidth=0.9, alpha=0.75,
                linestyle='--', label='Original')

        ax.set_xlim(0, N)
        ax.legend(fontsize=7, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                  edgecolor=SPINE_C, loc='upper right')

    def _plot_diff(self):
        ax = self.ax_diff
        ax.cla()
        _style_ax(ax, 'Removed component (noisy − denoised)',
                  'Sample (n)', 'Amplitude')

        removed = self._noisy - self._denoised
        n = np.arange(len(removed))
        step = max(1, len(removed) // 2000)

        ax.plot(n[::step], removed[::step],
                color=C_DIFF, linewidth=0.8, alpha=0.85)
        ax.fill_between(n[::step], removed[::step], 0,
                        alpha=0.15, color=C_DIFF)
        ax.axhline(0, color=GRID_C, linewidth=0.6)
        ax.set_xlim(0, len(removed))

        # Overlay true noise for comparison
        if self._noise is not None:
            ax.plot(n[::step], self._noise[::step],
                    color=TEXT_SEC, linewidth=0.5, alpha=0.4,
                    linestyle=':', label='True noise')
            ax.legend(fontsize=6, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                      edgecolor=SPINE_C)

    def _plot_spectra(self):
        ax = self.ax_spec
        ax.cla()
        _style_ax(ax, 'Magnitude spectrum comparison',
                  'Frequency (Hz)', 'Magnitude (dB)')

        def spec_db(x):
            X = np.abs(fft(x))
            X = X[:len(x) // 2]
            return 20 * np.log10(X + 1e-15)

        freqs = np.fft.rfftfreq(len(self._original), d=1.0 / self.fs)
        freqs = freqs[:len(self._original) // 2]

        ax.semilogx(freqs[1:], spec_db(self._noisy)[1:],
                    color=C_NOISY, linewidth=0.7, alpha=0.65,
                    label='Noisy')
        ax.semilogx(freqs[1:], spec_db(self._denoised)[1:],
                    color=C_CLEAN, linewidth=1.2, alpha=0.9,
                    label='Denoised')
        ax.semilogx(freqs[1:], spec_db(self._original)[1:],
                    color=C_ORIG, linewidth=0.9, alpha=0.75,
                    linestyle='--', label='Original')

        ax.set_xlim(freqs[1], self.fs / 2)
        ax.legend(fontsize=7, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                  edgecolor=SPINE_C, loc='lower left')

    def _plot_spectrogram(self):
        ax = self.ax_spect
        ax.cla()
        _style_ax(ax, 'Denoised spectrogram',
                  'Time (s)', 'Frequency (Hz)')

        nfft    = min(256, self.N // 4)
        noverlap= nfft * 3 // 4

        try:
            f, t, S = signal.spectrogram(
                self._denoised, fs=self.fs,
                nperseg=nfft, noverlap=noverlap)
            S_db = 10 * np.log10(S + 1e-15)
            im = ax.pcolormesh(t, f, S_db, shading='auto', cmap='magma',
                               vmin=np.percentile(S_db, 10),
                               vmax=np.percentile(S_db, 99))
            cb = self.fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
            cb.ax.tick_params(colors=TEXT_SEC, labelsize=6)
            cb.set_label('dB', color=TEXT_SEC, fontsize=7)
        except Exception:
            ax.text(0.5, 0.5, 'Spectrogram unavailable',
                    ha='center', va='center', color=TEXT_SEC,
                    transform=ax.transAxes)

    def _plot_metrics(self):
        ax = self.ax_metric
        ax.cla()
        _style_ax(ax, 'Signal quality metrics', '', '')

        m = compute_metrics(self._original, self._noisy, self._denoised)

        labels = ['SNR in\n(dB)', 'SNR out\n(dB)', 'SNR gain\n(dB)',
                  'RMSE\nnoisy×10', 'RMSE\nclean×10', 'Corr\n×100', 'LSD']
        values = [m['snr_in'], m['snr_out'], m['snr_gain'],
                  m['rmse_noisy'] * 10, m['rmse_clean'] * 10,
                  m['corr'] * 100, m['lsd']]
        colours = [C_NOISY, C_CLEAN,
                   C_METRIC if m['snr_gain'] > 0 else C_NOISY,
                   C_NOISY, C_CLEAN, C_METRIC, C_DIFF]

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colours, alpha=0.8, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, color=TEXT_PRI)
        ax.axhline(0, color=GRID_C, linewidth=0.6)

        for bar, val in zip(bars, values):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + abs(max(values, default=0)) * 0.02,
                    f'{val:.1f}',
                    ha='center', va='bottom',
                    color=TEXT_PRI, fontsize=6.5)

    def _plot_residual(self):
        ax = self.ax_resid
        ax.cla()
        _style_ax(ax, 'Residual distribution (denoised − original)',
                  'Amplitude', 'Count')

        residual = self._denoised[:len(self._original)] - self._original
        ax.hist(residual, bins=60, color=C_CLEAN, alpha=0.75, edgecolor='none')

        # Overlay noisy residual for comparison
        noisy_res = self._noisy - self._original
        ax.hist(noisy_res, bins=60, color=C_NOISY, alpha=0.35,
                edgecolor='none', label='Noisy residual')

        ax.axvline(0, color=TEXT_SEC, linewidth=0.8, linestyle='--')
        ax.axvline(residual.mean(), color=C_CLEAN, linewidth=1,
                   linestyle=':', label=f'Mean={residual.mean():.4f}')

        ax.legend(fontsize=6, facecolor=BG_DARK, labelcolor=TEXT_PRI,
                  edgecolor=SPINE_C)
        ax.text(0.98, 0.95,
                f'σ = {residual.std():.4f}',
                transform=ax.transAxes, ha='right', va='top',
                color=C_CLEAN, fontsize=7)

    def _update_metrics_bar(self):
        m = compute_metrics(self._original, self._noisy, self._denoised)
        wav_note = '' if HAS_PYWT else ' (pywavelets not installed)'

        txt = (
            f'  {self.algo}{wav_note}   '
            f'Signal: {self.sig_kind}   Noise: {self.noise_t}   '
            f'SNR in: {m["snr_in"]:.1f} dB   |   '
            f'SNR out: {m["snr_out"]:.1f} dB   '
            f'Gain: +{m["snr_gain"]:.1f} dB   '
            f'Corr: {m["corr"]:.4f}   '
            f'RMSE: {m["rmse_clean"]:.4f}   '
            f'LSD: {m["lsd"]:.2f} dB'
        )
        self.met_txt.set_text(txt)


# ─── Non-interactive API ──────────────────────────────────────────────────────

def denoise_signal(noisy, algo='wiener', fs=8000.0, **kwargs):
    """
    Standalone denoising function.

    Parameters
    ----------
    noisy : array_like   — noisy input signal (original is untouched)
    algo  : str          — 'spectral' | 'wiener' | 'wavelet' |
                           'savgol' | 'median' | 'kalman' | 'mavg'
    fs    : float        — sample rate (Hz)
    **kwargs             — algorithm-specific parameters

    Returns
    -------
    denoised : ndarray — denoised signal (same length as input)

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 8000)
    >>> clean = np.sin(2 * np.pi * 440 * t)
    >>> noisy = clean + 0.3 * np.random.randn(8000)
    >>> out = denoise_signal(noisy, algo='wiener', fs=8000)
    >>> out = denoise_signal(noisy, algo='wavelet', level=5)
    >>> out = denoise_signal(noisy, algo='kalman', process_var=1e-4)
    """
    x = np.asarray(noisy, dtype=float).copy()    # working copy — original safe

    if algo == 'spectral':
        return denoise_spectral_subtraction(
            x, fs,
            alpha=kwargs.get('alpha', 2.0),
            beta=kwargs.get('beta', 0.01))
    elif algo == 'wiener':
        return denoise_wiener(x, fs,
                              noise_frames=kwargs.get('noise_frames', 8))
    elif algo == 'wavelet':
        return denoise_wavelet(x,
                               level=kwargs.get('level', 4),
                               wavelet=kwargs.get('wavelet', 'db4'),
                               mode=kwargs.get('mode', 'soft'))
    elif algo == 'savgol':
        return denoise_savgol(x,
                              window=kwargs.get('window', 21),
                              poly=kwargs.get('poly', 3))
    elif algo == 'median':
        return denoise_median(x, kernel=kwargs.get('kernel', 5))
    elif algo == 'kalman':
        return denoise_kalman(x,
                              process_var=kwargs.get('process_var', 1e-5),
                              obs_var=kwargs.get('obs_var', 0.01))
    elif algo == 'mavg':
        return denoise_moving_average(x, window=kwargs.get('window', 11))
    else:
        raise ValueError(f"Unknown algorithm: {algo}. "
                         "Choose: spectral, wiener, wavelet, savgol, "
                         "median, kalman, mavg")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    NoiseRemovalTool()

    # ── Non-interactive examples (uncomment to use) ──────────────────────
    # import numpy as np
    # t     = np.linspace(0, 1, 8000)
    # clean = np.sin(2 * np.pi * 440 * t)
    # noisy = clean + 0.3 * np.random.randn(8000)
    #
    # out_w  = denoise_signal(noisy, algo='wiener',   fs=8000)
    # out_wv = denoise_signal(noisy, algo='wavelet',  level=5)
    # out_sg = denoise_signal(noisy, algo='savgol',   window=31, poly=3)
    # out_k  = denoise_signal(noisy, algo='kalman',   process_var=1e-4, obs_var=0.05)
    # out_m  = denoise_signal(noisy, algo='median',   kernel=7)
    # out_ss = denoise_signal(noisy, algo='spectral', fs=8000, alpha=2.0)
