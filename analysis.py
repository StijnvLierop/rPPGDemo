import math

import numpy as np
from scipy import signal

# --- Landmark settings ---
FOREHEAD_LANDMARKS = [67, 109, 10, 338, 297, 69, 108, 151, 337, 299]
LEFT_CHEEK_LANDMARKS = [123, 116, 117, 118, 101, 50, 205, 207, 119]
RIGHT_CHEEK_LANDMARKS = [352, 345, 346, 347, 330, 280, 425, 427, 348]
NOSE_BRIDGE_LANDMARKS = [6, 197, 195, 5, 4]
ROI_LANDMARK_LOCATIONS = (FOREHEAD_LANDMARKS + LEFT_CHEEK_LANDMARKS +
                          RIGHT_CHEEK_LANDMARKS + NOSE_BRIDGE_LANDMARKS)
FACE_CONTOUR_LANDMARK_LOCATIONS = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109, 234, 93,
                                   132, 58, 172, 136, 150, 149, 176, 148, 454, 323, 361, 288,
                                   397, 365, 379, 378, 400, 377]


def get_roi_values(frame, face_lms, h: int, w: int):
    """
    Extracts the values and specific landmark locations in a given frame.

    :param frame: The frame to extract the values from.
    :param face_lms: The landmark locations to extract.
    :param h: The height of video frame.
    :param w: The width of the video frame.
    """
    values = []
    for i in ROI_LANDMARK_LOCATIONS:
        # Calculate pixel coordinates
        x = int(face_lms.landmark[i].x * w)
        y = int(face_lms.landmark[i].y * h)

        # Check bounds to see if pixels are in frame
        if 0 <= x < w and 0 <= y < h:
            values.append(frame[y, x, :])
        else:
            continue
    return np.array(values) if len(values) > 0 else np.array([])


def extract_bvp_signal_green(rgb_values: np.ndarray, fs : float) -> np.ndarray:
    """
    Extracts the BVP signal using the POS method.

    :param rgb_values: The RGB values for each landmark location.
    :param fs: The sample frequency.
    :return: The estimated BVP signal.
    """
    # Take average GREEN value of all landmarks per frame and add to green signal array
    green_signal = []
    for frame in rgb_values:
        green_signal.append(np.median(frame[:, 1]))
    green_signal = np.array(green_signal)

    return green_signal


def extract_bvp_signal_pos(rgb_values: np.ndarray, fs: float) -> np.ndarray:
    """
    Extracts the BVP signal using the POS method, matching the POS_WANG implementation.
    """
    # 1. Van ROI-pixels naar RGB-gemiddelden per frame (zoals _process_video)
    rgb_signal = []
    for frame_pixels in rgb_values:
        if frame_pixels.size > 0:
            # POS_WANG gebruikt het gemiddelde van alle pixels in de ROI
            rgb_signal.append(np.mean(frame_pixels, axis=0))

    RGB = np.array(rgb_signal)
    N = RGB.shape[0]

    # Grootte van het sliding window
    window_size = 1.6
    l = math.ceil(window_size * fs)

    # Initialiseer H signaal (1D array voor overlap-add)
    H = np.zeros(N)

    # 2. POS Algoritme kern
    for n in range(N):
        m = n - l
        if m >= 0:
            # Venster selectie
            window = RGB[m:n, :]

            # Stap A: Normalisatie per kanaal (Cn)
            # np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            mean_window = np.mean(window, axis=0)
            if np.any(mean_window == 0): continue
            Cn = window / mean_window

            # Stap B: Matrix operaties
            # In POS_WANG: Cn = np.mat(Cn).H (transpositie naar 3 x L)
            Cn_t = Cn.T

            # Projectie matrix
            projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
            S = np.matmul(projection_matrix, Cn_t)

            # Stap C: Alpha berekening en h bepaling
            # h = S[0, :] + (std(S1) / std(S2)) * S[1, :]
            std_S1 = np.std(S[0, :])
            std_S2 = np.std(S[1, :])

            if std_S2 == 0: continue

            alpha = std_S1 / std_S2
            h = S[0, :] + alpha * S[1, :]

            # Stap D: Mean centering van het resultaat h
            # Dit komt overeen met: h[0, temp] = h[0, temp] - mean_h
            h_centered = h - np.mean(h)

            # Stap E: Overlap-add
            H[m:n] = H[m:n] + h_centered

    # 3. Post-processing (exact volgens POS_WANG)
    # Detrending
    BVP = signal.detrend(H)

    # Bandpass filter (0.75 Hz - 3.0 Hz)
    if len(BVP) > 30:
        nyquist = fs / 2
        # POS_WANG gebruikt [0.75 / fs * 2], wat gelijk is aan 0.75 / nyquist
        low = 0.75 / nyquist
        high = 3.0 / nyquist
        b, a = signal.butter(1, [low, high], btype='bandpass')
        filtered_signal = signal.filtfilt(b, a, BVP.astype(np.double))
    else:
        filtered_signal = BVP

    return filtered_signal


def bvp_signal_to_heart_rate(bvp_signal: np.ndarray, fps: float) -> np.ndarray:
    """
    Extracts a heart rate in BPM from a given BVP signal.

    :param bvp_signal: The BVP signal to extract the heart rate from.
    :param fps: The number of frames per second of the video the frames are from.
    :return: The estimated heart rate in bpm.
    """
    # Choose frequency resolution
    n_fft = 2048

    # Convert signal to the frequency domain
    fft = np.abs(np.fft.rfft(bvp_signal, n=n_fft))

    # Get frequencies in signal
    freqs = np.fft.rfftfreq(n_fft, 1 / fps)

    # Define reasonable bounds for heart rate 0.75 Hz (45 BPM) tot 3.0 Hz (180 BPM)
    lower_bound = 0.75
    upper_bound = 3.0

    # Slice freqs array to look at frequencies in these bounds
    valid_indices = np.where((freqs >= lower_bound) & (freqs <= upper_bound))[0]

    # Look for the most prominent frequency within the correct bounds
    most_prominent_frequency = freqs[valid_indices[np.argmax(fft[valid_indices])]]

    # Multiply by 60 to get BPM (60 seconds in a minute)
    estimated_bpm = most_prominent_frequency * 60

    return estimated_bpm