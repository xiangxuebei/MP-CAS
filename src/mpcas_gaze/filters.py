import numpy as np


class OneEuroFilter:
    class LowPassFilter:
        def __init__(self, alpha):
            self.alpha = alpha
            self.initialized = False
            self.last_raw = 0.0
            self.last_filtered = 0.0

        def filter(self, value, alpha=None):
            if alpha is not None:
                self.alpha = alpha
            if self.initialized:
                result = self.alpha * value + (1.0 - self.alpha) * self.last_filtered
            else:
                result = value
                self.initialized = True
            self.last_raw = value
            self.last_filtered = result
            return result

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x = self.LowPassFilter(self.alpha(self.min_cutoff))
        self.dx = self.LowPassFilter(self.alpha(self.d_cutoff))
        self.last_time = None

    def alpha(self, cutoff):
        te = 1.0 / max(self.freq, 1.0e-6)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, value, timestamp=None):
        if self.last_time is not None and timestamp is not None and timestamp > self.last_time:
            self.freq = 1.0 / (timestamp - self.last_time)
        self.last_time = timestamp
        derivative = (value - self.x.last_raw) * self.freq if self.x.initialized else 0.0
        ed = self.dx.filter(derivative, self.alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(ed)
        return self.x.filter(value, self.alpha(cutoff))


class KalmanFilter2D:
    def __init__(self, dt=1.0, process_noise=1.0e-4, measurement_noise=1.0e-2):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.f = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.r = measurement_noise * np.eye(2, dtype=np.float32)
        self.q = process_noise * np.eye(4, dtype=np.float32)
        self.p = np.eye(4, dtype=np.float32)

    def step(self, z):
        z = np.asarray(z, dtype=np.float32).reshape(2, 1)
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        y = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(4, dtype=np.float32) - k @ self.h) @ self.p
        return self.x[:2, 0]


def heuristic_filter(points, look_ahead=2, threshold=1.5):
    points = np.asarray(points, dtype=np.float32).copy()
    if len(points) < look_ahead * 2 + 1:
        return points
    output = points.copy()
    for idx in range(look_ahead, len(points) - look_ahead):
        before = np.mean(points[idx - look_ahead : idx], axis=0)
        after = np.mean(points[idx + 1 : idx + look_ahead + 1], axis=0)
        local = (before + after) / 2.0
        if np.linalg.norm(points[idx] - local) > threshold and np.linalg.norm(before - after) < threshold:
            output[idx] = local
    return output


def apply_filter(points, mode="none", **kwargs):
    mode = (mode or "none").lower()
    points = np.asarray(points, dtype=np.float32)
    if mode in {"none", "raw"}:
        return points
    if mode == "heuristic":
        return heuristic_filter(
            points,
            look_ahead=int(kwargs.get("look_ahead", 2)),
            threshold=float(kwargs.get("threshold", 1.5)),
        )
    if mode == "one_euro":
        fx = OneEuroFilter(
            freq=float(kwargs.get("freq", 30.0)),
            min_cutoff=float(kwargs.get("min_cutoff", 1.0)),
            beta=float(kwargs.get("beta", 0.007)),
            d_cutoff=float(kwargs.get("d_cutoff", 1.0)),
        )
        fy = OneEuroFilter(
            freq=float(kwargs.get("freq", 30.0)),
            min_cutoff=float(kwargs.get("min_cutoff", 1.0)),
            beta=float(kwargs.get("beta", 0.007)),
            d_cutoff=float(kwargs.get("d_cutoff", 1.0)),
        )
        return np.array([[fx.filter(p[0]), fy.filter(p[1])] for p in points], dtype=np.float32)
    if mode == "kalman":
        filt = KalmanFilter2D(
            dt=float(kwargs.get("dt", 1.0)),
            process_noise=float(kwargs.get("process_noise", 1.0e-4)),
            measurement_noise=float(kwargs.get("measurement_noise", 1.0e-2)),
        )
        return np.array([filt.step(p) for p in points], dtype=np.float32)
    raise ValueError(f"Unsupported filter mode: {mode}")

