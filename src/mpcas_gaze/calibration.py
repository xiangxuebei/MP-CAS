import numpy as np


MVO_SVR_PARAMS = {
    "x": {"C": 3.47527619, "epsilon": 6.80388839e-2, "gamma": 1.0e-3},
    "y": {"C": 3.47527619, "epsilon": 6.80388839e-2, "gamma": 1.0e-3},
}


def regression_input(pred, feature):
    pred = np.asarray(pred, dtype=np.float32)
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim == 1:
        feature = feature.reshape(1, -1)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    return np.concatenate([pred, feature], axis=1)


class OffsetCalibrator:
    def fit(self, pred, y):
        self.offset = np.mean(np.asarray(y) - np.asarray(pred), axis=0)
        return self

    def predict(self, pred, feature=None):
        del feature
        return np.asarray(pred) + self.offset


class AffineCalibrator:
    def fit(self, pred, y):
        pred = np.asarray(pred, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if len(pred) < 3:
            self.fallback = OffsetCalibrator().fit(pred, y)
            self.matrix = None
            return self
        src = np.hstack([pred, np.ones((pred.shape[0], 1), dtype=np.float32)])
        matrix, _, _, _ = np.linalg.lstsq(src, y, rcond=None)
        self.matrix = matrix.astype(np.float32)
        self.fallback = None
        return self

    def predict(self, pred, feature=None):
        del feature
        pred = np.asarray(pred, dtype=np.float32)
        if self.matrix is None:
            return self.fallback.predict(pred)
        src = np.hstack([pred, np.ones((pred.shape[0], 1), dtype=np.float32)])
        return src @ self.matrix


class SklearnCalibrator:
    def __init__(self, mode="ridge"):
        self.mode = mode.lower()

    def _make_model(self, axis):
        try:
            from sklearn import linear_model
            from sklearn.svm import SVR
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for ridge/svr/mvo-svr calibration") from exc

        if self.mode == "ridge":
            return linear_model.Ridge(alpha=0.1)
        if self.mode == "svr":
            return SVR(kernel="rbf", C=10.0, gamma=5.0e-3, epsilon=0.1)
        if self.mode in {"mvo-svr", "mvosvr"}:
            params = MVO_SVR_PARAMS[axis]
            return SVR(kernel="rbf", **params)
        raise ValueError(f"Unsupported sklearn calibration mode: {self.mode}")

    def fit(self, pred, feature, y):
        x = regression_input(pred, feature)
        y = np.asarray(y, dtype=np.float32)
        if len(x) < 2:
            self.fallback = OffsetCalibrator().fit(pred, y)
            self.models = None
            return self
        self.models = [self._make_model("x"), self._make_model("y")]
        self.models[0].fit(x, y[:, 0])
        self.models[1].fit(x, y[:, 1])
        self.fallback = None
        return self

    def predict(self, pred, feature):
        if self.models is None:
            return self.fallback.predict(pred)
        x = regression_input(pred, feature)
        return np.stack([self.models[0].predict(x), self.models[1].predict(x)], axis=1).astype(np.float32)


def build_calibrator(mode):
    mode = (mode or "none").lower()
    if mode in {"none", "raw"}:
        return None
    if mode == "offset":
        return OffsetCalibrator()
    if mode == "affine":
        return AffineCalibrator()
    if mode in {"ridge", "svr", "mvo-svr", "mvosvr"}:
        return SklearnCalibrator(mode)
    raise ValueError(f"Unsupported calibration mode: {mode}")


def normalize_support_tasks(support_tasks):
    if support_tasks is None:
        return None
    if isinstance(support_tasks, str):
        values = support_tasks.replace(";", ",").split(",")
    else:
        values = support_tasks
    normalized = {str(value).strip() for value in values if str(value).strip()}
    return normalized or None


def select_support_indices(records, points=9, strategy="first_unique", support_tasks=None):
    if points <= 0:
        return np.zeros(len(records), dtype=bool)
    support_task_filter = normalize_support_tasks(support_tasks)
    grouped = {}
    for idx, record in enumerate(records):
        if support_task_filter is not None and str(record.get("task", "")) not in support_task_filter:
            continue
        dot_num = record.get("dotNum", -1)
        if dot_num >= 0:
            key = (dot_num, record.get("orientation", -1))
        else:
            key = (round(record["label_x"], 6), round(record["label_y"], 6), record.get("orientation", -1))
        grouped.setdefault(key, []).append(idx)

    keys = list(grouped)
    if strategy == "random_unique":
        rng = np.random.default_rng(0)
        rng.shuffle(keys)
    else:
        keys = sorted(keys, key=lambda key: grouped[key][0])
    chosen = set()
    for key in keys[:points]:
        chosen.update(grouped[key])
    return np.array([idx in chosen for idx in range(len(records))], dtype=bool)
