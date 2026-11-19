import json
import sys
import time
import threading
from collections import deque, Counter

import numpy as np
import keras
import leap

from leap_features import FEATURE_DIM, PREPROCESS_VERSION, normalize_points

MODEL_PATH = "leap_pose_dnn.keras"           # <- saved by main.py
CONFIG_PATH = "leap_pose_config.json"        # <- saved by main.py
CONF_THRESHOLD = 0.80                        # re-tune after retraining on the subject-wise split
INFER_HZ = 30
SMOOTH_N = 7                                 # majority vote over last N predictions
STALE_S = 0.2                                # tracking data older than this counts as "no hand"
DEBUG = "--debug" in sys.argv                # print live feature vectors about once a second


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    if cfg["preprocess_version"] != PREPROCESS_VERSION:
        raise RuntimeError(
            f"Model was trained with preprocessing v{cfg['preprocess_version']} but "
            f"leap_features.py is v{PREPROCESS_VERSION}. Retrain or restore the old module."
        )
    if cfg["feature_dim"] != FEATURE_DIM:
        raise RuntimeError(f"Model feature_dim {cfg['feature_dim']} != {FEATURE_DIM}")
    return cfg


CFG = load_config()
CLASSES = CFG["class_names"]
TRAIN_HAND = CFG["train_hand"]               # hand the dataset was recorded with
SCALE = CFG["scale"]


def _xyz(v) -> list:
    return [v.x, v.y, v.z]


def _train_hand_type():
    return leap.HandType.Right if TRAIN_HAND == "right" else leap.HandType.Left


def needs_mirror(hand) -> bool:
    """True when the live hand is the opposite of the hand the dataset was recorded with."""
    return hand.type != _train_hand_type()


def pick_hand(hands):
    """Prefer the hand matching the training hand; otherwise use any hand (it gets mirrored)."""
    for h in hands:
        if h.type == _train_hand_type():
            return h
    return hands[0]


def hand_features(hand) -> np.ndarray:
    center = _xyz(hand.palm.position)
    joints = []
    for digit in hand.digits:                        # thumb, index, middle, ring, pinky
        meta, prox, inter, dist = digit.bones        # metacarpal, proximal, intermediate, distal
        joints.append(_xyz(dist.next_joint))         # TipPosition
        joints.append(_xyz(dist.prev_joint))         # dipPosition
        joints.append(_xyz(inter.prev_joint))        # pipPosition
        joints.append(_xyz(prox.prev_joint))         # mcpPosition
    return normalize_points(center, joints, mirror=needs_mirror(hand), scale=SCALE)


class PoseListener(leap.Listener):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._latest = None                          # (monotonic timestamp, features) or None

    def on_connection_event(self, event):
        print("Connected to tracking service")

    def on_device_event(self, event):
        try:
            with event.device.open():
                print("Device:", event.device.get_info().serial)
        except leap.LeapCannotOpenDeviceError:
            pass

    def on_tracking_event(self, event):
        item = (time.monotonic(), hand_features(pick_hand(event.hands))) if event.hands else None
        with self._lock:
            self._latest = item

    def get(self):
        with self._lock:
            item = self._latest
        if item is None or time.monotonic() - item[0] > STALE_S:
            return None
        return item[1]


def main():
    model = keras.models.load_model(MODEL_PATH)
    n_in = model.input_shape[-1]
    assert n_in == FEATURE_DIM, f"model expects {n_in} features, pipeline produces {FEATURE_DIM}"
    model(np.zeros((1, FEATURE_DIM), dtype=np.float32), training=False)   # warm-up

    listener = PoseListener()
    connection = leap.Connection()
    connection.add_listener(listener)
    history = deque(maxlen=SMOOTH_N)                 # (class id or -1, probability)
    last_debug = 0.0

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        period = 1.0 / INFER_HZ
        try:
            while True:
                feats = listener.get()
                if feats is None:
                    history.clear()
                    print("\rno hand" + " " * 30, end="")
                else:
                    if DEBUG and time.monotonic() - last_debug > 1.0:
                        print("\nfeatures:", np.round(feats, 3).tolist())
                        last_debug = time.monotonic()

                    x = feats[None, :]
                    probs = keras.ops.convert_to_numpy(model(x, training=False))[0]
                    k = int(np.argmax(probs))
                    history.append((k if probs[k] >= CONF_THRESHOLD else -1, float(probs[k])))

                    vote, _ = Counter(v for v, _ in history).most_common(1)[0]
                    mean_p = float(np.mean([p for v, p in history if v == vote]))
                    label = CLASSES[vote] if vote >= 0 else "none"
                    print(f"\r{label:<16} {mean_p:.2f}", end="")
                time.sleep(period)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
