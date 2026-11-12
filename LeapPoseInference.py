import time
import threading
from collections import deque, Counter
 
import numpy as np
import tensorflow as tf
import leap
 
MODEL_PATH = "leap_pose_dnn.keras"          # <- your saved model
CONF_THRESHOLD = 0.80
INFER_HZ = 30
SMOOTH_N = 7                                 # majority vote over last N predictions
TRAIN_HAND = "right"                         # hand the dataset was recorded with; the other hand is mirrored
 
# LabelEncoder sorts alphabetically; this is the order from your dataset summary.
CLASSES = ['01_palm', '02_l', '04_fist_moved', '05_down', '06_index', '07_ok',
           '08_palm_m', '09_c', '11_heavy', '12_hang', '13_two', '14_three',
           '15_four', '16_five', '17_palm_u', '18_up']
 
 
def _xyz(v) -> list:
    return [v.x, v.y, v.z]
 
 
def points_to_features(center, joints, mirror: bool = False) -> np.ndarray:
    """
    center: (3,) palm center.  joints: (20, 3) in loader order.
    Mirrors LeapDatasetLoader._normalize. mirror=True flips x (left <-> right hand).
    """
    pts = np.vstack([np.asarray(center, dtype=np.float64).reshape(1, 3),
                     np.asarray(joints, dtype=np.float64).reshape(20, 3)])
    pts = pts - pts[0]                               # palm center -> (0, 0, 0)
    if mirror:
        pts[:, 0] *= -1.0                            # reflect across the YZ plane
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist == 0:
        max_dist = 1.0
    pts = pts / max_dist
    return pts.reshape(-1).astype(np.float32)        # (63,)
 
 
def hand_features(hand) -> np.ndarray:
    center = _xyz(hand.palm.position)
    joints = []
    for digit in hand.digits:                        # thumb, index, middle, ring, pinky
        meta, prox, inter, dist = digit.bones        # metacarpal, proximal, intermediate, distal
        joints.append(_xyz(dist.next_joint))         # TipPosition
        joints.append(_xyz(dist.prev_joint))         # dipPosition
        joints.append(_xyz(inter.prev_joint))        # pipPosition
        joints.append(_xyz(prox.prev_joint))         # mcpPosition
    return points_to_features(center, joints, mirror=needs_mirror(hand))
 
 
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
 
 
class PoseListener(leap.Listener):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._latest = None
 
    def on_connection_event(self, event):
        print("Connected to tracking service")
 
    def on_device_event(self, event):
        try:
            with event.device.open():
                print("Device:", event.device.get_info().serial)
        except leap.LeapCannotOpenDeviceError:
            pass
 
    def on_tracking_event(self, event):
        feats = hand_features(pick_hand(event.hands)) if event.hands else None
        with self._lock:
            self._latest = feats
 
    def get(self):
        with self._lock:
            return self._latest
 
 
def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    n_in = model.input_shape[-1]
    assert n_in == 63, f"model expects {n_in} features, loader produces 63"
    model(np.zeros((1, 63), dtype=np.float32), training=False)   # warm-up
 
    listener = PoseListener()
    connection = leap.Connection()
    connection.add_listener(listener)
    history = deque(maxlen=SMOOTH_N)
 
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
                    probs = model(feats[None, :], training=False).numpy()[0]
                    k = int(np.argmax(probs))
                    history.append(k if probs[k] >= CONF_THRESHOLD else -1)
                    vote, _ = Counter(history).most_common(1)[0]
                    label = CLASSES[vote] if vote >= 0 else "none"
                    print(f"\r{label:<16} {probs[k]:.2f}", end="")
                time.sleep(period)
        except KeyboardInterrupt:
            pass
 
 
if __name__ == "__main__":
    main()
