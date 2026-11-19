import os
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from leap_features import (
    DEFAULT_SCALE,
    DEFAULT_TRAIN_HAND,
    FEATURE_DIM,
    N_JOINTS,
    fill_missing,
    normalize_batch,
    preprocess_tag,
)


class LeapDatasetLoader:

    FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    JOINTS = ['TipPosition', 'dipPosition', 'pipPosition', 'mcpPosition']
    _EXTRACT_MARKER = '.leap_extracted'

    def __init__(
        self,
        folder_path: str = './',
        zip_path: str | None = None,
        extract_dir: str | None = None,
        cache_path: str | None = 'leap_dataset_cache.npz',
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        scale: str = DEFAULT_SCALE,
        train_hand: str = DEFAULT_TRAIN_HAND,
    ):
        # folder_path    - folder containing the (already extracted) dataset.
        # zip_path       - ZIP to extract first (into extract_dir).
        # extract_dir    - destination folder for ZIP extraction (defaults to folder_path).
        # cache_path     - .npz file path to skip re-parsing XMLs on future runs.
        # test_size      - fraction used for the test set.
        # random_state   - seed for reproducible splits.
        # stratify       - only used by the legacy frame-level split_data().
        # scale          - scale reference, see leap_features ("middle_mcp" | "max_dist").
        # train_hand     - "right" | "left". Frames from the other hand are mirrored to match.

        self.folder_path = folder_path
        self.zip_path = zip_path
        self.extract_dir = extract_dir if extract_dir else folder_path
        self.cache_path = cache_path
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.scale = scale
        self.train_hand = train_hand

        # Full dataset (before splitting)
        self.X_all = np.array([])
        self.y_all = np.array([])
        self.subjects_all = np.array([])

        # After splitting
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.subjects_train = np.array([])

        self.X_val = np.array([])
        self.y_val = np.array([])
        self.subjects_val = np.array([])

        self.X_test = np.array([])
        self.y_test = np.array([])
        self.subjects_test = np.array([])

        self.label_encoder = LabelEncoder()
        self.is_loaded = False
        self.is_split = False

        self.n_missing_joint_frames = 0

    # ------------------------------------------------------------------ cache
    @property
    def tag(self) -> str:
        return preprocess_tag(self.scale, self.train_hand)

    def _save_cache(self) -> None:
        """Saves normalized arrays, encoder classes and the preprocessing tag."""
        if not self.cache_path:
            return
        cache_dir = os.path.dirname(os.path.abspath(self.cache_path))
        os.makedirs(cache_dir, exist_ok=True)

        classes = (
            self.label_encoder.classes_
            if hasattr(self.label_encoder, 'classes_')
            else np.array([])
        )

        np.savez(
            self.cache_path,
            X_all=self.X_all,
            y_all=self.y_all,
            subjects_all=self.subjects_all,
            classes=classes,
            preprocess_tag=np.array(self.tag),
        )
        print(f"Saved parsed dataset cache to {self.cache_path}")

    def _load_cache(self) -> bool:
        """
        Loads the .npz cache. Returns False (and loads nothing) when the cache was
        built with a different preprocessing version/scale/hand, so stale features
        can never be used silently.
        """
        with np.load(self.cache_path, allow_pickle=False) as data:
            cached_tag = str(data['preprocess_tag']) if 'preprocess_tag' in data.files else None
            if cached_tag != self.tag:
                print(f"Cache preprocessing '{cached_tag}' != current '{self.tag}'. "
                      f"Rebuilding from XML.")
                return False

            self.X_all = data['X_all']
            self.y_all = data['y_all']
            self.subjects_all = data['subjects_all']
            if len(data['classes']) > 0:
                self.label_encoder.classes_ = data['classes']
        return True

    # ------------------------------------------------------------------ unzip
    def unzip_dataset(self) -> None:
        """Extracts the dataset ZIP once (tracked with a marker file in extract_dir)."""
        marker = os.path.join(self.extract_dir, self._EXTRACT_MARKER)
        if os.path.exists(marker):
            print("Dataset already unzipped.")
            return
        print(f"Unzipping dataset to {self.extract_dir}...")
        os.makedirs(self.extract_dir, exist_ok=True)
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_dir)
        with open(marker, 'w') as f:
            f.write(os.path.basename(str(self.zip_path)))
        print("Unzipping complete!")

    # ---------------------------------------------------------------- parsing
    @staticmethod
    def _read_vec3(node) -> list[float] | None:
        """Reads a 3-vector from an OpenCV-matrix node (<data>x y z</data>)."""
        if node is None:
            return None
        data = node.find('data')
        if data is None or not data.text:
            return None
        try:
            vals = [float(v) for v in data.text.split()]  # whitespace-separated
        except ValueError:
            return None
        return vals if len(vals) == 3 else None

    def parse_xml_file_with_hand(self, xml_path: str) -> tuple[np.ndarray, str] | None:
        """
        Parses one XML file into (raw 63-vector, "right"|"left"), or None.
        Missing joints are filled with the palm center (-> 0 after centering).
        Values are RAW (not normalized); normalization happens once, in batch.
        """
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            return None

        side = 'right'
        hand = root.find('.//Hands/Right')
        if hand is None:
            hand = root.find('.//Hands/Left')
            side = 'left'
        if hand is None:
            return None

        center = self._read_vec3(hand.find('Center'))
        if center is None:
            return None

        joints = np.full((N_JOINTS - 1, 3), np.nan)
        k = 0
        for finger in self.FINGERS:
            finger_node = hand.find(f'.//Fingers/{finger}')
            for joint in self.JOINTS:
                vec = self._read_vec3(finger_node.find(joint)) if finger_node is not None else None
                if vec is not None:
                    joints[k] = vec
                k += 1

        center_arr = np.asarray(center, dtype=np.float64)
        if np.isnan(joints).any():
            self.n_missing_joint_frames += 1
        joints = fill_missing(joints, center_arr)

        raw = np.concatenate([center_arr, joints.reshape(-1)]).astype(np.float32)
        if raw.size != FEATURE_DIM:
            return None
        return raw, side

    def parse_xml_file(self, xml_path: str) -> np.ndarray | None:
        """Backwards-compatible: raw 63-vector or None."""
        out = self.parse_xml_file_with_hand(xml_path)
        return None if out is None else out[0]

    # ---------------------------------------------------------------- loading
    def load_data(self, force_reload: bool = False) -> None:
        if self.cache_path and os.path.exists(self.cache_path) and not force_reload:
            print(f"Cache found at {self.cache_path}.")
            if self._load_cache():
                print("Skipping XML parsing.")
                self.is_loaded = True
                self.is_split = False
                return

        if self.zip_path is not None:
            self.unzip_dataset()
        else:
            print(f"Using existing dataset folder: {self.extract_dir}")

        X_list, y_list, subject_list, side_list = [], [], [], []
        self.n_missing_joint_frames = 0

        print("Reading directory and extracting features...")

        for root, dirs, files in os.walk(self.extract_dir):
            dirs[:] = [d for d in dirs if d != '__pycache__' and not d.startswith('.')]

            xml_files = [f for f in files if f.endswith('.xml')]
            if not xml_files:
                continue

            parts = [p for p in root.replace('\\', '/').split('/') if p]

            skel_idx = -1
            for i, part in enumerate(parts):
                if part.lower() == 'skeletal':
                    skel_idx = i
                    break

            if skel_idx == -1 or len(parts) < skel_idx + 4:
                continue

            subject_id = parts[skel_idx + 1]
            split_folder = parts[skel_idx + 2].lower()
            pose_label = parts[skel_idx + 3]

            # ignore test_gesture
            if 'train' not in split_folder:
                continue

            for xml_filename in xml_files:
                parsed = self.parse_xml_file_with_hand(os.path.join(root, xml_filename))
                if parsed is None:
                    continue
                features, side = parsed
                X_list.append(features)
                y_list.append(pose_label)
                subject_list.append(subject_id)
                side_list.append(side)

        X = np.array(X_list)
        y = np.array(y_list)
        subjects = np.array(subject_list)

        if len(X) > 0:
            mirror = np.array([s != self.train_hand for s in side_list])
            print(f"Parsed {len(X)} frames | mirrored (non-{self.train_hand}) hand: "
                  f"{int(mirror.sum())} | frames with missing joints: "
                  f"{self.n_missing_joint_frames}")
            X = normalize_batch(X, mirror=mirror, scale=self.scale)
            y = self.label_encoder.fit_transform(y)

        self.X_all, self.y_all, self.subjects_all = X, y, subjects
        self.is_loaded = True
        self.is_split = False

        if len(self.X_all) > 0:
            self._save_cache()

    # --------------------------------------------------------------- splitting
    @staticmethod
    def group_split_indices(groups, test_size: float, seed: int):
        """Index split that keeps every group (subject) entirely on one side."""
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(gss.split(np.zeros(len(groups)), groups=groups))

    def get_grouped_split(
        self,
        val_size: float = 0.2,
        test_size: float | None = None,
        random_state: int | None = None,
    ):
        """
        Subject-wise train / val / test split (no subject appears in two sets).
        Returns X_train, y_train, X_val, y_val, X_test, y_test.
        """
        if not self.is_loaded:
            self.load_data()
        if len(self.X_all) == 0:
            raise RuntimeError("No data to split (dataset is empty).")

        test_size = self.test_size if test_size is None else test_size
        seed = self.random_state if random_state is None else random_state

        n_subjects = len(np.unique(self.subjects_all))
        if n_subjects < 3:
            raise ValueError(f"Need at least 3 subjects for a subject-wise "
                             f"train/val/test split, found {n_subjects}.")

        trval_idx, test_idx = self.group_split_indices(self.subjects_all, test_size, seed)
        rel_tr, rel_val = self.group_split_indices(self.subjects_all[trval_idx], val_size, seed)
        tr_idx, val_idx = trval_idx[rel_tr], trval_idx[rel_val]

        self.X_train, self.y_train, self.subjects_train = (
            self.X_all[tr_idx], self.y_all[tr_idx], self.subjects_all[tr_idx])
        self.X_val, self.y_val, self.subjects_val = (
            self.X_all[val_idx], self.y_all[val_idx], self.subjects_all[val_idx])
        self.X_test, self.y_test, self.subjects_test = (
            self.X_all[test_idx], self.y_all[test_idx], self.subjects_all[test_idx])
        self.is_split = True

        n_classes = len(self.get_classes()) or int(self.y_all.max()) + 1
        for name, labels in [("train", self.y_train), ("val", self.y_val), ("test", self.y_test)]:
            missing = sorted(set(range(n_classes)) - set(np.unique(labels).tolist()))
            if missing:
                print(f"WARNING: {name} split is missing classes {missing}. "
                      f"Try another random_state or use leave-one-subject-out.")

        return (self.X_train, self.y_train, self.X_val, self.y_val,
                self.X_test, self.y_test)

    def split_data(
        self,
        test_size: float | None = None,
        random_state: int | None = None,
        stratify: bool | None = None,
    ) -> None:
        """
        LEGACY frame-level split. Frames from the same subject/recording end up in
        both train and test, which inflates accuracy. Prefer get_grouped_split().
        """
        if not self.is_loaded:
            self.load_data()
        if len(self.X_all) == 0:
            print("No data to split (dataset is empty).")
            return

        test_size = self.test_size if test_size is None else test_size
        random_state = self.random_state if random_state is None else random_state
        stratify = self.stratify if stratify is None else stratify

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.subjects_train,
            self.subjects_test,
        ) = train_test_split(
            self.X_all,
            self.y_all,
            self.subjects_all,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_all if stratify else None,
        )
        self.is_split = True

    # -------------------------------------------------------------- utilities
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Compatibility wrapper around the shared normalization (no mirroring)."""
        return normalize_batch(X, scale=self.scale)

    def get_data(
        self, force_reload: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # LEGACY: returns X_train, y_train, X_test, y_test from the frame-level split.
        if not self.is_loaded or force_reload:
            self.load_data(force_reload=force_reload)
        if not self.is_split:
            self.split_data()
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_classes(self) -> list[str]:
        # Returns the mapped list of pose class names
        if hasattr(self.label_encoder, 'classes_'):
            return [str(c) for c in self.label_encoder.classes_]
        return []

    def print_summary(self) -> None:
        print("\nDataset Summary:")
        print(f"Preprocessing : {self.tag}")
        print(f"X_all shape   : {self.X_all.shape}")
        print(f"Subjects      : {len(np.unique(self.subjects_all))}")
        print(f"X_train shape : {self.X_train.shape}")
        print(f"X_val shape   : {self.X_val.shape}")
        print(f"X_test shape  : {self.X_test.shape}")
        classes = self.get_classes()
        if classes:
            print(f"Pose Classes  : {classes}")
        else:
            print("Pose Classes  : No classes found (Dataset is empty)")
