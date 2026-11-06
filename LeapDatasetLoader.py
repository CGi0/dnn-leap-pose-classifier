import os
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class LeapDatasetLoader:
    """
    Only the 'train_gesture' folders are read ('test_gesture' is ignored). The data is
    then split into train/test sets with sklearn's train_test_split.
    """

    FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    JOINTS = ['TipPosition', 'dipPosition', 'pipPosition', 'mcpPosition']

    def __init__(self, folder_path: str = './', zip_path: str | None = None,
                 extract_dir: str | None = None, test_size: float = 0.2,
                 random_state: int = 42, stratify: bool = True):

        # folder_path    -   folder containing the (already extracted) dataset.
        # zip_path       -   ZIP to extract first (into extract_dir).
        # test_size      -   fraction of frames used for the test set.
        # random_state   -   seed for a reproducible split.
        # stratify       -   keep the same pose proportions in train and test.
        
        self.zip_path = zip_path
        self.folder_path = folder_path
        self.extract_dir = extract_dir if extract_dir else folder_path
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

        # State containers (full dataset, before splitting)
        self.X_all = np.array([])
        self.y_all = np.array([])
        self.subjects_all = np.array([])

        # State containers (after splitting)
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.subjects_train = np.array([])

        self.X_test = np.array([])
        self.y_test = np.array([])
        self.subjects_test = np.array([])

        self.label_encoder = LabelEncoder()
        self.is_loaded = False
        self.is_split = False

    def unzip_dataset(self) -> None:
        # Extracts dataset ZIP file to destination folder
        if not os.path.exists(self.extract_dir):
            print(f"Unzipping dataset to {self.extract_dir}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
            print("Unzipping complete!")
        else:
            print("Dataset already unzipped.")

    @staticmethod
    def _read_vec3(node) -> list[float] | None:
        # Reads a 3-vector from an OpenCV-matrix node (<data>x y z</data>)
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

    def parse_xml_file(self, xml_path: str) -> np.ndarray | None:
        """
        Parses a single XML file and extracts a 63-element feature vector
        (palm center + 4 joints for each of 5 fingers). Supports Right and Left hands.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            return None

        hand = root.find('.//Hands/Right')
        if hand is None:
            hand = root.find('.//Hands/Left')
        if hand is None:
            return None

        features = []

        # 1. Palm center
        center = self._read_vec3(hand.find('Center'))
        if center is None:
            return None
        features.extend(center)

        # 2. Finger joints
        for finger in self.FINGERS:
            finger_node = hand.find(f'.//Fingers/{finger}')
            for joint in self.JOINTS:
                vec = self._read_vec3(finger_node.find(joint)) if finger_node is not None else None
                features.extend(vec if vec is not None else [0.0, 0.0, 0.0])

        if len(features) != 63:
            return None

        return np.array(features, dtype=np.float32)

    def load_data(self) -> None:
        if self.zip_path is not None:
            self.unzip_dataset()
        else:
            print("Using existing dataset folder.")

        X_list, y_list, subject_list = [], [], []

        print("Reading directory and extracting features...")

        for root, _, files in os.walk(self.extract_dir):
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
                features = self.parse_xml_file(os.path.join(root, xml_filename))
                if features is None:
                    continue
                X_list.append(features)
                y_list.append(pose_label)
                subject_list.append(subject_id)

        X = np.array(X_list)
        y = np.array(y_list)
        subjects = np.array(subject_list)

        if len(X) > 0:
            X = self._normalize(X)
            y = self.label_encoder.fit_transform(y)

        self.X_all, self.y_all, self.subjects_all = X, y, subjects
        self.is_loaded = True
        self.is_split = False

    def split_data(self, test_size: float | None = None, random_state: int | None = None,
                   stratify: bool | None = None) -> None:
        if not self.is_loaded:
            self.load_data()
        if len(self.X_all) == 0:
            print("No data to split (dataset is empty).")
            return

        test_size = self.test_size if test_size is None else test_size
        random_state = self.random_state if random_state is None else random_state
        stratify = self.stratify if stratify is None else stratify

        (self.X_train, self.X_test,
         self.y_train, self.y_test,
         self.subjects_train, self.subjects_test) = train_test_split(
            self.X_all, self.y_all, self.subjects_all,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_all if stratify else None,
        )
        self.is_split = True

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        # Translates coordinates to the palm center and scales by max joint distance
        palm_x = X[:, 0:1].copy()
        palm_y = X[:, 1:2].copy()
        palm_z = X[:, 2:3].copy()

        # 1. Translation: move palm center to (0,0,0)
        X[:, 3::3] -= palm_x
        X[:, 4::3] -= palm_y
        X[:, 5::3] -= palm_z
        X[:, 0:3] = 0.0

        # 2. Scale normalization
        max_dists = np.max(
            np.linalg.norm(X.reshape(-1, 21, 3), axis=2), axis=1, keepdims=True
        )
        max_dists[max_dists == 0] = 1.0
        X /= max_dists
        return X

    def get_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Returns X_train, y_train, X_test, y_test
        if not self.is_loaded:
            self.load_data()
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
        print(f"X_train shape : {self.X_train.shape}")
        print(f"y_train shape : {self.y_train.shape}")
        print(f"X_test shape  : {self.X_test.shape}")
        print(f"y_test shape  : {self.y_test.shape}")
        classes = self.get_classes()
        if classes:
            print(f"Pose Classes  : {classes}")
        else:
            print("Pose Classes  : No classes found (Dataset is empty)")