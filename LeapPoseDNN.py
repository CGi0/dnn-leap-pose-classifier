import numpy as np
import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


class LeapPoseDNN:
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 63,
        hidden_units: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.seed = seed

        keras.utils.set_random_seed(seed)
        self.model: keras.Model = self._build_model()
        self.history: keras.callbacks.History | None = None

    def _build_model(self) -> keras.Model:
        # Builds and compiles the network."""
        inputs = keras.Input(shape=(self.input_dim,), name="hand_features")
        x = inputs

        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units, use_bias=False, name=f"dense_{i}")(x)
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
            x = layers.Activation("relu", name=f"relu_{i}")(x)
            x = layers.Dropout(self.dropout, name=f"drop_{i}")(x)

        outputs = layers.Dense(
            self.num_classes, activation="softmax", name="pose_probs"
        )(x)

        model = keras.Model(inputs, outputs, name="leap_pose_dnn")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def summary(self) -> None:
        self.model.summary()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 128,
        balance_classes: bool = False,
        patience: int = 10,
        checkpoint_path: str | None = None,
        verbose: int = 1,
    ) -> keras.callbacks.History:

        # Trains the model. 
        # If validation data is given, early stopping and learning-rate reduction monitor validation loss and the best weights are restored at the end.
    
        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        monitor = "val_loss" if has_val else "loss"

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=monitor, patience=patience, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=0.5, patience=max(2, patience // 3), min_lr=1e-6
            ),
        ]
        if checkpoint_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    checkpoint_path, monitor=monitor, save_best_only=True
                )
            )

        class_weight = None
        if balance_classes:
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if has_val else None,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose,
        )
        return self.history

    def get_history(self) -> dict[str, list[float]]:
        # Returns the full Keras training history as a plain dict, e.g.
        # {"loss": [...], "val_loss": [...], "accuracy": [...], "val_accuracy": [...]}
        if self.history is None:
            raise RuntimeError("No history available. Call train() first.")
        return {k: [float(v) for v in vals] for k, vals in self.history.history.items()}

    def get_loss_history(self) -> dict[str, list[float]]:
        # Returns only the per-epoch training loss and (if validation data was
        # used) the validation loss: {"loss": [...], "val_loss": [...]}
        history = self.get_history()
        return {k: history[k] for k in ("loss", "val_loss") if k in history}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Returns class probabilities, shape (N, num_classes)
        return self.model.predict(X, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Returns predicted integer class ids, shape (N,)
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: list[str] | None = None,
        print_report: bool = True,
    ) -> dict:
        # Evaluates on a dataset. Returns loss, accuracy, confusion matrix and
        # the per-class report (precision / recall / f1) as a dict.
        
        loss, acc = self.model.evaluate(X, y, verbose=0)
        y_pred = self.predict(X)

        labels = np.arange(self.num_classes)
        names = [str(n) for n in class_names] if class_names else None
        report = classification_report(
            y, y_pred, labels=labels, target_names=names,
            zero_division=0, output_dict=True,
        )
        cm = confusion_matrix(y, y_pred, labels=labels)

        if print_report:
            print(f"\nLoss: {loss:.4f} | Accuracy: {acc:.4f}\n")
            print(
                classification_report(
                    y, y_pred, labels=labels, target_names=names, zero_division=0
                )
            )

        return {"loss": loss, "accuracy": acc, "report": report, "confusion_matrix": cm}

    def save(self, path: str = "leap_pose_dnn.keras") -> None:
        self.model.save(path)
        
"""
    @classmethod
    def load(cls, path: str) -> "LeapPoseDNN":
        # Loads a saved model into a LeapPoseDNN wrapper.
        model = keras.models.load_model(path)
        obj = cls.__new__(cls)
        obj.model = model
        obj.input_dim = model.input_shape[-1]
        obj.num_classes = model.output_shape[-1]
        obj.hidden_units = ()
        obj.dropout = 0.0
        obj.learning_rate = 1e-3
        obj.seed = 42
        obj.history = None
        return obj
"""