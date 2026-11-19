"""
Hyperparameter tuning for LeapPoseDNN with KerasTuner.

Stage 1: depth, per-layer widths, dropout, learning rate, batch size, class balance.
Stage 2: adds width pattern, L2, activation, input noise, BatchNorm toggle, optimizer.

Run stage 1 first, look at which parameters matter (tuner.results_summary()), then
move to stage 2. Tune on the subject-wise TRAIN and VAL sets only; never the test set.
"""
import numpy as np
import keras
import keras_tuner as kt
from sklearn.utils.class_weight import compute_class_weight

from LeapPoseDNN import LeapPoseDNN
from LeapDatasetFeatures import FEATURE_DIM


def sample_config(hp: kt.HyperParameters, stage: int = 2) -> tuple[dict, dict]:
    """
    Samples (or, on a finished trial's hp, reads back) the hyperparameters.
    Returns (model_kwargs for LeapPoseDNN, fit_kwargs with batch_size/balance_classes).
    Shared by build()/fit() and best_to_wrapper() so they can never disagree.
    """
    n_layers = hp.Int("n_layers", 1, 4)

    if stage >= 2:
        base = hp.Choice("base_units", [64, 128, 256, 512])
        shape = hp.Choice("width_shape", ["constant", "funnel", "expand"])
        hidden = []
        for i in range(n_layers):
            if shape == "constant":
                hidden.append(base)
            elif shape == "funnel":
                hidden.append(max(32, base // (2 ** i)))
            else:  # expand
                hidden.append(min(512, base * (2 ** i)))
        hidden_units = tuple(hidden)
    else:
        hidden_units = tuple(
            hp.Int(f"units_{i}", min_value=32, max_value=512, step=32)
            for i in range(n_layers)
        )

    model_kwargs = dict(
        hidden_units=hidden_units,
        dropout=hp.Float("dropout", 0.0, 0.5, step=0.1),
        learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log"),
    )

    if stage >= 2:
        model_kwargs.update(
            l2=hp.Float("l2", 1e-6, 1e-2, sampling="log"),
            activation=hp.Choice("activation", ["relu", "gelu", "silu"]),
            input_noise=hp.Choice("input_noise", [0.0, 0.01, 0.02, 0.05, 0.1]),
            use_batchnorm=hp.Boolean("use_batchnorm", default=True),
            optimizer=hp.Choice("optimizer", ["adam", "adamw"]),
        )

    fit_kwargs = dict(
        batch_size=hp.Choice("batch_size", [32, 64, 128, 256]),
        balance_classes=hp.Boolean("balance_classes"),
    )
    return model_kwargs, fit_kwargs


class LeapPoseHyperModel(kt.HyperModel):
    def __init__(self, num_classes: int, input_dim: int = FEATURE_DIM, stage: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.stage = stage

    def build(self, hp: kt.HyperParameters) -> keras.Model:
        model_kwargs, _ = sample_config(hp, self.stage)
        wrapper = LeapPoseDNN(
            num_classes=self.num_classes, input_dim=self.input_dim, **model_kwargs
        )
        return wrapper.model  # already compiled by LeapPoseDNN

    def fit(self, hp, model, x, y, **kwargs):
        # Tune things that live in fit() rather than build().
        _, fit_kwargs = sample_config(hp, self.stage)

        class_weight = None
        if fit_kwargs["balance_classes"]:
            classes = np.unique(y)
            w = compute_class_weight("balanced", classes=classes, y=y)
            class_weight = {int(c): float(v) for c, v in zip(classes, w)}

        return model.fit(
            x, y,
            batch_size=fit_kwargs["batch_size"],
            class_weight=class_weight,
            **kwargs,
        )


def tune(
    X_train, y_train, X_val, y_val,
    num_classes: int,
    input_dim: int = FEATURE_DIM,
    stage: int = 2,
    max_trials: int = 40,
    executions_per_trial: int = 1,   # use 2-3 if trial scores look noisy
    epochs: int = 100,
    project_name: str | None = None,
    overwrite: bool = False,
) -> kt.Tuner:
    tuner = kt.BayesianOptimization(
        hypermodel=LeapPoseHyperModel(num_classes, input_dim, stage),
        objective="val_loss",        # or "val_accuracy"
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="tuner_runs",
        project_name=project_name or f"leap_pose_s{stage}",
        overwrite=overwrite,         # False resumes an interrupted search
        seed=42,
    )

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=3, min_lr=1e-6),
        ],
        verbose=2,
    )
    tuner.results_summary(num_trials=5)
    return tuner


def best_config(tuner: kt.Tuner) -> tuple[dict, dict]:
    """(model_kwargs, train_kwargs) of the best trial."""
    hp = tuner.get_best_hyperparameters(1)[0]
    return sample_config(hp, tuner.hypermodel.stage)


def best_to_wrapper(tuner: kt.Tuner, num_classes: int, input_dim: int = FEATURE_DIM):
    """Fresh LeapPoseDNN built from the best trial, plus kwargs for .train()."""
    model_kwargs, train_kwargs = best_config(tuner)
    wrapper = LeapPoseDNN(num_classes=num_classes, input_dim=input_dim, **model_kwargs)
    return wrapper, train_kwargs, model_kwargs
