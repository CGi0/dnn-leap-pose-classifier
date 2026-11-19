import json
import numpy as np
import matplotlib.pyplot as plt

from LeapDatasetLoader import LeapDatasetLoader
from LeapPoseDNN import LeapPoseDNN
from LeapDatasetFeatures import FEATURE_DIM, PREPROCESS_VERSION
from LeapPoseTuner import tune, best_config

SEED = 42
MODEL_PATH = "LeapPoseDNN.keras"
CONFIG_PATH = "LeapPoseConfig.json"


def loso_eval(loader, model_kwargs, train_kwargs, num_classes) -> dict:
    """
    Leave-one-subject-out: train on all other subjects (a slice of them as
    validation), test on the held-out one. Slow, but the most honest estimate of
    how the model does on a new person. Useful when there are few subjects.
    """
    X, y, g = loader.X_all, loader.y_all, loader.subjects_all
    accs = {}
    for s in np.unique(g):
        held = g == s
        rest = np.where(~held)[0]
        rel_tr, rel_val = loader.group_split_indices(g[rest], 0.15, SEED)
        tr, val = rest[rel_tr], rest[rel_val]

        net = LeapPoseDNN(num_classes=num_classes, **model_kwargs)
        net.train(X[tr], y[tr], X_val=X[val], y_val=y[val], epochs=100,
                  verbose=0, **train_kwargs)
        _, acc = net.model.evaluate(X[held], y[held], verbose=0)
        accs[str(s)] = float(acc)
        print(f"  held-out subject {s}: acc {acc:.3f}")

    vals = np.array(list(accs.values()))
    print(f"LOSO accuracy: mean {vals.mean():.3f} | std {vals.std():.3f} | "
          f"min {vals.min():.3f}")
    return accs

def load_hparams(path: str = CONFIG_PATH) -> tuple[dict, dict]:
    with open(path) as f:
        cfg = json.load(f)
    return cfg["model_kwargs"], cfg["train_kwargs"]


def main(run_tuning: bool = True, stage: int = 2, run_loso: bool = False, reuse_config: bool = False) -> None:
    # --- Load + subject-wise train / val / test ---------------------------
    loader = LeapDatasetLoader(folder_path='./', cache_path='./leap_dataset_cache.npz')
    loader.load_data()
    X_tr, y_tr, X_val, y_val, X_test, y_test = loader.get_grouped_split(
        val_size=0.2, test_size=0.2, random_state=SEED
    )
    loader.print_summary()
    class_names = loader.get_classes()
    num_classes = len(class_names)

    # --- Tune (validation set only) or use the baseline config ------------
    if run_tuning:
        tuner = tune(X_tr, y_tr, X_val, y_val, num_classes, stage=stage)
        model_kwargs, train_kwargs = best_config(tuner)
    elif reuse_config:
        model_kwargs, train_kwargs = load_hparams()
    else:
        model_kwargs, train_kwargs = {}, {"balance_classes": True}

    # --- Final training, then a single look at the test set ---------------
    dnn = LeapPoseDNN(num_classes=num_classes, **model_kwargs)
    hist = dnn.train(X_tr, y_tr, X_val=X_val, y_val=y_val, epochs=100, **train_kwargs)
    dnn.evaluate(X_test, y_test, class_names=class_names, print_report=True)

    if run_loso:
        print("\nLeave-one-subject-out evaluation:")
        loso_eval(loader, model_kwargs, train_kwargs, num_classes)

    # --- Save model + config the inference script validates against -------
    dnn.save(MODEL_PATH)
    np.save("pose_classes.npy", np.array(class_names))
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "class_names": class_names,
            "feature_dim": FEATURE_DIM,
            "preprocess_version": PREPROCESS_VERSION,
            "preprocess_tag": loader.tag,
            "scale": loader.scale,
            "train_hand": loader.train_hand,
            "model_kwargs": model_kwargs,
            "train_kwargs": train_kwargs,
        }, f, indent=2, default=str)
    print(f"Saved {MODEL_PATH} and {CONFIG_PATH}")

    # --- Plot: train vs val on shared axes --------------------------------
    h = hist.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(h["loss"], label="Training loss")
    ax1.plot(h["val_loss"], label="Validation loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax2.plot(h["accuracy"], label="Training accuracy")
    ax2.plot(h["val_accuracy"], label="Validation accuracy")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for ax in (ax1, ax2):
        ax.grid(True)
        ax.legend()
    plt.show()


if __name__ == "__main__":
    # First run: main(run_tuning=False) for a baseline on the subject-wise split.
    # Then main(run_tuning=True, stage=1), and later stage=2.
    main(run_tuning=False, reuse_config=True)
    # main(run_tuning=True, stage=1)
