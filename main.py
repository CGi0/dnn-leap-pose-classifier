import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LeapDatasetLoader import LeapDatasetLoader
from LeapPoseDNN import LeapPoseDNN


def main() -> None:
    # Initialization
    loader = LeapDatasetLoader(folder_path='./')
    loader.load_data() 
    loader.split_data(test_size=0.2)
    loader.print_summary()

    # Data Split
    X_train, y_train, X_test, y_test = loader.get_data()
    class_names = loader.get_classes()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )


    # Build DNN and training
    dnn = LeapPoseDNN(num_classes=len(class_names))
    dnn_hist = dnn.train(X_tr, y_tr, X_val=X_val, y_val=y_val, epochs=100, balance_classes=True)
    dnn.evaluate(X_test, y_test, class_names=class_names, print_report=False)
    dnn.save("leap_pose_dnn.keras")
    
    # Plotting
    train_loss = dnn_hist.history["loss"]
    val_loss = dnn_hist.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_loss, label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(val_loss, label="Validation loss", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss")
    ax2.grid(True)
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    main()