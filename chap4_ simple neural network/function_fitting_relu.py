import os
import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def target_function(x: np.ndarray) -> np.ndarray:
    # A smooth but nonlinear target function
    return np.sin(2.0 * x) + 0.3 * (x ** 2) - 0.5 * np.cos(5.0 * x)


def build_dataset(n_train: int = 2000, n_test: int = 400):
    x_train = np.random.uniform(-3.0, 3.0, size=(n_train, 1)).astype(np.float32)
    x_test = np.linspace(-3.0, 3.0, n_test, dtype=np.float32).reshape(-1, 1)

    noise = np.random.normal(loc=0.0, scale=0.05, size=(n_train, 1)).astype(np.float32)
    y_train = target_function(x_train).astype(np.float32) + noise
    y_test = target_function(x_test).astype(np.float32)
    return (x_train, y_train), (x_test, y_test)


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main() -> None:
    set_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")

    (x_train, y_train), (x_test, y_test) = build_dataset()
    model = build_model()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=64,
        verbose=0,
    )

    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, verbose=0)

    mse = float(np.mean((y_pred - y_test) ** 2))
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {float(test_mae):.6f}")
    print(f"Final Train Loss: {float(history.history['loss'][-1]):.6f}")
    print(f"Final Val Loss: {float(history.history['val_loss'][-1]):.6f}")

    try:
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(x_test[:, 0], y_test[:, 0], label="Ground Truth", linewidth=2)
        plt.plot(x_test[:, 0], y_pred[:, 0], label="Model Prediction", linewidth=2)
        plt.scatter(x_train[:300, 0], y_train[:300, 0], s=8, alpha=0.25, label="Train Samples")
        plt.title("ReLU Network Function Fitting")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "function_fitting_result.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "function_fitting_loss_curve.png"), dpi=160)
        plt.close()

        print("Saved plots to", output_dir)
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
