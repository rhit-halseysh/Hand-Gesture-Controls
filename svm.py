# Image Recognition project
# Ethan Shan, Kelvin Zhou, Spencer Halsey, Tyler Moore
# SVM Classifier Implementation
# AI utilized to write progress statements and testing code

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

TEST_SIZE = 0.2
RANDOM_STATE = 42

def rgbtolst(img_rgb: np.ndarray) -> np.ndarray:
    rgb = img_rgb.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    L = (R + G + B) / np.sqrt(3)
    S = (R - B) / np.sqrt(2)
    T = (R - 2*G + B) / np.sqrt(6)

    return np.stack([L, S, T], axis=-1)


def extract_features(img: np.ndarray) -> np.ndarray:
    H, W, _ = img.shape
    tile_h = H // 7
    tile_w = W // 7
    features = []

    for row in range(7):
        for col in range(7):
            tile = img[
                row * tile_h : (row + 1) * tile_h,
                col * tile_w : (col + 1) * tile_w,
                :
            ]
            for ch in range(3):
                features.append(tile[..., ch].mean())
                features.append(tile[..., ch].std())

    return np.array(features, dtype=np.float64)


def load_dataset(datadir: str) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    count = 0

    for gesture_name in sorted(os.listdir(datadir)):
        gesture_path = os.path.join(datadir, gesture_name)
        if not os.path.isdir(gesture_path):
            continue

        for fname in sorted(os.listdir(gesture_path)):
            if not fname.lower().endswith(".jpg"):
                continue

            img_path = os.path.join(gesture_path, fname)
            try:
                img_rgb = np.array(Image.open(img_path).convert("RGB"))
                lst = rgbtolst(img_rgb)
                vec = extract_features(lst)
                features.append(vec)
                labels.append(gesture_name)
                count += 1
                if count % 100 == 0:
                    print(f"  Processed {count} images …")
            except Exception as e:
                print(f"  [warn] skipping {img_path}: {e}")

    print(f"  Done — {count} images total.")
    return np.array(features), np.array(labels)


def main():
    if os.path.exists("features.npz"):
        print(f"[1/4] Cache found — loading features from 'features.npz' …")
        data = np.load("features.npz", allow_pickle=True)
        X, y_raw = data["X"], data["y"]
    else:
        print("[1/4] No cache found — extracting features from images …")
        X, y_raw = load_dataset("data")
        np.savez("features.npz", X=X, y=y_raw)
        print(f"       Saved features to 'features.npz' for next time.")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print("[2/4] Splitting into train and test sets …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"       Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    print("[3/4] Running Grid Search")

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ["rbf"],
        "gamma":  ["auto"],
    }

    grid_search = GridSearchCV(
        estimator=SVC(),
        param_grid=param_grid,
        cv=5,                  
        scoring="accuracy",
        n_jobs=-1,             
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\n       Best parameters : {grid_search.best_params_}")
    print(f"       Best CV accuracy: {grid_search.best_score_:.4f}")

    print("[4/4] Evaluating on test set …\n")
    y_pred = grid_search.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_
    ))


if __name__ == "__main__":
    main()