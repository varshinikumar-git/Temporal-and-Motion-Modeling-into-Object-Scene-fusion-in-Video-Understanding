import numpy as np
import pandas as pd
import os
import cv2

# === CONFIGURATION ===
BASE_DIR = r"F:\IMAGE ANALYSIS PROJECT"
OUT_DIR = os.path.join(BASE_DIR, "outputs")

# Input CSVs from previous days
FRAME_DIFF_CSV = os.path.join(OUT_DIR, "sample_frame_diff_stats.csv")
FLOW_CSV = os.path.join(OUT_DIR, "sample_flow_stats.csv")
TRACK_CSV = os.path.join(OUT_DIR, "sample_tracks_stats.csv")

# Outputs
OUT_DESCRIPTOR_NPY = os.path.join(OUT_DIR, "sample_temporal_descriptor.npy")
OUT_DESCRIPTOR_IMG = os.path.join(OUT_DIR, "sample_temporal_descriptor.png")
OUT_DESCRIPTOR_CSV = os.path.join(OUT_DIR, "sample_temporal_descriptor.csv")

# === 1. LOAD METRICS FROM PREVIOUS STEPS ===
def load_stats():
    fd = pd.read_csv(FRAME_DIFF_CSV)
    flow = pd.read_csv(FLOW_CSV)
    track = pd.read_csv(TRACK_CSV)

    # Flatten all numeric values into a single list
    combined = []
    for df in [fd, flow, track]:
        for col in df.columns:
            combined.append(float(df[col].values[0]))
    combined = np.array(combined, dtype=np.float32)
    print(f"Loaded {len(combined)} raw features from CSVs.")
    return combined

# === 2. NORMALIZE AND EXPAND DESCRIPTOR ===
def normalize_and_expand(features, target_dim=64):
    # Simple z-score normalization
    f_norm = (features - np.mean(features)) / (np.std(features) + 1e-6)

    # Repeat or pad to reach target dimension
    if len(f_norm) < target_dim:
        reps = int(np.ceil(target_dim / len(f_norm)))
        f_norm = np.tile(f_norm, reps)[:target_dim]
    elif len(f_norm) > target_dim:
        f_norm = f_norm[:target_dim]

    np.save(OUT_DESCRIPTOR_NPY, f_norm)
    pd.DataFrame(f_norm).T.to_csv(OUT_DESCRIPTOR_CSV, index=False)
    print(f"Saved temporal descriptor: {OUT_DESCRIPTOR_NPY}")
    return f_norm

# === 3. VISUALIZE DESCRIPTOR AS OPENCV IMAGE ===
def visualize_descriptor(descriptor):
    h, w = 180, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw bars
    n = len(descriptor)
    bar_w = int(w / n)
    min_val, max_val = np.min(descriptor), np.max(descriptor)

    # Normalize for visualization
    norm = (descriptor - min_val) / (max_val - min_val + 1e-8)

    for i, val in enumerate(norm):
        bar_h = int(val * (h - 40))
        color = (int(255 * val), int(255 * (1 - val)), 150)
        cv2.rectangle(img, (i * bar_w, h - bar_h), ((i + 1) * bar_w - 1, h), color, -1)

    # Add axis and labels
    cv2.putText(img, "Temporal Descriptor Summary", (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(img, f"Dim: {len(descriptor)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(img, f"Mean: {np.mean(descriptor):.3f}  Var: {np.var(descriptor):.3f}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imwrite(OUT_DESCRIPTOR_IMG, img)
    print(f"Saved visualization: {OUT_DESCRIPTOR_IMG}")

    # Display visually (optional)
    cv2.imshow("Temporal Descriptor", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === MAIN ===
def main():
    raw_features = load_stats()
    descriptor = normalize_and_expand(raw_features, target_dim=64)
    visualize_descriptor(descriptor)

if __name__ == "__main__":
    main()
