import numpy as np
import pandas as pd
import cv2
import os

# === CONFIG ===
BASE_DIR = r"F:\IMAGE ANALYSIS PROJECT"
OSF_FEATURES = os.path.join(BASE_DIR, "osf_features", "sample_osf_per_frame.npy")
TEMPORAL_DESC = os.path.join(BASE_DIR, "outputs", "sample_temporal_descriptor.npy")
FRAME_DIFF_STATS = os.path.join(BASE_DIR, "outputs", "sample_frame_diff_stats.csv")

OUT_AVG = os.path.join(BASE_DIR, "outputs", "sample_osf_avg.npy")
OUT_WEIGHTED = os.path.join(BASE_DIR, "outputs", "sample_osf_motion_weighted.npy")
OUT_CONCAT = os.path.join(BASE_DIR, "outputs", "sample_osf_concat.npy")
OUT_IMG = os.path.join(BASE_DIR, "outputs", "sample_osf_integration.png")

# === 1. LOAD OR MOCK OSF FEATURES ===
def load_osf_per_frame():
    if os.path.exists(OSF_FEATURES):
        arr = np.load(OSF_FEATURES)
        print(f"Loaded OSF per-frame features: {arr.shape}")
    else:
        print("No OSF feature file found, generating mock features for demo.")
        np.random.seed(42)
        arr = np.random.rand(50, 128).astype(np.float32)  # 50 frames × 128-dim
    return arr

# === 2. LOAD TEMPORAL DESCRIPTOR ===
def load_temporal_descriptor():
    desc = np.load(TEMPORAL_DESC)
    print(f"Loaded temporal descriptor: {desc.shape}")
    return desc

# === 3. LOAD MOTION WEIGHTS FROM FRAME DIFFERENCING ===
def load_motion_weights(num_frames):
    df = pd.read_csv(FRAME_DIFF_STATS)
    mean_energy = float(df["mean_motion_energy"].iloc[0])
    var_energy = float(df["var_motion_energy"].iloc[0])
    # Generate synthetic per-frame weights around mean ± variance
    np.random.seed(0)
    w = np.abs(np.random.normal(loc=mean_energy, scale=var_energy, size=num_frames))
    w = w / (np.sum(w) + 1e-8)
    return w

# === 4. MOTION-WEIGHTED POOLING ===
def motion_weighted_pooling(osf_frames, weights):
    weights = weights[:osf_frames.shape[0]]
    pooled = np.sum(osf_frames * weights[:, None], axis=0)
    return pooled

# === 5. CONCATENATION ===
def concatenate_features(osf_avg, temporal_desc):
    # Resize if necessary
    if temporal_desc.ndim > 1:
        temporal_desc = temporal_desc.flatten()
    concat = np.concatenate([osf_avg, temporal_desc])
    return concat

# === 6. VISUALIZATION ===
def visualize_integration(avg_vec, weighted_vec, concat_vec):
    # Normalize for bar plotting
    def norm(v): return (v - v.min()) / (v.max() - v.min() + 1e-8)
    avg_norm = norm(avg_vec)
    wgt_norm = norm(weighted_vec)
    con_norm = norm(concat_vec[:len(avg_vec)])  # same length section for comparison

    h, w = 180, 320
    n = len(avg_norm)
    bar_w = max(1, int(w / n))

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n):
        y1 = h - int(avg_norm[i] * 50)
        y2 = h - int(wgt_norm[i] * 100)
        cv2.line(canvas, (i * bar_w, y1), (i * bar_w, y2), (0, 255, 255), 1)

    cv2.putText(canvas, "OSF Integration", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, "Yellow: motion-weighted  |  Cyan: baseline", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(OUT_IMG, canvas)
    print(f"Visualization saved: {OUT_IMG}")
    return canvas

# === MAIN ===
def main():
    osf_frames = load_osf_per_frame()
    temporal_desc = load_temporal_descriptor()

    # --- baseline average ---
    osf_avg = np.mean(osf_frames, axis=0)
    np.save(OUT_AVG, osf_avg)

    # --- motion-weighted pooling ---
    weights = load_motion_weights(osf_frames.shape[0])
    osf_motion = motion_weighted_pooling(osf_frames, weights)
    np.save(OUT_WEIGHTED, osf_motion)

    # --- concatenation with temporal descriptor ---
    concat_vec = concatenate_features(osf_avg, temporal_desc)
    np.save(OUT_CONCAT, concat_vec)

    # --- visualization ---
    vis = visualize_integration(osf_avg, osf_motion, concat_vec)
    cv2.imshow("OSF Integration", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
