import cv2
import numpy as np
import os
import pandas as pd

# ==== CONFIG ====
BASE_DIR = r"F:\IMAGE ANALYSIS PROJECT"
FRAME_DIR = os.path.join(BASE_DIR, "frames", "sample")
OUTPUT_IMG = os.path.join(BASE_DIR, "outputs", "sample_frame_diff_energy.png")
OUTPUT_STATS = os.path.join(BASE_DIR, "outputs", "sample_frame_diff_stats.csv")

THRESHOLD = 25  # motion threshold (pixel difference)
BLUR_SIZE = (5, 5)
COLORMAP = cv2.COLORMAP_JET

# ==== 1. Load frames ====
def load_frames(folder):
    frames = []
    files = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
    for file in files:
        img = cv2.imread(os.path.join(folder, file))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    print(f"Loaded {len(frames)} frames from {folder}")
    return frames

# ==== 2. Compute frame differences ====
def compute_motion_energy(frames):
    diffs = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i-1])
        diff = cv2.GaussianBlur(diff, BLUR_SIZE, 0)
        _, mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
        diffs.append(mask.astype(np.float32) / 255.0)
    motion_energy_map = np.mean(diffs, axis=0)
    return motion_energy_map

# ==== 3. Overlay visualization ====
def overlay_motion(frame, energy_map):
    # Normalize and colorize the energy map
    norm_map = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_map, COLORMAP)
    blended = cv2.addWeighted(frame, 0.6, heatmap, 0.8, 0)
    return blended

# ==== 4. Compute and save stats ====
def compute_stats(energy_map):
    mean_energy = np.mean(energy_map)
    var_energy = np.var(energy_map)
    active_fraction = np.sum(energy_map > 0.1) / energy_map.size
    stats = {
        "mean_motion_energy": mean_energy,
        "var_motion_energy": var_energy,
        "active_fraction": active_fraction
    }
    pd.DataFrame([stats]).to_csv(OUTPUT_STATS, index=False)
    print(f"Stats saved to {OUTPUT_STATS}")
    return stats

# ==== 5. Main workflow ====
def main():
    frames = load_frames(FRAME_DIR)
    if len(frames) < 2:
        print("Need at least 2 frames to compute motion.")
        return

    motion_map = compute_motion_energy(frames)
    sample_frame = cv2.cvtColor(frames[len(frames)//2], cv2.COLOR_GRAY2BGR)
    overlay = overlay_motion(sample_frame, motion_map)

    # Add text annotations
    stats = compute_stats(motion_map)
    text_lines = [
        "Frame Differencing Motion Energy Map",
        f"Mean: {stats['mean_motion_energy']:.4f}",
        f"Variance: {stats['var_motion_energy']:.4f}",
        f"Active Fraction: {stats['active_fraction']:.4f}"
    ]
    y0 = 30
    for line in text_lines:
        cv2.putText(overlay, line, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        y0 += 30

    cv2.imwrite(OUTPUT_IMG, overlay)
    print(f"Motion overlay saved to {OUTPUT_IMG}")

    # Optionally display interactively
    img = cv2.imread(OUTPUT_IMG)
    cv2.imshow("Motion Energy Overlay", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
