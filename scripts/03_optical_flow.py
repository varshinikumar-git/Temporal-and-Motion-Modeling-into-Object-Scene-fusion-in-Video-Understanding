import cv2
import numpy as np
import os
import pandas as pd

# === CONFIGURATION ===
BASE_DIR = r"F:\IMAGE ANALYSIS PROJECT"
FRAME_DIR = os.path.join(BASE_DIR, "frames", "sample")
OUTPUT_IMG = os.path.join(BASE_DIR, "outputs", "sample_flow_visualization.png")
OUTPUT_STATS = os.path.join(BASE_DIR, "outputs", "sample_flow_stats.csv")

RESIZE = (320, 180)   # keep consistent with previous step
COLORMAP = cv2.COLORMAP_HSV  # HSV color encodes flow direction

# === 1. Load frames ===
def load_frames(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
    frames = []
    for file in files:
        img = cv2.imread(os.path.join(folder, file))
        if img is None:
            continue
        img = cv2.resize(img, RESIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    print(f"Loaded {len(frames)} frames from {folder}")
    return frames

# === 2. Compute optical flow between consecutive frames ===
def compute_farneback_flow(frames):
    flows = []
    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(
            prev=frames[i-1], next=frames[i],
            flow=None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flows.append(flow)
    print(f"Computed {len(flows)} optical flow fields")
    return flows

# === 3. Convert flow to magnitude and angle ===
def compute_mag_angle(flows):
    mags, angs = [], []
    for f in flows:
        mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
        mags.append(mag)
        angs.append(ang)
    mags = np.stack(mags)
    angs = np.stack(angs)
    return mags, angs

# === 4. Aggregate flow descriptors ===
def compute_flow_stats(mags, angs, bins=8):
    mean_mag = np.mean(mags)
    var_mag = np.var(mags)
    median_mag = np.median(mags)
    
    # Orientation histogram (weighted by magnitude)
    hist = np.zeros(bins)
    for i in range(mags.shape[0]):
        mag_flat = mags[i].flatten()
        ang_flat = angs[i].flatten()
        bin_idx = np.int32(bins * ang_flat / (2 * np.pi)) % bins
        for b in range(bins):
            hist[b] += np.sum(mag_flat[bin_idx == b])
    hist /= np.sum(hist) + 1e-8

    stats = {
        "mean_magnitude": mean_mag,
        "var_magnitude": var_mag,
        "median_magnitude": median_mag
    }
    pd.DataFrame([stats]).to_csv(OUTPUT_STATS, index=False)
    print(f"Saved flow stats to {OUTPUT_STATS}")
    return stats, hist

# === 5. Create visualization (color-coded flow map) ===
def visualize_flow(frames, flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2          # Hue: direction
    hsv[..., 1] = 255                            # Saturation: full
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Overlay on a mid-frame
    mid_frame = cv2.cvtColor(frames[len(frames)//2], cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(mid_frame, 0.5, bgr, 0.8, 0)
    return blended

# === MAIN ===
def main():
    frames = load_frames(FRAME_DIR)
    if len(frames) < 2:
        print("Not enough frames for optical flow.")
        return

    flows = compute_farneback_flow(frames)
    mags, angs = compute_mag_angle(flows)
    stats, hist = compute_flow_stats(mags, angs)

    # Visualize using the last computed flow (most recent motion)
    flow_visual = visualize_flow(frames, flows[-1])

    # Draw text and small histogram bar
    y = 25
    for line in [
        "Optical Flow Visualization",
        f"Mean Mag: {stats['mean_magnitude']:.4f}",
        f"Variance: {stats['var_magnitude']:.4f}",
        f"Median Mag: {stats['median_magnitude']:.4f}"
    ]:
        cv2.putText(flow_visual, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30

    # Draw orientation histogram as small bars
    h_img = np.zeros((100, 320, 3), dtype=np.uint8)
    bin_w = int(320 / len(hist))
    for i, h in enumerate(hist):
        cv2.rectangle(h_img, (i * bin_w, 100), ((i + 1) * bin_w - 2, 100 - int(h * 100)),
                      (0, 255, 255), -1)
    h_img = cv2.flip(h_img, 0)
    combined = cv2.vconcat([flow_visual, h_img])

    cv2.imwrite(OUTPUT_IMG, combined)
    print(f"Saved flow visualization to {OUTPUT_IMG}")

    # Optionally show
    img = cv2.imread(OUTPUT_IMG)
    cv2.imshow("Optical Flow Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
