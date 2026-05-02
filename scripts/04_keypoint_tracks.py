import cv2
import numpy as np
import os
import pandas as pd

# === CONFIG ===
BASE_DIR = r"F:\IMAGE ANALYSIS PROJECT"
FRAME_DIR = os.path.join(BASE_DIR, "frames", "sample")
OUTPUT_IMG = os.path.join(BASE_DIR, "outputs", "sample_keypoint_tracks.png")
OUTPUT_STATS = os.path.join(BASE_DIR, "outputs", "sample_tracks_stats.csv")

MAX_CORNERS = 300
QUALITY = 0.01
MIN_DISTANCE = 7
TRACK_COLOR = (0, 255, 0)  # default color (will vary per track)
MAX_TRACK_LEN = 20  # how many frames to draw a single track

# === 1. Load frames ===
def load_frames(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
    frames = []
    for file in files:
        img = cv2.imread(os.path.join(folder, file))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    print(f"Loaded {len(frames)} frames from {folder}")
    return frames

# === 2. Detect and track keypoints ===
def track_keypoints(frames):
    # output folder for per-frame visualizations
    output_folder = os.path.join(BASE_DIR, "outputs", "tracks_per_frame")
    os.makedirs(output_folder, exist_ok=True)

    # initialize tracking
    p_prev = cv2.goodFeaturesToTrack(
        frames[0], maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY, minDistance=MIN_DISTANCE
    )
    prev_gray = frames[0]

    all_tracks = []

    for i in range(1, len(frames)):
        # calculate optical flow between prev and current
        p_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frames[i], p_prev, None)
        if p_next is None:
            break

        good_new = p_next[st == 1]
        good_old = p_prev[st == 1]

        # save motion pairs for stats
        all_tracks.append(np.hstack([good_old, good_new]))

        # === create per-frame visualization ===
        color_img = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
        for (x0, y0, x1, y1) in np.hstack([good_old, good_new]).astype(np.int32):
            cv2.line(color_img, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cv2.circle(color_img, (x1, y1), 2, (0, 255, 255), -1)

        out_path = os.path.join(output_folder, f"tracks_{i:04d}.png")
        cv2.imwrite(out_path, color_img)

        # === update for next iteration ===
        prev_gray = frames[i]
        p_prev = good_new.reshape(-1, 1, 2)

    print(f"Saved {len(all_tracks)} per-frame track images to {output_folder}")
    return all_tracks


# === 3. Compute simple track stats ===
def compute_track_stats(all_tracks):
    displacements = []
    for t in all_tracks:
        dx = t[:, 2] - t[:, 0]
        dy = t[:, 3] - t[:, 1]
        dist = np.sqrt(dx**2 + dy**2)
        displacements.extend(dist)
    displacements = np.array(displacements)
    stats = {
        "mean_displacement": float(np.mean(displacements)),
        "std_displacement": float(np.std(displacements)),
        "max_displacement": float(np.max(displacements)),
        "num_tracks": len(displacements)
    }
    pd.DataFrame([stats]).to_csv(OUTPUT_STATS, index=False)
    print(f"Saved track stats to {OUTPUT_STATS}")
    return stats

# === 4. Visualize tracks ===
def visualize_tracks(frames, all_tracks):
    color_img = cv2.cvtColor(frames[len(frames)//2], cv2.COLOR_GRAY2BGR)
    h, w = color_img.shape[:2]

    # assign colors per track
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(len(all_tracks), 3))

    for i, t in enumerate(all_tracks[-MAX_TRACK_LEN:]):  # last few tracks
        for (x0, y0, x1, y1) in t.astype(np.int32):
            cv2.line(color_img, (x0, y0), (x1, y1), tuple(int(c) for c in colors[i]), 1)
            cv2.circle(color_img, (x1, y1), 2, (0, 255, 255), -1)

    return color_img

# === MAIN ===
def main():
    frames = load_frames(FRAME_DIR)
    if len(frames) < 2:
        print("Need more frames.")
        return

    all_tracks = track_keypoints(frames)
    stats = compute_track_stats(all_tracks)
    vis = visualize_tracks(frames, all_tracks)

    # Annotate
    y = 30
    for line in [
        "Keypoint Trajectory Visualization",
        f"Mean Disp: {stats['mean_displacement']:.3f}",
        f"Std Disp: {stats['std_displacement']:.3f}",
        f"Max Disp: {stats['max_displacement']:.3f}",
        f"Num Tracks: {stats['num_tracks']}"
    ]:
        cv2.putText(vis, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        y += 30

    cv2.imwrite(OUTPUT_IMG, vis)
    print(f"Saved trajectory visualization to {OUTPUT_IMG}")

    # Optionally display
    img = cv2.imread(OUTPUT_IMG)
    cv2.imshow("Keypoint Tracks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
