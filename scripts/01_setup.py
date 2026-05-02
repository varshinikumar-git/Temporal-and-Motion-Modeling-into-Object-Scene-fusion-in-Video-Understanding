import cv2
import os
import numpy as np
from datetime import datetime

# === CONFIGURATION ===
BASE_DIR = r"F:\IMAGE ANALYSIS PROJECT"
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "sample.mp4")
FRAME_DIR = os.path.join(BASE_DIR, "frames", "sample")
OUTPUT_IMG = os.path.join(BASE_DIR, "outputs", "sample_frames_check.png")
LOG_FILE = os.path.join(BASE_DIR, "outputs", "setup_log.txt")

MAX_FRAMES = 100        # number of frames to read (adjust if needed)
STRIDE = 5              # sample every nth frame
RESIZE = (320, 180)     # width, height

# === 1. CREATE FOLDER STRUCTURE ===
def make_folders():
    subfolders = ["videos", "frames", "osf_features", "outputs", "scripts", "reports"]
    for sf in subfolders:
        path = os.path.join(BASE_DIR, sf)
        os.makedirs(path, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

# === 2. ENVIRONMENT CHECK ===
def env_check():
    versions = {
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
    print("Environment Check:")
    for k, v in versions.items():
        print(f"  {k}: {v}")
    return versions

# === 3. FRAME EXTRACTION ===
def extract_frames(video_path, stride=5, max_frames=100, resize=(320,180)):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % stride == 0:
            frame = cv2.resize(frame, resize)
            frames.append(frame)
            count += 1
            if count >= max_frames:
                break
        frame_id += 1
    cap.release()
    print(f"Extracted {len(frames)} frames from {os.path.basename(video_path)}")
    return frames

# === 4. SAVE FRAMES AND COMPOSITE GRID ===
def save_frames_and_grid(frames):
    for i, f in enumerate(frames):
        out_path = os.path.join(FRAME_DIR, f"frame_{i:04d}.jpg")
        cv2.imwrite(out_path, f)

    # Create a small grid preview (first 9 frames)
    grid_frames = frames[:9]
    if not grid_frames:
        print("No frames extracted.")
        return

    # Pad list to 9 for a full 3x3 grid
    while len(grid_frames) < 9:
        grid_frames.append(np.zeros_like(grid_frames[0]))

    rows = []
    for i in range(0, 9, 3):
        row = np.hstack(grid_frames[i:i+3])
        rows.append(row)
    grid = np.vstack(rows)

    # Annotate
    cv2.putText(grid, "Sample Frame Grid", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(OUTPUT_IMG, grid)
    print(f"Saved sample grid to {OUTPUT_IMG}")

# === 5. LOG ===
def write_log(info_dict):
    with open(LOG_FILE, "w") as f:
        f.write("=== SETUP LOG ===\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        for k, v in info_dict.items():
            f.write(f"{k}: {v}\n")
    print(f"Log written to {LOG_FILE}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    make_folders()
    env_info = env_check()
    try:
        frames = extract_frames(VIDEO_PATH, STRIDE, MAX_FRAMES, RESIZE)
        save_frames_and_grid(frames)
    except FileNotFoundError:
        print(f"[!] Please place a small test video at: {VIDEO_PATH}")
    write_log(env_info)
    print("Day 1 setup complete ✅")
