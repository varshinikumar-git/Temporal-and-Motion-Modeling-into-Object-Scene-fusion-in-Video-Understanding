import cv2
import numpy as np
import os

# === CONFIGURATION ===
BASE_DIR = r"F:\IMAGE ANALYSIS PROJECT"
FRAMES_DIR = os.path.join(BASE_DIR, "frames", "sample")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
OUT_VIDEO = os.path.join(OUT_DIR, "motion_evolution_slow.avi")

# Create outputs folder if missing
os.makedirs(OUT_DIR, exist_ok=True)

# === LOAD FRAME SEQUENCE ===
frame_files = sorted([
    os.path.join(FRAMES_DIR, f)
    for f in os.listdir(FRAMES_DIR)
    if f.lower().endswith((".jpg", ".png"))
])

if len(frame_files) < 2:
    raise ValueError("Need at least 2 frames in 'frames/sample' to create motion evolution video.")

print(f"Found {len(frame_files)} frames. Starting motion evolution generation...")

# === READ FIRST FRAME ===
prev = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
if prev is None:
    raise ValueError("Failed to read first frame. Check image format and path.")

h, w = prev.shape

# ✅ TRUE SLOW MOTION: reduce FPS
output_fps = 1   # 1 frame per second (very slow)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter(OUT_VIDEO, fourcc, output_fps, (w * 2, h))

if not video_writer.isOpened():
    raise IOError("VideoWriter failed to open. Try a different codec or output path.")

# === PROCESS EACH FRAME PAIR ===
for i in range(1, len(frame_files)):
    curr = cv2.imread(frame_files[i], cv2.IMREAD_GRAYSCALE)
    if curr is None:
        print(f"Skipping unreadable frame: {frame_files[i]}")
        continue

    diff = cv2.absdiff(curr, prev)
    diff_color = cv2.applyColorMap(cv2.convertScaleAbs(diff), cv2.COLORMAP_JET)

    flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    combined = np.hstack([diff_color, flow_color])
    cv2.putText(combined, f"Frame {i}/{len(frame_files)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    video_writer.write(combined)
    prev = curr

video_writer.release()
print(f"✅ Slow motion video saved: {OUT_VIDEO}")

# === SLOW PLAYBACK PREVIEW (ESC to exit) ===
cap = cv2.VideoCapture(OUT_VIDEO)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Slow Motion Playback", frame)

    # ✅ Increase delay for real slow view
    if cv2.waitKey(800) & 0xFF == 27:  # 800ms per frame = slow playback
        break

cap.release()
cv2.destroyAllWindows()
