import os, cv2, numpy as np

BASE = r"F:\IMAGE ANALYSIS PROJECT\outputs"
OUT_IMG = os.path.join(BASE, "final_composite_visualization.png")

def load_img(name):
    path = os.path.join(BASE, name)
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Missing {name}, creating blank placeholder.")
        img = np.zeros((180, 320, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (320, 180))
    return img

# === 1. LOAD STAGE IMAGES ===
diff_img  = load_img("sample_frame_diff_energy.png")
flow_img  = load_img("sample_flow_visualization.png")
track_img = load_img("sample_keypoint_tracks.png")
desc_img  = load_img("sample_temporal_descriptor.png")
osf_img   = load_img("sample_osf_integration.png")

# === 2. LOAD NUMERIC OSF FEATURES ===
try:
    f1 = np.load(os.path.join(BASE, "sample_osf_avg.npy"))
    f2 = np.load(os.path.join(BASE, "sample_osf_motion_weighted.npy"))
    sim = np.dot(f1, f2) / (np.linalg.norm(f1)*np.linalg.norm(f2) + 1e-8)

    sim_img = np.ones((180, 320, 3), dtype=np.uint8)
    col = int(255 * sim)
    sim_img[:] = (col, 0, 255-col)
    cv2.putText(sim_img, f"{sim:.2f}", (100,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
except:
    sim_img = np.zeros((180,320,3), dtype=np.uint8)

# === 3. COMBINE 2×3 GRID ===
top_row = np.hstack([diff_img, flow_img, track_img])
bottom_row = np.hstack([desc_img, osf_img, sim_img])
composite = np.vstack([top_row, bottom_row])

# === 4. ADD LABEL BANDS ABOVE EACH TILE ===
TILE_W, TILE_H = 320, 180
LABEL_H = 28   # Height of label bar
font = cv2.FONT_HERSHEY_SIMPLEX

labels = [
    "Frame Differencing", "Optical Flow", "Keypoint Tracks",
    "Temporal Descriptor", "OSF Integration", "OSF Similarity"
]

for i, text in enumerate(labels):
    col = i % 3          # column index
    row = i // 3         # row index

    x0 = col * TILE_W
    y0 = row * TILE_H

    # draw semi-transparent rectangle as label background
    overlay = composite.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + TILE_W, y0 + LABEL_H), (0,0,0), -1)
    composite = cv2.addWeighted(overlay, 0.45, composite, 0.55, 0)

    # center the text inside the label bar
    text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
    text_x = x0 + (TILE_W - text_size[0]) // 2
    text_y = y0 + LABEL_H - 6

    cv2.putText(composite, text, (text_x, text_y),
                font, 0.6, (255,255,255), 2)

# === 5. SAVE & SHOW ===
cv2.imwrite(OUT_IMG, composite)
print(f"✅ Final composite saved: {OUT_IMG}")
