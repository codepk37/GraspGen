import json
import os
import numpy as np
from PIL import Image

# 1. Define paths
source_dir = "GraspGenModels/sample_data/aygrasp_data"
target_dir = os.path.join(source_dir, "01")  # e.g. scene "01"
os.makedirs(target_dir, exist_ok=True)

# 2. Rename/Move color.png -> rgb.png
color_img = Image.open(os.path.join(source_dir, "color.png")).convert("RGB")
color_img.save(os.path.join(target_dir, "rgb.png"))

# 3. Rename/Move seg_mask.png -> seg.png
seg_img = Image.open(os.path.join(source_dir, "seg_mask.png"))
seg_img.save(os.path.join(target_dir, "seg.png"))
seg_arr = np.array(seg_img)

# 4. Convert depth image -> depth.npy (meters)
# If depth.png is float (e.g. EXR/TIFF) or grayscale already representing meters:
depth_img = Image.open(os.path.join(source_dir, "depth.png"))
depth_arr = np.array(depth_img, dtype=np.float32)

# NOTE: If depth.png is 16-bit PNG in millimeters, convert to meters:
depth_arr = depth_arr / 1000.0

np.save(os.path.join(target_dir, "depth.npy"), depth_arr)

# 5. Extract unique labels from segmentation mask to generate label_map
unique_ids = np.unique(seg_arr)
label_map = {}
for uid in unique_ids:
    if uid == 0:  # Skip background if 0
        continue
    label_map[f"obj_{uid}"] = int(uid)

# 6. Build and save meta_data.json
meta_data = {
    "intrinsics": [[927.17, 0.0, 651.32], [0.0, 927.37, 349.62], [0.0, 0.0, 1.0]],
    "camera_pose": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],  # Identity matrix if frame is already in camera coordinates
    "label_map": label_map,
}

with open(os.path.join(target_dir, "meta_data.json"), "w") as f:
    json.dump(meta_data, f, indent=4)

print(
    f"Successfully formatted dataset into {target_dir} with {len(label_map)} objects!"
)
