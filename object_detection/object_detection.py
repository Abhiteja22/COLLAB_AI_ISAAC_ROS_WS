from PIL import Image
from lang_sam import LangSAM
import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

print("Warming up the camera...")
for i in range(30):  # Wait for 30 frames (about 1 second at 30 FPS)
	frames = pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
if not color_frame:
    print("Error: Could not get a color frame.")
else:
    color_image = np.asanyarray(color_frame.get_data())
    
    cv2.imwrite("realsense_snapshot.png", color_image)
    print("Image saved as realsense_snapshot.png")

pipeline.stop()
print("Langsam")
model = LangSAM()
image_pil = Image.open("realsense_snapshot.png").convert("RGB")
text_prompt = "mug."
masks, boxes, phrases, logits = model.predict([image_pil], [text_prompt])
print(masks)
print(boxes)
print(phrases)
print(logits)
print("Langsam end")
