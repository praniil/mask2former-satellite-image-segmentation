from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch

import matplotlib.pyplot as plt
import numpy as np

image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-ade-semantic"
)

image = Image.open("../test_image/car.jpg").convert("RGB")
inputs = image_processor(image, return_tensors = "pt")
# print(inputs)

with torch.no_grad():
    outputs = model(**inputs)

class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# Perform post-processing to get semantic segmentation map
pred_semantic_map = image_processor.post_process_semantic_segmentation(
    outputs, target_sizes=[(image.height, image.width)]
)[0]
print(pred_semantic_map.shape)

# Convert the original image to a numpy array
image_np = np.array(image)

# Convert the predicted semantic map to numpy (should be shape H x W)
seg_map = pred_semantic_map.cpu().numpy()

# Plot the original image and the segmentation map
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg_map)
plt.title("Predicted Segmentation Map")
plt.axis("off")

plt.tight_layout()
plt.savefig("../output_segmentation/output_semantic.png")
