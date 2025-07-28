from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F


image = Image.open("../test_image/car.jpg").convert("RGB")
print(image.width)

image_tensor = F.to_tensor(image)

image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-coco-instance")

inputs = image_processor([image], return_tensors = "pt")

with torch.no_grad():
    outputs = model(**inputs)

class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# print(class_queries_logits.shape)
# print(masks_queries_logits.shape)

pred_instance_map = image_processor.post_process_instance_segmentation(
    outputs, target_sizes=[(image.height, image.width)]
)[0]

print(outputs.keys())
print(pred_instance_map.keys())
# print(pred_instance_map['segmentation'].shape)

#predicted masks
segmentation = pred_instance_map["segmentation"]
print(pred_instance_map["segments_info"])
# print(masks)
# print(torch.unique(masks))
segments_info = pred_instance_map["segments_info"]

#persegment mask
masks = []
for segment in segments_info:
    segment_id = segments_info
    mask = segmentation == segment['id']
    masks.append(mask)

print(type(segment['id']))
if masks:
    masks_tensor = torch.stack(masks)

    colored_overlay = draw_segmentation_masks(
        image=image_tensor,
        masks=masks_tensor, 
        alpha = 0.5
    )

    # Show result
    plt.figure(figsize=(10, 10))
    plt.imshow(colored_overlay.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("Panoptic Segmentation Overlay")
    plt.show()
    plt.savefig("../output_segmentation/output.png")
else:
    print("No segments found.")