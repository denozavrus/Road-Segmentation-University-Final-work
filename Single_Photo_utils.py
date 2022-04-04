import numpy as np
import torch
def draw_segmentation_map(outputs):
    preds=(outputs > 0.5).int().cpu().detach()
    label=preds.squeeze()
    red_map=np.zeros_like(label).astype(np.uint8)

    red_map=np.zeros_like(label).astype(np.uint8)
    red_map[label == 1]=255.0

    segmented_image=np.stack([red_map, np.zeros_like(label), np.zeros_like(label)], axis=2)
    return segmented_image.astype(np.uint8)