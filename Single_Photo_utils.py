import numpy as np
import cv2
import torch
import albumentations as A


def read_image(path: str) -> np.array:
    image = cv2.imread(path)
    return image


def resize_image(image: np.array) -> np.array:
    t_test_orig = A.Resize(480, 640, p=1)
    return t_test_orig(image=image)['image']


def save_image(image: np.array, name: str):
    cv2.imwrite(f"outputs/{name}.jpg", image)


def draw_countours(outputs: np.array, image: np.array) -> np.array:

    outputs = outputs[0][0]
    mask = 255 * outputs.astype(np.uint8)

    im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Draw contours on image
    image_copy = image.copy()

    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    shapes = [con.shape[0] for con in contours]  # get biggest contour
    main_road_index = shapes.index(max(shapes))
    contours[main_road_index] = contours[main_road_index].reshape((contours[main_road_index].shape[0], 2))

    return image_copy, contours[main_road_index]


def image_overlay(image: np.array, outputs: np.array) -> np.array:

    def draw_segmentation_map(outputs: np.array):

        label = outputs.squeeze()

        red_map = np.zeros_like(label).astype(np.uint8)
        red_map[label == 1] = 255.0

        segmented_image = np.stack([red_map, np.zeros_like(label), np.zeros_like(label)], axis=2)
        return segmented_image.astype(np.uint8)

    segmented_image = draw_segmentation_map(outputs)
    alpha = 0.3
    beta = 1 - alpha
    gamma = 0
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)

    return image
