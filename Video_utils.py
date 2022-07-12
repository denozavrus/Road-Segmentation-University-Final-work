import cv2
import torch
import time
import albumentations as A
from Model_utils import get_preds
from Single_Photo_utils import image_overlay
from Inference_utils import evaluate
import onnxruntime


def segment_video_onnx(path: str, ort_session: onnxruntime.InferenceSession, output_path: str):
    t_test_orig = A.Resize(480, 640, p=1)

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')

    save_name = "test video UNet++ MobileNet"

    out = cv2.VideoWriter(f"{output_path}\{save_name}.avi",
                          cv2.VideoWriter_fourcc(*'MJPG'), 30,
                          (640, 480))

    frame_count = 0
    total_fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            start_time = time.time()

            with torch.no_grad():
                preds = evaluate(frame, ort_session)
                image_orig = t_test_orig(image=frame)['image']
                end_time = time.time()
                final_image = image_overlay(image_orig, preds)
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(final_image)

                cv2.imshow('image', final_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            break

    cap.release()
    out.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

    return


def segment_video(path: str, model: torch.nn.Module, output_path: str, device: torch.device):
    t_test_orig = A.Resize(480, 640, p=1)
    model.eval()

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')

    save_name = "test video"

    out = cv2.VideoWriter(f"{output_path}\{save_name}.avi",
                          cv2.VideoWriter_fourcc(*'MJPG'), 30,
                          (640, 480))

    frame_count = 0
    total_fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            start_time = time.time()
            with torch.no_grad():

                preds = get_preds(model, frame, device)
                image_orig = t_test_orig(image=frame)['image']
                end_time = time.time()
                final_image = image_overlay(image_orig, preds)
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(final_image)

                cv2.imshow('image', final_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            break

    cap.release()
    out.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

    return
