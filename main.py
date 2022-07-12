from Model_utils import load_model, get_preds, model_to_onnx
from Polynom_utils import fit_polynom
from Single_Photo_utils import draw_countours, read_image, resize_image, image_overlay, save_image
from Video_utils import segment_video, segment_video_onnx
from Inference_utils import initialize_model, evaluate
import matplotlib.pyplot as plt
import cv2
import torch
import warnings


def start(model_name='Unet++_mobilenet',
          model_path='./Model_Checkpoints/model_testing.onnx',
          video_path=r'C:\Users\Денис\DataScience\Выпускная работа\Video_summer.mp4',
          result_path=r'C:\Users\Денис\DataScience\Выпускная работа'
          ):

    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    model_name = model_name
    model = load_model(model_name)
    model.to(device)

    segment_video(video_path,
                  model,
                  result_path,
                  device)

    # model_to_onnx(model, device)

    # image = resize_image(read_image('../Datasets/KITTI/testing/image_2/000032_10.png'))
    # ort_session = initialize_model(model_path)

    # preds = evaluate(image, ort_session)
    # preds = get_preds(model, image, device)

    # plt.imshow(image)
    # plt.show()
    # plt.imshow((preds > 0.5)[0][0, :, :])
    # plt.show()

    # countered_image, contours = draw_countours(preds, image)

    # segmented_image = image_overlay(image, preds)

    # cv2.imshow('Image overlay', segmented_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # segment_video_onnx(video_path,
    #                    ort_session,
    #                    result_path)

    return


if __name__ == '__main__':
    start()
