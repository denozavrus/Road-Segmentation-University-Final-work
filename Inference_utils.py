import torch
import onnx
import onnxruntime
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import onnxruntime.backend


def initialize_model(path: str):

    onnx_model = onnx.load_model(path)
    onnx.checker.check_model(onnx_model)

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider'
    ]

    ort_session = onnxruntime.InferenceSession(path, providers=providers)
    print(onnxruntime.get_device())

    return ort_session


def evaluate(image: np.array, ort_session: onnxruntime.InferenceSession):

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    t_test = A.Compose([A.Resize(480, 640, p=1), A.Normalize(mean, std), ToTensorV2()])
    image_copy = torch.unsqueeze(t_test(image=image)['image'], dim=0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image_copy)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return (img_out_y > 0.5).astype(int)

