from catalyst import utils
import segmentation_models_pytorch as smp
import os
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.onnx


# Model load from catalyst checkpoint
def load_model(model: str) -> torch.nn.Module:

    model_checkpoints = {
        'Deeplabv3+_effnetb0': 'checkpoint_DeepLabV3Plus_efficientnet-b0',
        'Deeplabv3+_effnetb1': 'checkpoint_DeepLabV3Plus_efficientnet-b1',
        'PSP_effnetb1': 'checkpoint_PSPNet_efficientnet-b1',
        'Unet_effnetb1': 'checkpoint_Unet_efficientnet-b1',
        'Unet++_effnetb1': 'checkpoint_UnetPlusPlus_efficientnet-b1',
        'Unet++_mobilenet': 'checkpoint_UnetPlusPlus_timm-mobilenetv3_large_minimal_100',
        'FPN_effnetb0': 'checkpoint_FPN_efficientnet-b0',
        'Deeplabv3+_mobilenet': 'checkpoint_DeepLabV3Plus_timm-mobilenetv3_large_minimal_100'
    }

    if model in model_checkpoints.keys():
        name = model_checkpoints[model]
        checkpoint_path = os.path.join(os.getcwd(), 'Model_Checkpoints', name)
    else:
        raise NameError('Invalid model name')

    parameters = name.split('_')
    if len(parameters) > 3:
        parameters[2] = '_'.join(name.split('_')[2:])

    exec(f'nn_model = smp.{parameters[1]}('
         f'encoder_name = "{parameters[2]}",'
         f'encoder_weights="imagenet", '
         f'in_channels=3, '
         f'classes=1)', globals())

    if 'efficientnet' in parameters[2]:
        nn_model.encoder.set_swish(memory_efficient=False)

    try:
        checkpoint = utils.load_checkpoint(path=checkpoint_path)
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=nn_model
        )
    except:
        raise NameError('Error loading checkpoint')

    return nn_model


# Get predictions from model
def get_preds(model: torch.nn.Module, inputs, device) -> torch.Tensor:

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    t_test = A.Compose([A.Resize(480, 640, p=1), A.Normalize(mean, std), ToTensorV2()])

    image_copy = torch.unsqueeze(t_test(image=inputs)['image'], dim=0)
    with torch.no_grad():
        model.eval()
        preds = model.forward(image_copy.to(device))

    return (preds > 0.5).int().cpu().detach().numpy()


def model_to_onnx(model: torch.nn.Module, device):

    model.eval()
    x = torch.randn(1, 3, 480, 640, requires_grad=True)
    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      os.path.join(os.getcwd(), 'Model_Checkpoints', "model_testing.onnx"),  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})



