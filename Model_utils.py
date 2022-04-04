from catalyst import utils
import segmentation_models_pytorch as smp
import os
import torch

model_checkpoints = {
    'Deeplabv3+_effnetb0' : 'checkpoint_DeepLabV3Plus_efficientnet-b0',
    'Deeplabv3+_effnetb1' : 'checkpoint_DeepLabV3Plus_efficientnet-b1',
    'PSP_effnetb1' : 'checkpoint_PSPNet_efficientnet-b1',
    'Unet_effnetb1' : 'checkpoint_Unet_efficientnet-b1',
    'Unet++_effnetb1' : 'checkpoint_UnetPlusPlus_efficientnet-b1'
}

# Model load from catalyst checkpoint
def load_model(model:str) -> torch.nn.Module:

    if model in model_checkpoints.keys():
        name = model_checkpoints[model]
        checkpoint_path = os.path.join(os.getcwd(), 'Model_Checkpoints', name)
    else:
        raise NameError('Invalid model name')

    parameters = name.split('_')
    exec(f'nn_model = smp.{parameters[1]}('
         f'encoder_name = "{parameters[2]}",'
         f'encoder_weights="imagenet", '
         f'in_channels=3, '
         f'classes=1)', globals())

    try:
        checkpoint = utils.load_checkpoint(path=checkpoint_path)
        utils.unpack_checkpoint(
            checkpoint=checkpoint,
            model=nn_model
        )
    except:
        raise NameError('Error loading checkpoint')

    return nn_model

model = load_model('Unet++_effnetb1')
print(model)