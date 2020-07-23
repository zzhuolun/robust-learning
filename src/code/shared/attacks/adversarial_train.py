import os
import sys

import torch
import torch.utils.data
import torchvision
from PIL import Image
from attack_utils import *
from perturbations import *
from torch.autograd import Variable
from torchvision import transforms


def mix_batch(model, images, targets, batch_size, epsilon=4, alpha=1, iterations=10, mix_thre=0.5,
              attack_type='Normal', model_type='YOLO', sign_grad=False):
    if attack_type == 'Normal':
        # print('No attack')
        return images
    elif attack_type == 'FGSM':
        attack = generate_fgsm_image(model, images, targets, epsilon, model_type)
    elif attack_type == 'PGD':
        attack = generate_pgd_image(model, images, targets, alpha, epsilon, iterations, model_type=model_type,
                                    sign_grad=sign_grad)
    elif attack_type == 'RFGSM':
        attack = generate_rfgsm_image(model, images, targets, epsilon, model_type)
    else:
        raise ValueError('Unsupported attack type')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bit_mask = torch.rand(batch_size) > mix_thre  # 0.5 threshold of mixing batch
    bit_mask = bit_mask.float()
    bit_mask = bit_mask.reshape(-1, 1, 1, 1)
    images = Variable(images.to(device))
    bit_mask = Variable(bit_mask.to(device))
    batch_mix = (1 - bit_mask) * images + bit_mask * attack
    return batch_mix