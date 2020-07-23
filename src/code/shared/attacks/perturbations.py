import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from attack_utils import *
from torch.autograd import Variable
from torchvision import transforms

def generate_fgsm_image(model, images, targets, epsilon=4, model_type='YOLO', loss_type='all'):
    # Load images and targets to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = Variable(images.to(device).float(), requires_grad=True)
    targets = targets.to(device)

    if model_type == 'YOLO':
        loss_loc, loss_cls, loss_conf, _ = model(images, targets)
        if loss_type == 'all':
            loss = loss_loc + loss_cls + loss_conf
        elif loss_type == 'loc':
            loss = loss_loc
        elif loss_type == 'cls':
            loss = loss_cls
        elif loss_type == 'conf':
            loss = loss_conf
    elif model_type == 'retina':
        classification_loss, regression_loss = model([images, targets])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        if loss_type =='all':
            loss = classification_loss + regression_loss
        elif loss_type == 'loc':
            loss = regression_loss
        elif loss_type == 'cls':
            loss = classification_loss
    else:
        raise ValueError('Unsupported model type')

    grad = torch.autograd.grad(loss, images, only_inputs=True)[0]
    grad = torch.sign(grad.data)

    if type(epsilon) is tuple:
        eps = torch.rand(1).item() * (epsilon[1] - epsilon[0]) + epsilon[0]
        eps/=255
    else:
        eps = epsilon / 255.
    fgsm_images = images.data + (eps * grad)
    result = torch.clamp(fgsm_images, 0, 1)

    return result


def generate_pgd_image(model, images, targets, alpha=1, epsilon=4, iterations=10, model_type='YOLO', plot_losses=False,
                       sign_grad=False):
    # Load targets to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = targets.to(device)
    images = images.to(device)
    if type(epsilon) is tuple:
        eps = torch.rand(1).item() * (epsilon[1] - epsilon[0]) + epsilon[0]
        eps /= 255
    else:
        eps = epsilon / 255.
    startpoint = torch.randn_like(images) * (eps / 2)
    pgd_images = torch.clamp((images + startpoint), 0, 1)

    plt_losses = []
    for _ in range(iterations):
        pgd_images = Variable(pgd_images.to(device).float(), requires_grad=True)

        if model_type == 'YOLO':
            loss_loc, loss_cls, loss_conf , _ = model(pgd_images, targets)
            loss = loss_loc + loss_cls + loss_conf
        elif model_type == 'retina':
            classification_loss, regression_loss = model([pgd_images, targets])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
        else:
            raise ValueError('Unsupported model type')

        plt_losses.append(loss.item())
        grad = torch.autograd.grad(loss, pgd_images, only_inputs=True)[0].data
        if sign_grad:
            grad = torch.sign(grad)
            # alpha = 2.5 * eps / iterations
        adv_images = torch.clamp(pgd_images.data + (alpha * grad), 0, 1)
        # Projection on L-infinity ball
        pgd_images = torch.max(torch.min(adv_images, images + eps), images - eps)

    result = pgd_images

    if plot_losses:
        return result, plt_losses

    return result


def generate_rfgsm_image(model, images, targets, epsilon, model_type='YOLO',):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = targets.to(device)
    images = images.to(device)
    if type(epsilon) is tuple:
        eps = torch.rand(1).item() * (epsilon[1] - epsilon[0]) + epsilon[0]
        eps/=255
    else:
        eps = epsilon / 255.
    alp = eps/2
    startpoint = torch.sign(torch.randn_like(images)) * alp
    pgd_images = torch.clamp((images + startpoint), 0, 1)
    pgd_images = Variable(pgd_images.to(device).float(), requires_grad=True)

    if model_type == 'YOLO':
        loss_loc, loss_cls, loss_conf , _ = model(pgd_images, targets)
        loss = loss_loc + loss_cls + loss_conf
    elif model_type == 'retina':
        classification_loss, regression_loss = model([pgd_images, targets])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
    else:
        raise ValueError('Unsupported model type')

    grad = torch.autograd.grad(loss, pgd_images, only_inputs=True)[0].data
    grad = torch.sign(grad)
    adv_images = torch.clamp(pgd_images.data + ((eps-alp) * grad), 0, 1)
    return adv_images

def generate_noisy_image(images, noise):
    noi = round(noise / 255, 3)
    gaus_noise = torch.randn_like(images)
    gaus_noise = torch.clamp(gaus_noise, -1, 1)
    noisy_images = images.data + (gaus_noise * noi)
    return torch.clamp(noisy_images, 0, 1)


def observe_fgsm_loss(model, images, targets, model_type='retina', loss_type='all'):
    # Load images and targets to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = Variable(images.to(device).float(), requires_grad=True)
    targets = targets.to(device)

    if model_type == 'YOLO':
        loss_loc, loss_cls, loss_conf, _ = model(images, targets)
        if loss_type == 'all':
            loss = loss_loc + loss_cls + loss_conf
        elif loss_type == 'loc':
            loss = loss_loc
        elif loss_type == 'cls':
            loss = loss_cls
        elif loss_type == 'conf':
            loss = loss_conf
    elif model_type == 'retina':
        classification_loss, regression_loss = model([images, targets])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
    else:
        raise ValueError('Unsupported model type')

    grad = torch.autograd.grad(loss, images, only_inputs=True)[0]
    grad = torch.sign(grad.data)
    gradv = torch.sign(torch.rand_like(grad))
    # cos =torch.nn.CosineSimilarity(dim=1)
    # cos_sim = cos(grad,gradv)
    losses = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            eps1 = i/255
            eps2 = j/255
            fgsm_image = torch.clamp(images.data + eps1 * grad + eps2 * gradv, 0, 1)
            classification_loss, regression_loss = model([fgsm_image, targets])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            losses[i,j] = classification_loss + regression_loss
    return losses
