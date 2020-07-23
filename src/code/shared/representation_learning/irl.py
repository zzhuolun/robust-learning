import os
import sys

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
sys.path.append('code/shared/attacks')
from perturbations import *


class IRL(object):
    def __init__(self, noise_types, model_type, loss_type = 1, adv_attack_type='FGSM', epsilon=2, alpha=0.5, iterations=10):
        self.noise_types = noise_types
        self.model_type = model_type
        self.loss_type = loss_type
        self.adv_attack_type = adv_attack_type
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations

    def get_transform(self, typ):
        if typ == 'in_domain':
            return transforms.ColorJitter(brightness=0.5, contrast=0.5)
        elif typ == 'out_domain':
            return transforms.ColorJitter(saturation=0.5, hue=[-0.5, 0.5])
        else:
            return None

    def generate_noises(self, model, images, targets):
        noises = []
        for typ in self.noise_types:
            if typ == 'adversarial':
                if self.adv_attack_type == "PGD":
                    noise = generate_pgd_image(model, images, targets, self.alpha, self.epsilon,
                                               self.iterations, model_type=self.model_type)
                else:
                    noise = generate_fgsm_image(model, images, targets, self.epsilon, self.model_type)
            elif typ == 'random_noise':
                noise = generate_noisy_image(images, self.epsilon)
            else:
                pil_images = [transforms.ToPILImage()(img) for img in images]
                noise = []
                _tranform = self.get_transform(typ)
                for image in pil_images:
                    img = _tranform(image)
                    noise.append(img)
                noise = torch.stack([transforms.ToTensor()(img) for img in noise])
            noises.append(noise)

        return noises

    def compute_losses(self, model, images, targets, activations, epoch_num, batch_num, regularized_layers_yolo=None, training_name=None, avg_layers=False):
        noises = self.generate_noises(model, images, targets)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        noise = noises[0]

        if self.model_type == 'retina':
            _cls_loss, _reg_loss, noi_actvs = model([Variable(noise.to(device)), targets], send_activations=True)
            noise_loss = _cls_loss.mean() + _reg_loss.mean()
            # filepath = os.path.join('output', 'activations', training_name)
        elif self.model_type == 'YOLO':
            _loc_loss, _cls_loss, _conf_loss, noi_actvs = model(Variable(noise.to(device)), targets, regularized_layers=regularized_layers_yolo)
            noise_loss = _loc_loss + _cls_loss + _conf_loss
            # filepath = os.path.join('output', 'activations', (training_name + '_activations.txt'))

        layer_loss = []
        if avg_layers:
            gamma_layer = (1. / len(activations))
        else:
            gamma_layer = 1.
        # fp = open(filepath, 'a+')
        for layer, org_ft_maps in enumerate(activations):
            noisy_ft_maps = noi_actvs[layer]
            if self.loss_type == 1:
                l2_loss = torch.norm(org_ft_maps - noisy_ft_maps, p=2, dim=(1, 2, 3)).mean()
                cos_loss = cosine_distance(org_ft_maps, noisy_ft_maps).mean()
                layer_loss.append((l2_loss - cos_loss)*gamma_layer )
                fp.write("{}, {}, {}, {}\n".format(epoch_num, batch_num, layer, ((l2_loss - cos_loss)*gamma_layer).item()))
            elif self.loss_type == 2:
                l2_loss  = torch.norm(org_ft_maps - noisy_ft_maps, p = 2, dim =(1,2,3)).mean()
                l1_loss  = torch.norm(org_ft_maps - noisy_ft_maps, p = 1, dim =(1,2,3)).mean()
                layer_loss.append( ((l2_loss + l1_loss)/2)*gamma_layer )
                fp.write("{}, {}, {}, {}\n".format( epoch_num, batch_num, layer, ( ((l2_loss + l1_loss)/2)*gamma_layer ).item() ))
            elif self.loss_type == 3:
                l1_loss  = torch.norm(org_ft_maps - noisy_ft_maps, p = 1, dim =(1,2,3)).mean()
                cos_loss = cosine_distance(org_ft_maps, noisy_ft_maps).mean()
                layer_loss.append( (l1_loss - cos_loss)*gamma_layer )
                fp.write("{}, {}, {}, {}\n".format( epoch_num, batch_num, layer, ( (l1_loss - cos_loss)*gamma_layer ).item() ))
            elif self.loss_type == 4:
                linf_loss  = torch.norm(org_ft_maps - noisy_ft_maps, p = float("inf"), dim =(1,2,3)).mean()
                cos_loss = cosine_distance(org_ft_maps, noisy_ft_maps).mean()
                layer_loss.append( (linf_loss - cos_loss)*gamma_layer )
                fp.write("{}, {}, {}, {}\n".format( epoch_num, batch_num, layer, ( (linf_loss - cos_loss)*gamma_layer ).item() ))
            elif self.loss_type == 5:
                cos_loss = cosine_distance(org_ft_maps, noisy_ft_maps).mean()
                layer_loss.append( (1 - cos_loss)*gamma_layer )
                fp.write("{}, {}, {}, {}\n".format( epoch_num, batch_num, layer, ( (1 - cos_loss)*gamma_layer ).item() ))
            elif self.loss_type == 6:
                l2_loss = torch.norm(org_ft_maps - noisy_ft_maps, p=2, dim=(1, 2, 3))
                cos_sim = cosine_distance(org_ft_maps, noisy_ft_maps)
                mask = cos_sim > 0
                cos_loss = 1 - cos_sim * mask # + (cos_sim * (1 - mask.to(float)))
                layer_loss.append(gamma_layer *( 0.01 * l2_loss + cos_loss).mean())
            elif self.loss_type == 7:
                l2_loss = torch.norm(org_ft_maps - noisy_ft_maps, p=2, dim=(1, 2, 3))
                cos_sim = cosine_distance(org_ft_maps, noisy_ft_maps)
                mask = cos_sim > 0
                cos_loss = 1 - cos_sim * mask # + (cos_sim * (1 - mask.to(float)))
                layer_loss.append(gamma_layer *( 0.001 * l2_loss + 0.1 * cos_loss).mean())
            elif self.loss_type == 8:
                l2_loss = torch.norm(org_ft_maps - noisy_ft_maps, p=2, dim=(1, 2, 3))
                cos_sim = cosine_distance(org_ft_maps, noisy_ft_maps)
                mask = cos_sim > 0
                cos_loss = 1 - cos_sim * mask # + (cos_sim * (1 - mask.to(float)))
                layer_loss.append(gamma_layer * ( 0.1 * l2_loss + cos_loss).mean())
            elif self.loss_type == 9:
                layer_loss.append(gamma_layer * 0.5 * cmd_2(org_ft_maps, noisy_ft_maps))
            elif self.loss_type == 10:
                l2_loss = torch.norm(org_ft_maps - noisy_ft_maps, p=2, dim=(1, 2, 3))
                cos_sim = cosine_distance(org_ft_maps, noisy_ft_maps)
                mask = cos_sim > 0
                cos_loss = 1 - cos_sim * mask # + (cos_sim * (1 - mask.to(float)))
                layer_loss.append(gamma_layer *( 0.05 * l2_loss + cos_loss).mean())
            elif self.loss_type == 11:
                layer_loss.append(gamma_layer * cmd_2(org_ft_maps, noisy_ft_maps))

        # fp.close()

        distance_loss = torch.stack(layer_loss).sum()

        return noise_loss, distance_loss


# It takes two tensors x and y of the same shape. The expected input is the activations
# of one layer for n regular inputs (in one tensor) and n noisy inputs (in one tensor)
# In the torch.dist method it will convert into a long 1-d array and then
# calculate the pointwise l2 distance. Similarly cosine distance is calculated
# It will return  the mean of the desired distance between the n values.
def _distance(x, y, dist_type='l2'):
    if dist_type == 'l2':
        return torch.norm(x - y, p=2, dim=(1, 2, 3)).mean()
    elif dist_type == 'cos':
        return torch.abs(cosine_distance(x, y)).mean()
    elif dist_type == 'both':
        l2 = torch.norm(x - y, p=2, dim=(1, 2, 3)).mean()
        cos = torch.abs(cosine_distance(x, y)).mean()
        return torch.abs(l2 - cos)


def cosine_distance(x, y):
    x_flat = torch.flatten(x,start_dim=1)
    y_flat = torch.flatten(y,start_dim=1)
    return F.cosine_similarity(x_flat, y_flat, dim=1)


#note: return type is a 1-element tensor
def cmd(x, y, K=5, pnorm=2):
    x_flat = torch.flatten(x, start_dim=1)
    y_flat = torch.flatten(y, start_dim=1)
    b = max(x_flat.max().item(), y_flat.max().item())
    a = min(x_flat.min().item(), y_flat.min().item())
    x_mean = x_flat.mean(dim=0, keepdim=True)
    y_mean = y_flat.mean(dim=0, keepdim=True)
    x_centralized = x_flat - x_mean
    y_centralized = y_flat - y_mean
    dm = torch.norm(x_mean - y_mean, p=pnorm)/(b-a)
    scms = dm
    for order in range(2, K+1):
        # order = i + 2
        x_order = (x_centralized**order).mean(dim=0)
        y_order = (y_centralized**order).mean(dim=0)
        scms += torch.norm(x_order - y_order, p=pnorm)/(b-a)**order
    return scms

def cmd_2(x, y, K=3, pnorm=2):
    x_flat = torch.flatten(x, start_dim=1)
    y_flat = torch.flatten(y, start_dim=1)
    b = max(x_flat.max().item(), y_flat.max().item())
    a = min(x_flat.min().item(), y_flat.min().item())
    dm = torch.norm(x_flat - y_flat, p=pnorm, dim=1)/abs(b-a)
    scms = dm
    for order in range(2, K+1):
        # order = i + 2
        x_order = (x_flat**order)
        y_order = (y_flat**order)
        scms += torch.norm(x_order - y_order, p=pnorm, dim=1)/abs(b-a)**order
    return torch.mean(scms)


