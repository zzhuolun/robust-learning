import sys
import os
sys.path.append('code/YOLOv3/')
sys.path.append('code/retinanet/')
sys.path.append("code/shared/attacks")
sys.path.append('/nfs/students/summer-term-2020/project-3/src/code/shared/representation_learning')

from perturbations import *
from attack_utils import *
from irl import cmd_2
from utils.datasets import *
import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter


def hidden_distance_retina(model, path, class_names, dist_type, save_dir, attack_type='Randn', eps=2, batch_size=8):
    dataset = CSVDataset(train_file=path, class_list=class_names, transform=transforms.Compose([Resizer()]))
    sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, num_workers=1, collate_fn=collater, batch_sampler=sampler)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print("Getting activations for distance type:", dist_type, "with attack_type: ", attack_type, "and epsilon: ", eps)
    model.eval()
    final_activations = {}
    iters = 0
    for batch_i, data in enumerate(tqdm.tqdm(dataloader)):
        imgs = data['img']
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        acts = model(imgs, send_activations=True)

        if attack_type == 'Randn':
            noi_imgs = generate_noisy_image(imgs, eps)
        elif attack_type == 'FGSM':
            model.train()
            model.training = True
            noi_imgs = generate_fgsm_image(model, imgs, data['annot'], eps, model_type="retina")
            model.training = False
            model.eval()

        noi_acts = model(noi_imgs, send_activations=True)
        final_activations = compute_loss(acts, dist_type, noi_acts, final_activations)
        iters += 1
    all_acts = {layer: (act / iters) for layer, act in final_activations.items()}
    output_to_txt(all_acts, save_dir)



def hidden_distance_yolo(model, path, img_size, dist_type, save_dir, attack_type='random_noise', eps=2, batch_size=2):
    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print("Getting activations for distance type:", dist_type, "with attack_type: ", attack_type, "and epsilon: ", eps)
    final_activations = {}
    iters = 0
    for batch_i, (img_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Getting activations")):
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        acts = model(imgs, visualize=True)

        if attack_type == 'random_noise':
            noi_imgs = generate_noisy_image(imgs, eps)
        elif attack_type == 'FGSM':
            noi_imgs = generate_fgsm_image(model, imgs, targets, eps, model_type="YOLO")

        noi_acts = model(noi_imgs, visualize=True)
        final_activations = compute_loss(acts, dist_type, noi_acts, final_activations)
        iters += 1
    all_acts = {layer: (act / iters) for layer, act in final_activations.items()}
    output_to_txt(all_acts, save_dir)


def compute_loss(acts, dist_type, noi_acts, final_activations):
    for layer, clean_acts in acts.items():
        noisy_acts = noi_acts[layer]
        if dist_type == 'l2':
            dist = torch.norm(clean_acts - noisy_acts, p=2, dim=(1, 2, 3)).mean()
        elif dist_type == 'l1':
            dist = torch.norm(clean_acts - noisy_acts, p=1, dim=(1, 2, 3)).mean()
        elif dist_type == 'cos':
            dist = cosine_distance(clean_acts, noisy_acts).mean()
        elif dist_type == 'cmd2':
            dist = cmd_2(clean_acts, noisy_acts)
        final_activations[layer] = (
                final_activations[layer] + dist.item()) if layer in final_activations.keys() else dist.item()
    return final_activations

def output_to_txt(activations,save_dir):
    with open(save_dir,'w+') as f:
        for layer in activations.keys():
            f.write(layer)
            f.write(',')
        f.write("\n")
        for dist in activations.values():
            f.write(str(dist))
            f.write(',')


def cosine_distance(x, y):
    x_flat = torch.flatten(x,start_dim=1)
    y_flat = torch.flatten(y,start_dim=1)
    return F.cosine_similarity(x_flat, y_flat, dim=1)

def get_layer_size(activations):
    layer_size={}
    for layer,vector in activations.items():
        layer_size[layer] = vector[0].flatten().shape[0]
    return layer_size