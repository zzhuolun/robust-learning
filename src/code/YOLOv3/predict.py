from __future__ import division

import sys
sys.path.append('code/YOLOv3/')
sys.path.append('code/shared/attacks')
sys.path.append('code/shared/plots')

from perturbations import *
from attack_utils import *
from layer_distance import hidden_distance_yolo

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def predict(model, path, conf_thres, nms_thres, img_size, batch_size, save_dir, prediction_mode, attack_type, eps, alpha, iterations, sign_grad, loss_type):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for batch_i, (img_path, img, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        img_name = img_path[0].split('/')[-1].split('.')[0]
        img = Variable(img.type(Tensor), requires_grad=False)

        if attack_type == "FGSM":
            img = generate_fgsm_image(model, img, targets, eps, model_type="YOLO", loss_type=loss_type)
        elif attack_type == "PGD":
            img = generate_pgd_image(model, img, targets, alpha, eps, iterations, 
                                            model_type="YOLO", plot_losses=False, sign_grad=sign_grad)
        elif attack_type == "random_noise":
            img = generate_noisy_image(img, eps)
        elif attack_type != "none":
            raise ValueError(f"Attack type {attack_type} not implemented.")

        # plt.figure()
        # plt.plot(losses)
        # plt.savefig(f"output/{img_name}-{alpha}.png")

        with torch.no_grad():
            outputs = model(img)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        if outputs[0] is not None:
            outputs = rescale_boxes(outputs[0], img_size, (720, 1280))
        else:
            outputs = None
        
        # Save label
        with open(save_dir + '/' + img_name + '.txt', 'w') as f:
            if outputs is not None:
                for output in outputs:
                    f.write(" ".join(output.numpy().astype(str)) + '\n')
        # Save image in results dir
        if prediction_mode == 'detect':
            img = resize(img.squeeze(), 1280)
            img = img[:, 280:1000, :]
            img = transforms.ToPILImage()(img.squeeze().detach().cpu()).convert("RGB")
            img.save(save_dir + '/' + img_name + '.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="code/YOLOv3/config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="code/YOLOv3/config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou threshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    parser.add_argument("--prediction_mode", type=str, default='test')
    parser.add_argument("--attack_type", type=str, default='none')
    parser.add_argument("--loss_type", type=str, default='all')
    parser.add_argument("--eps", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--sign_grad", type=bool, default=False)
    parser.add_argument("--dist_type", type=str, default='l2')
    opt = parser.parse_args()
    print(opt)

    attack_name = opt.attack_type
    if opt.attack_type != "none":
        attack_name += f'-loss={opt.loss_type}-eps={opt.eps}'
        if opt.attack_type == "PGD":
            attack_name += f"-alpha={opt.alpha}-iter={opt.iterations}-sign_grad={opt.sign_grad}"

    weights_path = opt.weights_path.replace("checkpoints/", "ckpt=").replace('/', '-')

    if opt.prediction_mode == 'test':
        save_dir = f"results/test-conf={opt.conf}-{attack_name}-{weights_path}"
        conf_thres = opt.conf
    elif opt.prediction_mode == 'detect':
        save_dir = f"results/detect-{attack_name}-{weights_path}"
        conf_thres = opt.conf
        # conf_thres = 0.8
    elif opt.prediction_mode == 'activations':
        save_file = f"output/activations-{attack_name}-{opt.dist_type}-{weights_path}"
    else:
        raise ValueError(f"Prediction mode {opt.prediction_mode} not recognized.")

    if opt.prediction_mode != 'activations':
        os.makedirs(save_dir, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    if (opt.prediction_mode == 'test') or (opt.prediction_mode == 'activations'):
        test_path = data_config["test"]
    elif opt.prediction_mode == 'detect':
        test_path = data_config["samples"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights 
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    if opt.prediction_mode == 'activations':
        hidden_distance_yolo(
            model=model, 
            path=test_path,
            img_size=opt.img_size,
            dist_type=opt.dist_type,
            save_dir=save_file,
            attack_type=opt.attack_type,
            eps=opt.eps
        )

    else:
        predict(
            model,
            path=test_path,
            conf_thres=conf_thres,
            nms_thres=opt.nms_thres,
            img_size=opt.img_size,
            batch_size=1,
            save_dir=save_dir,
            prediction_mode=opt.prediction_mode,
            attack_type=opt.attack_type,
            eps=opt.eps,
            alpha=opt.alpha,
            iterations=opt.iterations,
            sign_grad=opt.sign_grad,
            loss_type=opt.loss_type
        )
