from __future__ import division

import sys

sys.path.append('code/YOLOv3/')

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

sys.path.append('/nfs/students/summer-term-2020/project-3/src/code/shared/attacks')
from perturbations import *
from attack_utils import *
from adversarial_train import mix_batch as mix_batch_adv

sys.path.append('/nfs/students/summer-term-2020/project-3/src/code/shared/representation_learning')
from irl import IRL


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, attack_type=None, alpha=None, eps=None, iterations=None):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (paths, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        if attack_type == "FGSM":
            imgs = generate_fgsm_image(model, imgs, targets, eps, model_type="YOLO")
        elif attack_type == "PGD":
            imgs = generate_pgd_image(model, imgs, targets, alpha, eps, iterations, model_type="YOLO")
        elif attack_type == "random_noise":
            imgs = generate_noisy_image(imgs, eps)

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrained_weights", type=str, default="code/YOLOv3/weights/darknet53.conv.74")
    parser.add_argument("--training_type", type=str, default='Normal', help="Normal, Adversarial or IRL")
    parser.add_argument("--attack_type", type=str, default='FGSM', help="FGSM, PGD or random_noise")
    parser.add_argument("--eps", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--irl_loss_type", type=int, default=6)
    parser.add_argument("--irl_alpha", type=float, default=0.5)
    parser.add_argument("--irl_beta", type=float, default=0.5)
    parser.add_argument("--irl_gamma", type=float, default=1.0)
    opt = parser.parse_args()
    print(opt)

    # Parameters parsing
    epochs = opt.epochs
    pretrained_weights = opt.pretrained_weights
    training_type = opt.training_type
    attack_type = opt.attack_type
    eps = opt.eps
    alpha = opt.alpha
    iterations = opt.iterations
    irl_loss_type = opt.irl_loss_type
    irl_alpha = opt.irl_alpha
    irl_beta = opt.irl_beta
    irl_gamma = opt.irl_gamma

    # Parameters definition
    batch_size = 4
    img_size = 608
    saved_optim = None
    gradient_accumulations = 2
    model_def = "code/YOLOv3/config/yolov3-custom.cfg"
    data_config = "code/YOLOv3/config/custom.data"
    n_cpu = 8
    checkpoint_interval = 1
    evaluation_interval = 1
    avg_layers = False
    mix_thre = 0.5
    irl_epoch_interval = 1
    multiscale_training = True
    start_epoch = 0
    regularized_layers = [74, 75, 76, 77]

    # Training name 
    training_name = f"{training_type}-{attack_type}-epochs={epochs}-batch={batch_size}-img_size={img_size}-multi_scale={multiscale_training}"
    if training_type != "Normal":
        training_name += f"-eps={eps}"
        if attack_type == "PGD":
            training_name += f"-alpha={alpha}-iter={iterations}"
        if training_type == "Adversarial":
            training_name += f"-mix_thres={mix_thre}"
        elif (training_type == 'IRL'):
            training_name += f"-loss_type={irl_loss_type}-irl_alpha={irl_alpha}-irl_beta={irl_beta}-irl_gamma={irl_gamma}-interval={irl_epoch_interval}-reg_layers={str(regularized_layers)}-avg_layers={avg_layers}"       
    print(training_name)

    logger = Logger(f"logs/{training_name}") 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Found device: ', device)

    os.makedirs(f"checkpoints/YOLOv3/{training_name}", exist_ok=False) 

    # Get data configuration
    data_config = parse_data_config(data_config)
    train_path = data_config["train"] 
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if pretrained_weights:
        if pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(pretrained_weights))
        else:
            model.load_darknet_weights(pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, img_size=img_size, augment=True, multiscale=multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())
    
    if saved_optim is not None:
        optimizer.load_state_dict(torch.load(saved_optim))

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    if training_type == 'IRL':
        if attack_type == 'random_noise':
            irl_obj = IRL(noise_types=['random_noise'], model_type='YOLO', loss_type=irl_loss_type, epsilon=eps)
        else:
            irl_obj = IRL(noise_types=['adversarial'], model_type='YOLO', loss_type=irl_loss_type, adv_attack_type=attack_type, 
                        epsilon=eps, alpha=alpha, iterations=iterations)

    print('Starting training.')
    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            if (training_type == 'IRL') and ((epoch + 1) % irl_epoch_interval == 0):
                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)
                # Clean loss
                loss_loc, loss_cls, loss_conf, clean_activations = model(imgs, targets, regularized_layers=regularized_layers)
                clean_loss = loss_loc + loss_cls + loss_conf

                # Get noisy and distance loss
                noise_loss, dist_loss = irl_obj.compute_losses(model, imgs, targets, clean_activations, epoch, batch_i, regularized_layers, training_name, avg_layers)

                # IRL loss
                if irl_gamma == "log":
                    loss = irl_alpha * clean_loss + irl_beta * noise_loss + np.log10(epochs + 1) * dist_loss
                else:
                    loss = irl_alpha * clean_loss + irl_beta * noise_loss + irl_gamma * dist_loss


            elif training_type == 'Adversarial':
                input_batch = mix_batch_adv(model, imgs, targets, imgs.shape[0], epsilon=eps, alpha=alpha,
                                    iterations=iterations, mix_thre=mix_thre, attack_type=attack_type)

                input_batch = Variable(input_batch.to(device))
                targets = Variable(targets.to(device), requires_grad=False)

                loss_loc, loss_cls, loss_conf, _ = model(input_batch, targets)
                loss = loss_loc + loss_cls + loss_conf

            elif (training_type == 'Normal') or ((training_type == 'IRL') and ((epoch + 1) % irl_epoch_interval != 0)):
                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)

                loss_loc, loss_cls, loss_conf, _ = model(imgs, targets)
                loss = loss_loc + loss_cls + loss_conf

            else:
                raise Exception("Unrecognized training combination.")

            loss.backward()

            if batches_done % gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- epoch: {epoch}, mAP {AP.mean()}")

            print("\n---- Evaluating Model with Attack ----")
            # Evaluate the model on the perturbed validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=img_size,
                batch_size=8,
                attack_type=attack_type,
                eps=eps,
                alpha=alpha,
                iterations=iterations
            )
            evaluation_metrics = [
                ("val_precision_attack", precision.mean()),
                ("val_recall_attack", recall.mean()),
                ("val_mAP_attack", AP.mean()),
                ("val_f1_attack", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- epoch: {epoch}, Attack mAP {AP.mean()}")

        if epoch % checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/YOLOv3/{training_name}/ckpt_%d.pth" % (epoch))
            torch.save(optimizer.state_dict(), f"checkpoints/YOLOv3/{training_name}/optim_%d.pth" % (epoch))