import argparse
import collections
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from retinanet import csv_eval
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from retinanet.utils import parse_data_config
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append('code/retinanet/')
sys.path.append('code/shared/attacks')
sys.path.append('code/shared/representation_learning')
from perturbations import *
from attack_utils import *
from adversarial_train import *
from irl import *

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument("--data_config", type=str, default="data/retina_label/custom.data",
                        help="path to data config file")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument('--pretrained_model', type=str, default=None, help='load pretrained model')
    parser.add_argument('--optim_scheduler', type=str, default=None, help='load pretrained optimizer and scheduler')
    parser.add_argument("--attack_type", type=str, default="Normal",
                        help="type of adversarial attack; Normal or FGSM or PGD")
    parser.add_argument("--eps", type=str, default='2', help="epsilon value for FGSM")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sign_grad", type=bool, default=True,
                        help="whether use signed gradient and alpha=2.5*eps/iter in PGD")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--irl", type=int, default=0)
    parser.add_argument("--irl_noise_type", type=str, default='in_domain')
    parser.add_argument("--irl_loss_type", type=int, default=1)
    parser.add_argument("--irl_attack_type", type=str, default='fgsm',
                        help="type of attack to be implemented in small case")
    parser.add_argument("--irl_alpha", type=float, default='0.8')
    parser.add_argument("--irl_beta", type=float, default='0.2')
    parser.add_argument("--irl_gamma", type=float, default='1')
    parser.add_argument("--irl_alt", type=int, default=0)
    parser.add_argument("--irl_avg", type=int, default=0, help="Set true to average over all layers in irl distance loss")
    parser.add_argument("--mix_thre", type=float, default=0.5,
                        help="percentage of clean data in each mixed batch; range:[0,1], the larger, the more clean data there are in each batch")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--evaluation_attack_interval", type=int, default=3,
                        help="interval evaluations on validation set")
    parser.add_argument("--evalute_attacktype", type=str, default='FGSM', help="FGSM/Randn/Normal")
    parser = parser.parse_args(args)
    print(parser)
    eps = convert_eps(parser.eps)
    training_name = train_name(parser, eps)
    os.makedirs(f"checkpoints/retina/{training_name}", exist_ok=False)
    print(f"checkpoints stored as {training_name}")
    # Get data configuration
    data_config = parse_data_config(parser.data_config)
    train_path = data_config["train"]
    val_path = data_config["val"]
    class_names = data_config["names"]

    dataset_train = CSVDataset(train_file=train_path, class_list=class_names,
                               transform=transforms.Compose([Augmenter(), Resizer()]))

    dataset_val = CSVDataset(train_file=val_path, class_list=class_names,
                             transform=transforms.Compose([Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.n_cpu, collate_fn=collater, batch_sampler=sampler)

    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=False)
    # dataloader_val = DataLoader(dataset_val, num_workers=parser.n_cpu, collate_fn=collater, batch_sampler=sampler_val)
    if parser.pretrained_model:
        retinanet = torch.load(parser.pretrained_model)
    else:
        if parser.depth == 18:
            retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 34:
            retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 50:
            retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 101:
            retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
        elif parser.depth == 152:
            retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    use_irl = bool(parser.irl)
    irl_alt = bool(parser.irl_alt)
    irl_avg = bool(parser.irl_avg)
    if use_irl:
        irl_obj = IRL(noise_types=[parser.irl_noise_type], adv_attack_type=parser.irl_attack_type, model_type='retina',
                      loss_type=parser.irl_loss_type, epsilon=eps, alpha=parser.alpha,
                      iterations=parser.iterations)
        act_file_name = ('retina_fnl_layers-resnet4_loss-type' + str(
            parser.irl_loss_type) + '_' + parser.irl_noise_type + '_alt' + str(parser.irl_alt))
        act_file_name += f"-alpha{parser.irl_alpha}-beta{parser.irl_beta}-gamma{parser.irl_gamma}_activations.txt"
        print("Saving activations in: ", str(act_file_name))

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    if parser.optim_scheduler is not None:
        optim_scheduler = torch.load(parser.optim_scheduler)
        optimizer.load_state_dict(optim_scheduler['optimizer'])
        scheduler.load_state_dict(optim_scheduler['scheduler'])
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    print('Starting training.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):

            optimizer.zero_grad()
            batch_mixed = mix_batch(retinanet, data['img'], data['annot'], data['img'].shape[0],
                                    epsilon=eps, alpha=parser.alpha,
                                    mix_thre=parser.mix_thre, attack_type=parser.attack_type,
                                    model_type='retina', sign_grad=parser.sign_grad)
            if use_irl and (not irl_alt or epoch_num % 2 == 1):
                classification_loss, regression_loss, activations = retinanet(
                    [Variable(batch_mixed.to(device)), data['annot']], send_activations=True)
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                noise_loss, distance_loss = irl_obj.compute_losses(model=retinanet, images=data['img'],
                                                                   targets=data['annot'],
                                                                   activations=activations, epoch_num=epoch_num,
                                                                   batch_num=iter_num,
                                                                   training_name=act_file_name, avg_layers=irl_avg)
                regular_loss = classification_loss + regression_loss
                loss = parser.irl_alpha * regular_loss + parser.irl_beta * noise_loss + parser.irl_gamma * distance_loss
            else:
                classification_loss, regression_loss = retinanet([Variable(batch_mixed.to(device)), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            if iter_num % 500 == 0:
                if use_irl and (not irl_alt or epoch_num % 2 == 1):
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Noise Loss: {:1.5f} | Distance Loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(parser.irl_alpha * classification_loss),
                            float(parser.irl_alpha * regression_loss),
                            float(parser.irl_beta * noise_loss), float(parser.irl_gamma * distance_loss),
                            np.mean(loss_hist)))
                    del noise_loss
                    del distance_loss
                else:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss),
                            np.mean(loss_hist)))

            del classification_loss
            del regression_loss

        scheduler.step(np.mean(epoch_loss))

        if epoch_num % parser.checkpoint_interval == 0:
            torch.save(retinanet.module, f"checkpoints/retina/{training_name}/ckpt_{epoch_num}.pt")
            torch.save({'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()},
                       f"checkpoints/retina/{training_name}/optim_scheduler_{epoch_num}.pt")

        if epoch_num % parser.evaluation_interval == 0:
            print("\n------Evaluating model------")
            AP, mAP = csv_eval.evaluate(dataset_val, retinanet)
            print(
                'Epoch: {} | AP: {} | mAP: {}'.format(epoch_num, AP, mAP))
            # write logs of the model to log.txt, format: epoch number, mAP, AP per class
            print(
                f"{epoch_num},{mAP},{AP[0][0]},{AP[1][0]},{AP[2][0]},{AP[3][0]},{AP[4][0]},{AP[5][0]},{AP[6][0]},{AP[7][0]},{AP[8][0]},{AP[9][0]}\n")
            with open(f"checkpoints/retina/{training_name}/log.txt", 'a+') as log:
                log.write(
                    f"{epoch_num},{mAP},{AP[0][0]},{AP[1][0]},{AP[2][0]},{AP[3][0]},{AP[4][0]},{AP[5][0]},{AP[6][0]},{AP[7][0]},{AP[8][0]},{AP[9][0]}\n")

        # Evaluating the model on noise now
        if parser.evalute_attacktype and epoch_num % parser.evaluation_attack_interval == 0:
            print("\n-------Evaluating on noise-----")
            AP_n, mAP_n = csv_eval.evaluate(dataset_val, retinanet, perturbed=parser.evalute_attacktype,
                                            _epsilon=eps)
            print('Noise Epoch: {} | AP: {} | mAP: {}'.format(epoch_num, AP_n, mAP_n))
            with open(f"checkpoints/retina/{training_name}/log_attack.txt", 'a+') as log:
                log.write(
                    f"{epoch_num},{mAP_n},{AP_n[0][0]},{AP_n[1][0]},{AP_n[2][0]},{AP_n[3][0]},{AP_n[4][0]},{AP_n[5][0]},{AP_n[6][0]},{AP_n[7][0]},{AP_n[8][0]},{AP_n[9][0]}\n")


def convert_eps(parser_eps):
    if ',' in parser_eps:
        eps = (int(parser_eps.split(',')[0]), int(parser_eps.split(',')[1]))

    else:
        eps = int(parser_eps)
    return eps


def train_name(parser, eps):
    training_name = f"{parser.attack_type}-epochs{parser.epochs}-batch{parser.batch_size}"
    if parser.attack_type != "Normal":
        training_name += f"-eps{eps}"
        if parser.attack_type == "PGD":
            training_name += f"-alpha{parser.alpha}-iterations{parser.iterations}"
        training_name += f"-mix_thres{parser.mix_thre}"
    if parser.irl == 1:
        training_name += f"-irl_fnl_layers-resnet4-alpha{parser.irl_alpha}-beta{parser.irl_beta}-gamma{parser.irl_gamma}"
        training_name += f"-loss-type{parser.irl_loss_type}-noise-type-{parser.irl_noise_type}-alt{parser.irl_alt}-avg{parser.irl_avg}"
        if parser.irl_noise_type == 'adversarial':
            training_name += f"-attack-type-{parser.irl_attack_type}-eps{eps}-alpha{parser.alpha}"
    return training_name


if __name__ == '__main__':
    main()
