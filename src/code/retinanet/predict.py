from __future__ import division
import sys
sys.path.append('code/YOLOv3/')
sys.path.append("code/shared/attacks")
sys.path.append("code/shared")
sys.path.append("code/shared/plots")
from perturbations import *
from attack_utils import *
from layer_distance import hidden_distance_retina

from torch.autograd import Variable
from perturbations import *
from attack_utils import *
from evaluate.test import evaluate_test
from evaluate.print_detections import print_detections
import os
import sys
import argparse
import tqdm
import matplotlib.pyplot as plt
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter
from retinanet.utils import parse_data_config
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from retinanet import csv_eval
import torch.nn.functional as F


def predict(model, path, class_names, save_dir, prediction_mode, attack_type, eps,
            alpha, iterations, sign_grad=False,loss_type='all'):
    # Get dataloader
    dataset = CSVDataset(train_file=path, class_list=class_names,
                         transform=transforms.Compose([Resizer()]))

    # This implementation Can only evaluate with batch_size=1
    sampler = AspectRatioBasedSampler(dataset, batch_size=1, drop_last=False)
    dataloader = DataLoader(dataset, num_workers=1, collate_fn=collater, batch_sampler=sampler)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for batch_i, data in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        img = data['img']
        targets = data['annot']
        scale = data['scale'][0]
        img_name = data['name'][0]
        img = Variable(img.type(Tensor), requires_grad=False)
        model.train()
        model.training = True
        try:
            if attack_type != "Normal":
                if attack_type == "FGSM":
                    img = generate_fgsm_image(model, img, targets, eps, model_type="retina",loss_type=loss_type)
                elif attack_type == "PGD":
                    img = generate_pgd_image(model, img, targets, alpha, eps, iterations, model_type="retina",
                                                     plot_losses=False, sign_grad=sign_grad)
                    # plt.figure()
                    # plt.plot(losses)
                    # plt.title('PGD loss on adversarial sample w.r.t iteration')
                    # plt.xlabel('iteration')
                    # plt.ylabel('loss')
                    # plt.savefig(f"output/{img_name}-{alpha}-{sign_grad}.png")

                # elif attack_type == "RFSGM":
                #     losses, cos = observe_fgsm_loss(model,img, targets, model_type="retina")
                #     print(cos)
                #     im = plt.imshow(losses, cmap='hot')
                #     plt.colorbar(im, orientation='horizontal')
                #     plt.show()
                elif attack_type == "Randn":
                    img = generate_noisy_image(img, noise=eps)
                else:
                    raise ValueError("Unsupported attack type")
        except:
            print('an error occured')
            continue
        with torch.no_grad():
            model.eval()
            scores, classification, anchors = model(img)

        if prediction_mode == 'output_bbx':
            anchors = anchors / scale
        # Save label
        with open(save_dir + '/' + img_name + '.txt', 'w') as f:
            for i in range(scores.shape[0]):
                if prediction_mode == "detect" and scores[i].item()<0.5:
                    continue
                f.write(
                    " ".join(anchors[i].detach().cpu().numpy().astype(str)) + " " + str(scores[i].item()) + " " + str(
                        scores[i].item()) + " " + str(classification[i].item()) + "\n")
        if prediction_mode == 'detect':
            img = transforms.ToPILImage()(img.squeeze().detach().cpu()).convert("RGB")
            img.save(save_dir + '/' + img_name + '_' + attack_type + '.png')


def _test(model, path, class_names, attack_eps=2, alpha=0.5, iteration=10, attack_type='Normal', batch_size = 8):
    dataset    = CSVDataset(train_file=path, class_list=class_names, transform=transforms.Compose([Resizer()]))
    print("\n------Testing on testset-------")
    printmsg = f"attack_type: {attack_type}"
    if attack_type != 'Normal':
        printmsg += f" | attack_epsilon: {attack_eps}"
        if attack_type == 'PGD':
            printmsg += f" | alpha: {alpha} | iteration: {iteration}"
    print(printmsg)
    AP, mAP = csv_eval.evaluate(dataset, model, perturbed = attack_type, _epsilon = attack_eps, _alpha=alpha, _iteration=iteration)
    print('Perturbed Test Set AP: {} | mAP: {}'.format(AP, mAP))


def get_activations(model, path, class_names, dist_type, attack_type = 'Randn', eps = 2, batch_size = 4):
    dataset = CSVDataset(train_file=path, class_list=class_names, transform=transforms.Compose([Resizer()]))
    sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, num_workers=1, collate_fn=collater, batch_sampler=sampler)
    Tensor  = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print("Getting activations for distance type:", dist_type, "with attack_type: ", attack_type, "and epsilon: ", eps)
    model.eval()
    final_activations = {}
    iters = 0
    for batch_i, data in enumerate(tqdm.tqdm(dataloader)):
        imgs = data['img']
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        acts = model(imgs, send_activations = True)
        
        if attack_type=='Randn':
            noi_imgs = generate_noisy_image(imgs, eps)
        elif attack_type=='FGSM':
            model.train()
            model.training = True
            noi_imgs = generate_fgsm_image(model, imgs, data['annot'], eps, model_type="retina")
            model.training = False
            model.eval()

        noi_acts = model(noi_imgs, send_activations = True)
        for layer, clean_acts in acts.items():
            noisy_acts = noi_acts[layer]
            if dist_type=='l2':
                dist = torch.norm(clean_acts - noisy_acts, p = 2, dim =(1,2,3)).mean()
            elif dist_type=='l1':
                dist = torch.norm(clean_acts - noisy_acts, p = 1, dim =(1,2,3)).mean()
            elif dist_type=='cos':
                dist = cosine_distance(clean_acts, noisy_acts).mean()
            
            final_activations[layer] = ( final_activations[layer] + dist.item() ) if layer in final_activations.keys() else dist.item()

        iters += 1
    return { layer: (act / iters) for layer, act in final_activations.items()}
                    

def cosine_distance(x, y):
    x_flat = torch.flatten(x,start_dim=1)
    y_flat = torch.flatten(y,start_dim=1)
    return F.cosine_similarity(x_flat, y_flat, dim=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_config", type=str, default="data/retina_label/custom.data",
                        help="path to data config file")
    parser.add_argument("--model", type=str, default="code/retinanet/checkpoints/retina_ckpt_448_800_9.pt",
                        help="path to weights file")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--prediction_mode", type=str, default='test', help="test/detect/get_acts/out_bbx")
    parser.add_argument("--attack_type", type=str, default='Normal', help='Normal/FGSM/PGD/Randn')
    parser.add_argument("--sign_grad", type=bool, default=False)
    parser.add_argument("--eps", type=int, default=2, help="eps of FGSM or Random noise")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--distance_type", type=str, default='l2', help="l2|l1|cos|cmd2")
    parser.add_argument("--loss_type", type=str, default='all', help="all|loc|cls")
    opt = parser.parse_args()
    print(opt)
    save_dir = f"results/retina-{opt.prediction_mode}-{opt.attack_type}-{opt.model.split('/')[-2]}--{os.path.basename(opt.model)}"
    if opt.attack_type == 'FGSM'or 'Randn':
        save_dir += f"-eps{opt.eps}"
    elif opt.attack_type == 'PGD':
        save_dir += f"-eps{opt.eps}-alpha{opt.alpha}-iter{opt.iterations}"
    if opt.loss_type !='all':
        save_dir += f"-{opt.loss_type}"
    print(f"save to {save_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)

    if opt.prediction_mode == 'detect':
        test_path = data_config["smallsample"]
    else:
        test_path = data_config["test"]

    class_names = data_config["names"]

    # Initiate model
    retinanet = torch.load(opt.model).to(device)
    retinanet = torch.nn.DataParallel(retinanet).to(device)
    
    print("Running for: ", opt.model)
    if opt.prediction_mode == 'test':
        _test(retinanet, data_config["test"], class_names, opt.eps, opt.alpha, opt.iterations, opt.attack_type)
   
    if opt.prediction_mode == 'get_acts':
       acts = hidden_distance_retina(retinanet, test_path, class_names, opt.distance_type, save_dir+f"{opt.distance_type}-hiddenDist.txt", attack_type = opt.attack_type, eps = opt.eps, batch_size = opt.batch_size)
       # acts = get_activations(retinanet, test_path, class_names, opt.distance_type, attack_type = opt.attack_type, eps = opt.eps, batch_size = opt.batch_size)
       print("The final activations: ", acts)
    print(f"saving at {save_dir}")
    if opt.prediction_mode == 'output_bbx' or opt.prediction_mode == 'detect':
        os.makedirs(save_dir, exist_ok=True)
        predict(
          retinanet,
          path=test_path,
          class_names=class_names,
          save_dir=save_dir,
          prediction_mode=opt.prediction_mode,
          attack_type=opt.attack_type,
          eps=opt.eps,
          alpha=opt.alpha,
          iterations=opt.iterations,
          sign_grad=opt.sign_grad,
          loss_type=opt.loss_type
        )
    
    if opt.prediction_mode == 'detect':
        print_detections(save_dir)
