import argparse

import matplotlib.pyplot as plt
import numpy as np

"""
training_type: FGSM/PGD/Normal
noise_type: [FGSM,Randn]
distance_type: l1/l2/cos
model: yolo/retina
files: fgsm file, random file

e.g.
    train_type='FGSM'
    noise_type=['Randn','FGSM']
    distance_type='cos'
    model='retina'
    path1= 'results/retina-get_acts-Randn-FGSM-epochs29-batch8-eps2-mix_thres0.5--best_21.pt-eps2hiddenDist.txt'
    path2='results/retina-get_acts-FGSM-FGSM-epochs29-batch8-eps2-mix_thres0.5--best_21.pt-eps2hiddenDist.txt'
"""

retina_layer_size = {'conv1': 856064, 'layer1': 3424256, 'layer2': 1712128, 'layer3': 856064, 'layer4': 428032,
                     'P3': 856064, 'P3_cls_0': 856064, 'P3_cls_1': 856064, 'P3_cls_2': 856064, 'P3_cls_3': 856064,
                     'P3_reg_0': 856064, 'P3_reg_1': 856064, 'P3_reg_2': 856064, 'P3_reg_3': 856064,
                     'P4': 214016, 'P4_cls_0': 214016, 'P4_cls_1': 214016, 'P4_cls_2': 214016, 'P4_cls_3': 214016,
                     'P4_reg_0': 214016, 'P4_reg_1': 214016, 'P4_reg_2': 214016, 'P4_reg_3': 214016,
                     'P5': 53504, 'P5_cls_0': 53504, 'P5_cls_1': 53504, 'P5_cls_2': 53504, 'P5_cls_3': 53504,
                     'P5_reg_0': 53504, 'P5_reg_1': 53504, 'P5_reg_2': 53504, 'P5_reg_3': 53504,
                     'P6': 15360, 'P6_cls_0': 15360, 'P6_cls_1': 15360, 'P6_cls_2': 15360, 'P6_cls_3': 15360,
                     'P6_reg_0': 15360, 'P6_reg_1': 15360, 'P6_reg_2': 15360, 'P6_reg_3': 15360,
                     'P7': 3840, 'P7_cls_0': 3840, 'P7_cls_1': 3840, 'P7_cls_2': 3840, 'P7_cls_3': 3840,
                     'P7_reg_0': 3840, 'P7_reg_1': 3840, 'P7_reg_2': 3840, 'P7_reg_3': 3840}


# def compare_noise_type(*args):
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("--train_type", type=str, default="Normal",
#                         help="FGSM/PGD/Normal")
#     parser.add_argument("--noise_type", '--list', nargs='+', default=['FGSM'],
#                         help="path to data config file")
#     parser.add_argument('--distance_type', type=str, default="l2",
#                         help="l1/l2/cos")
#     parser.add_argument('--model', type=str, default="retina",
#                         help="retina/yolo")
#     # parser.add_argument("--files", '--list', nargs='+', help="path to the .txt file")
#
#     parser = parser.parse_args()
#     print(parser)
#     color = ['b', 'r', 'g', 'c', 'm']
#     files = []
#     for file in args:
#         files.append(file)
#
#     fig, ax = plt.subplots(figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
#     for idx, file in enumerate(files):
#         with open(file, 'r') as f:
#             lines = f.readlines()
#         layers = lines[0].split(',')[:-1]
#         acts = lines[1].split(',')[:-1]
#         assert len(acts) == len(layers)
#         activations = []
#         for act in acts:
#             activations.append(float(act))
#
#         ax.plot(layers, activations, c=color[idx], label=f"{parser.train_type} on {parser.noise_type[idx]} noise")
#     if parser.model =='retina':
#         ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
#         ax2.set_ylabel('layer size')  # we already handled the x-label with ax1
#         ax2.plot(list(retina_layer_size.values()), 'k:',label="layer size")
#         ax2.tick_params(axis='y')
#
#     fig.legend(loc='best')
#     plt.title(f"{parser.distance_type}distance-layers on {parser.model}")
#     plt.xlabel('layers')
#     plt.ylabel(f"{parser.distance_type} distance")
#     plt.xticks(rotation=90)
#     plt.savefig(f"results/dist-layer-{parser.model}-{parser.train_type}-{parser.distance_type}.png")


def draw(title, ylabel, legends, files, model='retina', normalize_acts=False, plt_layersize=False):
    color = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
    for idx, file in enumerate(files):
        with open(file, 'r') as f:
            lines = f.readlines()
        layers = lines[0].split(',')[:-1]
        acts = lines[1].split(',')[:-1]
        assert len(acts) == len(layers)
        activations = []
        for act in acts:
            activations.append(float(act))
        if normalize_acts:
            activations = np.asarray(activations)
            metric_range = [np.min(activations), np.max(activations)]
            activations /= np.max(activations)
            ax.plot(layers, activations, c=color[idx], label=legends[idx] + f" x {int(metric_range[1])}")
        ax.plot(layers, activations, c=color[idx], label=legends[idx])
        ax.set_ylabel(ylabel,fontsize='large')  # we already handled the x-label with ax1
    if model == 'retina' and plt_layersize:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('layer size')  # we already handled the x-label with ax1
        ax2.plot(list(retina_layer_size.values()), 'k:', label="layer size")
        ax2.tick_params(axis='y')

    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.legend(loc='best',fontsize='large')
    plt.title(title)
    plt.xlabel('layers', fontsize='large')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"results/dist-layer-{title.replace(' ', '')}.png")
    print(f"save to: results/dist-layer-{title.replace(' ', '')}.png")

if __name__ == "__main__":
    # path to the .txt file, you can path multiple paths
    # title = 'compare distance metrics:normal retina on random noise'
    # ylabel = 'distance(normalized)'
    Normal_onRandn_cmd2 = 'results/retina-get_acts-Randn-Final--Normal-best-from-epoch60.pt-eps16cmd2-hiddenDist.txt'
    Normal_onRandn_l2 = 'results/retina-get_acts-Randn-Final--Normal-best-from-epoch60.pt-eps16l2-hiddenDist.txt'
    Normal_onRandn_l1 = 'results/retina-get_acts-Randn-Final--Normal-best-from-epoch60.pt-eps16l1-hiddenDist.txt'
    Normal_onRandn_cos = 'results/retina-get_acts-Randn-Final--Normal-best-from-epoch60.pt-eps16cos-hiddenDist.txt'
    Normal_onFGSM_l2 = 'results/retina-get_acts-FGSM-Final--Normal-best-from-epoch60.pt-eps2l2-hiddenDist.txt'
    FGSM_onRandn_l2 = 'results/retina-get_acts-Randn-Final--FGSM-best-from-epoch29.pt-eps16l2-hiddenDist.txt'
    FGSM_onFGSM_l2 = 'results/retina-get_acts-FGSM-Final--FGSM-best-from-epoch29.pt-eps2l2-hiddenDist.txt'
    IRL_onRandn_l2 = 'results/retina-get_acts-Randn-Normal-epochs30-batch4-irl_fnl_layers-resnet4-fpn4-5-alpha0.8-beta0.2-gamma1.0-loss-type6-noise-type-random_noise-alt0--ckpt_26.pt-eps16l2-hiddenDist.txt'
    model = 'retina'
    # draw(title, ylabel, ['l2', 'l1', 'CMD2'], [Normal_onRandn_l2, Normal_onRandn_l1, Normal_onRandn_cmd2], model, normalize_acts=False)


    # title = 'Retina: distance between hidden layer activation on clean and noised data'
    # ylabel = 'L2 distance'
    # draw(title,ylabel,['Random noise', 'FGSM attack'], [Normal_onRandn_l2,Normal_onFGSM_l2])

    # title = 'distance of activations between clean and noisy data on RetinaNet'
    # ylabel = 'L2 distance'
    # draw(title,ylabel,['IRL training on random noise', 'FGSM training on random noise', 'Normal model on random noise'], [IRL_onRandn_l2, FGSM_onRandn_l2, Normal_onRandn_l2])


    title = 'distance of activations between clean and noisy data on RetinaNet'
    ylabel = 'L2 distance'
    draw(title, ylabel,
         ['FGSM training on FGSM attack','FGSM training on random noise'],
         [FGSM_onFGSM_l2, FGSM_onRandn_l2])
