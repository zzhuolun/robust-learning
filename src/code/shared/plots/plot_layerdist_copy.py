import numpy as np
import matplotlib.pyplot as plt
import argparse
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

def main(*args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_type", type=str, default="Normal",
                        help="FGSM/PGD/Normal")
    parser.add_argument("--noise_type", '--list', nargs='+', default=['FGSM', 'Noise'],
                        help="path to data config file")
    parser.add_argument('--distance_type', type=str, default="l2",
                        help="l1/l2/cos")
    parser.add_argument('--model', type=str, default="retina",
                        help="retina/yolo")
    # parser.add_argument("--files", '--list', nargs='+', help="path to the .txt file")

    parser = parser.parse_args()
    print(parser)
    color=['b','r','g','c','m']
    files=[]
    for file in args:
        files.append(file)

    train_type = ['Normal', 'Normal']

    fig, ax = plt.subplots(figsize=(12, 8), dpi= 80, facecolor='w', edgecolor='k')
    for idx, file in enumerate(files):
        with open(file,'r') as f:
            lines = f.readlines()
        layers = lines[0].split(',')[:-1]
        acts = lines[1].split(',')[:-1]
        assert len(acts)==len(layers)
        activations = []
        new_layers = []
        for i, act in enumerate(acts):
            if not layers[i].startswith('shortcut'):
                new_layers.append(layers[i])
                activations.append(float(act))
        ax.plot(new_layers, activations, c=color[idx], label=f"{train_type[idx]} Training on {parser.noise_type[idx]} Attack")

    legend = ax.legend(loc='best')
    plt.title(f"Distance of activations between clean and noisy test data on YOLO")
    plt.xlabel('layers')
    plt.ylabel(f"{parser.distance_type} distance")
    plt.xticks(rotation=90)
    plt.savefig(f"results/dist-layer-{parser.model}-{parser.train_type}-{parser.distance_type}-{parser.noise_type}.png")


if __name__ == "__main__":
    # path to the .txt file, you can path multiple paths
    path1 = 'output/activations-FGSM-loss=all-eps=2-l2-ckpt=YOLOv3-Normal-random_noise-epochs=50-batch=4-img_size=608-multi_scale=True-ckpt_20.pth'
    path2 = 'output/activations-random_noise-loss=all-eps=16-l2-ckpt=YOLOv3-Normal-random_noise-epochs=50-batch=4-img_size=608-multi_scale=True-ckpt_20.pth'
    main(path1, path2)