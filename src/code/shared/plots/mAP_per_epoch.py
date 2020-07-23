"""log file format:
clean_txt to save the epoch number and clean mAP
attacked_txt to save the epoch number and attacked mAP:
Format:
epochnumber,mAP
epochnumber,mAP
...
e.g.:
0,0.03\n
1,0.1\n
2,0.4\n
"""
import os
import numpy as np
import matplotlib.pyplot as plt
def main(logdir, flogdir, clean_txt, attacked_txt):

    with open(os.path.join(logdir, clean_txt), 'r') as log:
        logs = log.readlines()
    with open(os.path.join(logdir, attacked_txt), 'r') as log_attack:
        logs_a = log_attack.readlines()
    epochs = []
    mAPs = []
    epochs_a = []
    mAPs_a = []
    for line in logs:
        epoch = int(line.split(',')[0])
        mAP = float(line.split(',')[1])
        epochs.append(epoch)
        mAPs.append(mAP)
    for line in logs_a:
        epoch = int(line.split(',')[0])
        mAP = float(line.split(',')[1])
        epochs_a.append(epoch)
        mAPs_a.append(mAP)

    with open(os.path.join(flogdir, 'log.txt'), 'r') as log:
        logs = log.readlines()
    with open(os.path.join(flogdir, 'log_attack.txt'), 'r') as log_attack:
        logs_a = log_attack.readlines()
    fepochs = []
    fmAPs = []
    fepochs_a = []
    fmAPs_a = []
    for line in logs:
        epoch = int(line.split(',')[0])
        mAP = float(line.split(',')[1])
        fepochs.append(epoch)
        fmAPs.append(mAP)
    for line in logs_a:
        epoch = int(line.split(',')[0])
        mAP = float(line.split(',')[1])
        fepochs_a.append(epoch)
        fmAPs_a.append(mAP)

    plt.figure()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=80, facecolor='w', edgecolor='w')
    ax.plot(epochs, mAPs, 'r--', label='Normal model on clean data')
    ax.plot(epochs_a, mAPs_a, 'r:', label='Normal model on FGSM attacks')

    ax.plot(fepochs, fmAPs, 'b--', label='FGSM trained model on clean data')
    ax.plot(fepochs_a, fmAPs_a, 'b:', label='FGSM trained model on FGSM attacks')
    # ax.legend(loc='center right')  # , fontsize='x-large')

    # plt.plot(epochs, mAPs, 'r--', label='Normal model on clean data')
    # plt.plot(epochs_a, mAPs_a, 'r:', label='Normal model on FGSM attacks')
    #
    # plt.plot(fepochs, fmAPs, 'b--', label='FGSM trained model on clean data')
    # plt.plot(fepochs_a, fmAPs_a, 'b:', label='FGSM trained model on FGSM attacks')

    plt.legend()
    plt.title('retina: mAP with epochs')
    plt.xlabel('epochs')
    plt.ylabel('mAP')

    plt.savefig(f"results/{os.path.basename(logdir)}-mAP-epoch.jpg")

def get_map_data(path):
    with open(path,'r') as f:
        lines = f.readlines()
    for line in lines:
        print(line.split(',')[0])
    temp=[]
    for line in lines:
        # print(line.split(',')[1])
        temp.append(float(line.split(',')[1]))
    print(temp)

def get_dist_data(path):
    with open(path,'r') as f:
        lines = f.readlines()
    dist = lines[1]
    dist = dist.split(',')[:-1]
    dist = [float(i) for i in dist]
    maxi = max(dist)
    print(maxi)
    dist = [i/maxi for i in dist]
    for i in dist:
        print(i)
    # for line in lines:
    #     line = line.split(',')
    #     for i in line:
    #         print(i)
if __name__ == '__main__':
    os.chdir('/nfs/students/summer-term-2020/project-3/src/')
    logdir = 'checkpoints/retina/Normal-epochs40-batch8'
    flogdir = 'checkpoints/retina/FGSM-epochs29-batch8-eps2-mix_thres0.5'
    clean_txt = 'log.txt'
    attacked_txt = 'log_attack.txt'
    distancepath='/nfs/students/summer-term-2020/project-3/src/results/retina-get_acts-FGSM-Final--FGSM-best-from-epoch29.pt-eps2l1-hiddenDist.txt'
    # main(logdir,flogdir, clean_txt, attacked_txt)
    get_dist_data(distancepath)