from __future__ import division

import sys
sys.path.append('code/shared')

from evaluate.utils import *
from evaluate.datasets import *
from evaluate.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(path, results_path, iou_thres, batch_size, detect_mode, conf_thres_retina):

    # Get dataloader
    dataset = ListDataset(path, results_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
   
    for batch_i, (paths, predictions, targets) in enumerate(tqdm.tqdm(dataloader, desc="Computing batch statistics")):
        # Extract labels
        labels += targets[:, 1].tolist()
        
        if detect_mode:
            sample_metrics += get_batch_statistics_detection(predictions, targets, iou_threshold=iou_thres, conf_thres_retina=conf_thres_retina)
        else:
            sample_metrics += get_batch_statistics(predictions, targets, iou_threshold=iou_thres, conf_thres_retina=conf_thres_retina)

    # Concatenate sample statistics
    if detect_mode:
        true_positives, pred_scores, pred_labels, TP_iou, misclassifications, false_negatives, num_gt = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        return true_positives, TP_iou, misclassifications, false_negatives, num_gt

    # DEBUG - the length was found to be 0
    # print("The length is, ", len([np.concatenate(x, 0) for x in list(zip(*sample_metrics))]))
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def evaluate_test(results_path, detect_mode=False, conf_thres_retina=None, batch_size=8, data_config="data/custom.data", iou_thres=0.5, n_cpu=8):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(data_config)
    test_path = data_config["test"]
    class_names = load_classes(data_config["names"])

    if not detect_mode:

        print("Compute mAP...")

        precision, recall, AP, f1, ap_class = evaluate(
            path=test_path,
            results_path=results_path,
            iou_thres=iou_thres,
            batch_size=batch_size,
            detect_mode=detect_mode,
            conf_thres_retina=conf_thres_retina
        )

        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")

    else:
        true_positives, TP_iou, misc, false_negatives, num_gt = evaluate(
            path=test_path,
            results_path=results_path,
            iou_thres=iou_thres,
            batch_size=batch_size,
            detect_mode=detect_mode,
            conf_thres_retina=conf_thres_retina
        )

        print(f"Average number of misclassifications: {np.sum(misc) / misc.shape[0]}")
        print(f"Average number of True Positives: {len(np.where(true_positives)[0]) / misc.shape[0]}")
        print(f"Average number of False Positives: {len(np.where(1 - true_positives)[0]) / misc.shape[0]}")
        print(f"Average IoU of True Positives: {np.sum(TP_iou) / len(np.where(TP_iou)[0])}")    
        print(f"Misclassification rate: {np.sum(misc) / num_gt.sum()}")
        print(f"TP rate: {len(np.where(true_positives)[0]) / len(true_positives)}")
        print(f"FP rate: {len(np.where(1 - true_positives)[0]) / len(true_positives)}")
        print(f"FN rate: {false_negatives.sum() / num_gt.sum()}")  # number of ground truth = 83978
