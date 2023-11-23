# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

import torchvision.transforms as T

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize

from util.box_ops import box_cxcywh_to_xyxy
import matplotlib.pyplot as plt
import numpy as np

import random


# CLASSES = [
#     'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
#     'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
#     'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#     'toothbrush'
# ]
CLASSES = ["N/A", "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"]

CLASS_2_COLOR = {
    "Person" : [0.0, 1.0, 0.0],
    "Rider" : [0.0, 0, 1.0],
    "Car" : [1.0, 1.0, 0.0],
    "Truck" : [1.0, 0, 1.0],
    "Bus" : [1.0, 0.0, 0.5],
    "Train" : [1.0, 0.5, 0.0],
    "Motorcycle" : [0.0, 0.5, 1.0],
    "Bicycle" : [0.0, 1.0, 1.0]
}

def rescale_bboxes(out_bbox, size, device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
    return b

def visualize(samples, output, targets, num_samples, device, output_dir, ids):
    # iterate thru batch 
    # print(samples)
    for i in range(len(samples)):
        pil_img = samples[i]
        target = targets[i]
        id = ids[i]
        
        boxes = output[i]
        pil_img = pil_img.resize(boxes.size)

        bboxes = []
        prob = []
        labels = []
        for score, label, box in zip(boxes.extra_fields['scores'], boxes.extra_fields['labels'], boxes.bbox):
            if score > 0.7:
                prob.append(score)
                labels.append(label)
                bboxes.append(box)

        has_obj = False

        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        for p, cl, (xmin, ymin, xmax, ymax) in zip(prob, labels, bboxes):
            clas = CLASSES[cl]
            color = CLASS_2_COLOR[clas] 
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=color, linewidth=3))       
            
        for cl, (xmin, ymin, xmax, ymax) in zip(target.extra_fields['labels'], target.bbox):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color="r", linewidth=1))

        plt.axis('off')
        plt.savefig(os.path.join(output_dir, str(id)+ ".png"))
    plt.close('all')

def compute_on_dataset(dataset, model, data_loader, device, output_dir=None, vis=True):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")

    for i, batch in enumerate(tqdm(data_loader)):
        ori_imgs, images, targets, image_ids = batch
        images = images.to(device)

        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]            

        if vis and i % 10 == 0:
            visualize(ori_imgs, output, targets, len(batch), device, output_dir, image_ids)
      
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )

        i += 1
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]

    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    
    start_time = time.time()
    predictions = compute_on_dataset(dataset, model, data_loader, device, output_dir=output_folder, vis=True)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    print("prediction for id 25: " + str(predictions[25]))
    for l, b in zip(predictions[25].extra_fields['labels'], predictions[25].bbox):
        if l == 1:
            print("label: " + str(l) + " bbox: " + str(b))

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
