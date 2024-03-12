# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
import os
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference

from maskrcnn_benchmark.structures.image_list import ImageList

from .visualization import plot_one

import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from torchviz import make_dot, make_dot_from_trace


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])

            # print(k)
            # print(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    # training loss for plotting 
    training_losses = []
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        # track training loss
        training_losses.append(losses_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    plot_one(training_losses, "Per Forward Pass Training Loss", "Epoch", "Loss", "Training", output_dir)

def do_da_train(
    model,
    source_data_loader,
    target_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, ((orisource_images, source_images, source_targets, idx1), (orisource_images, target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        scheduler.step()
        images = (source_images+target_images).to(device)
        targets = [target.to(device) for target in list(source_targets+target_targets)]

        # print("================================Passing to Model================================")
        # print("images: " + str(images.image_sizes))
        gt_bboxes = []
        gt_labels = []
        img_metas = []
        for i in range(len(targets)):
            target = targets[i]
            # print("target bboxes: " + str(target.bbox))
            # print("target mode: " + str(target.mode))
            # print("target labels: " + str(target.extra_fields))

            to_add = {
                'ori_shape' : images.image_sizes[i],
                'img_shape' : images.image_sizes[i],
                'pad_shape' : images.image_sizes[i],
                'scale_factor' : np.array([1.0, 1.0, 1.0, 1.0]),
            }

            img_metas.append(to_add)
            gt_bboxes.append(target.bbox)
            gt_labels.append(target.extra_fields['labels'])

        pred_loss_dict = model(images, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

        # print(pred_loss_dict)

        loss_dict = {}
        for key in pred_loss_dict:
            if "loss" in key:
                if isinstance(pred_loss_dict[key], list):
                    loss_dict[key] = sum(pred_loss_dict[key])

        losses = 0
        for key in loss_dict:
            loss = loss_dict[key]
            if loss.dim() == 0:
                losses += loss
            else:
                losses += torch.squeeze(loss, 0)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)

        losses_reduced = 0
        for key in  loss_dict_reduced:
            loss_list = loss_dict_reduced[key]
            if loss_list.dim() == 0:
                losses_reduced += loss_list
            else:
                losses_reduced += torch.squeeze(loss_list, 0)

        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter-1:
            checkpointer.save("model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    
def plot_grad_flow(named_parameters, title):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and ("box" in n):
            # print(n)
            # print(p.grad)
            if p.grad is not None:
                print(n)
                n = n[14:]
                n = n[:len(n) - 7]
                
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())
            else:
                print(n)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=5)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title(title)
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("gradient_plot_temp0.6HW0.55.png")


def do_mask_da_train(
    model, model_teacher,
    source_data_loader,
    target_data_loader,
    masking,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg,
    checkpointer_teacher,
    data_loader_val
):
    from maskrcnn_benchmark.structures.image_list import to_image_list

    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    logger.info("with_MIC: On")

    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    logger.info("start iteration: " + str(start_iter) + " running to max iterations: " + str(max_iter))

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)

    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    model.train()
    model_teacher.eval()

    start_training_time = time.time()
    end = time.time()

    # track loss for plotting 
    training_losses = []
    masked_losses = []
    for iteration, ((ori_imgs, source_images, source_targets, idx1), (ori_imgs, target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        model.train()
        model_teacher.eval()
        
        data_time = time.time() - end
        arguments["iteration"] = iteration
        # print("===================iteration: " + str(iteration) + "===================")

        # print("SRC IMAGES! iter: " + str(iteration))
        # for img in source_images.tensors:
        #     print(img.shape)

        # print("TARGET IMAGES!")
        # for img in target_images.tensors:
        #     print(img.shape)

        source_images = source_images.to(device)
        target_images = target_images.to(device)

        images = source_images+target_images
        targets = [target.to(device) for target in list(source_targets+target_targets)]

        # mask the target image
        masked_target_images = masking(target_images.tensors.clone().detach()).detach()

        # update teacher weights 
        model_teacher.update_weights(model, iteration)

        # target goes to the teacher 
        print("######################################PASSING TARGET IMGS TO TEACHER######################################")
        target_output = model_teacher(target_images)

        # print("TEACHER TARGET IMG OUTPUT: " + str(target_output))

        # process output to get pseudo masks 
        target_pseudo_labels, pseudo_masks = process_pred2label(target_output, threshold=cfg.MODEL.PSEUDO_LABEL_THRESHOLD)

        # student gets source and target
        print("######################################PASSING SOURCE AND TARGET IMGS TO STUDENT######################################")
        record_dict = model(images, targets)
        
        # apply pseudo label on masked images
        if len(target_pseudo_labels)>0:
            print("pseudo masks: " + str(pseudo_masks))
            print("target pseudo labels: " + str(target_pseudo_labels))
            masked_images = masked_target_images[pseudo_masks]
            masked_taget = target_pseudo_labels

            # print("BEFORE PASSING IN MASKED IMAGES!")
            sizes = []
            # for img in masked_images:
            #     print("masked image shape: " + str(img.shape))
            for img in masked_taget:
                sizes.append((img.size[1], img.size[0]))
                # print("masked target shape: " + str(img.size))

            # print(masked_taget)
            
            # convert to image list with same size of masked target 
            masked_images = ImageList(masked_images, sizes)
            # print(masked_images.image_sizes)

            # student gets masked images 
            print("######################################PASSING MASKED TARGET IMGS TO STUDENT######################################")
            masked_loss_dict = model(masked_images, masked_taget, use_pseudo_labeling_weight=cfg.MODEL.PSEUDO_LABEL_WEIGHT, with_DA_ON=False)
            
            new_record_all_unlabel_data = {}
            for key in masked_loss_dict.keys():
                new_record_all_unlabel_data[key + "_mask"] = masked_loss_dict[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

        # weight losses
        loss_dict = {}
        ml = 0
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_box_reg_mask" or key == "loss_rpn_box_reg_mask":
                    # pseudo bbox regression <- 0
                    loss_dict[key] = record_dict[key] * 0
                elif key.endswith('_mask') and 'da' in key:
                    loss_dict[key] = record_dict[key] * 0
                elif key == 'loss_classifier_mask' or key == 'loss_objectness_mask':
                    loss_dict[key] = record_dict[key] * cfg.MODEL.PSEUDO_LABEL_LAMBDA
                    ml += record_dict[key] * cfg.MODEL.PSEUDO_LABEL_LAMBDA
                else:  # supervised loss
                    loss_dict[key] = record_dict[key] * 1
                    
        print("loss contrast: " + str(loss_dict['loss_contrast']))
        if len(target_pseudo_labels)>0:   
            print("loss contrast mask: " + str(loss_dict['loss_contrast_mask']))
            
        # dot = make_dot(loss_dict['loss_contrast'], params=dict(model.named_parameters()))
        # dot.format = 'png'
        # dot.render('mic_imitation_graph_contrast_loss')
        
        # dot = make_dot(loss_dict['loss_classifier'], params=dict(model.named_parameters()))
        # dot.format = 'png'
        # dot.render('mic_imitation_graph_classification_loss')

        if ml != 0:
            ml = ml.cpu().detach().numpy()

        masked_losses.append(ml)
        losses = sum(loss for loss in loss_dict.values())
        print("loss sum: " + str(losses))

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        # track losses 
        training_losses.append(losses_reduced.cpu().detach().numpy())

        optimizer.zero_grad()
        losses.backward()
        
        # if iteration % 1000 == 0 or iteration == max_iter:
        #     plot_grad_flow(model.named_parameters(), "Gradient flow for ROI Box Head Temp 0.6 HQ 0.55")
        
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration >= 15000:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            checkpointer_teacher.save("model_teacher_{:07d}".format(iteration), **arguments)

            dataset_name = cfg.DATASETS.TEST[0]
            output_folder_stu = os.path.join(cfg.OUTPUT_DIR, "inference_student_" + str(iteration), dataset_name)
            output_folder_tea = os.path.join(cfg.OUTPUT_DIR, "inference_teacher_" + str(iteration), dataset_name)
            mkdir(output_folder_stu)
            mkdir(output_folder_tea)

            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder_stu,
            )

            inference(
                model_teacher,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder_tea,
            )

        if iteration == max_iter-1:
            checkpointer.save("model_final", **arguments)
            checkpointer_teacher.save("model_final_teacher", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return  

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    plot_one(training_losses, "Per Forward Pass Training Loss", "Iterations", "Loss", "Training", cfg.OUTPUT_DIR)
    plot_one(masked_losses, "Per Forward Pass Masked Training Loss", "Iterations", "Loss", "Training", cfg.OUTPUT_DIR)


def process_pred2label(target_output, threshold=0.7):
    from maskrcnn_benchmark.structures.bounding_box import BoxList

    pseudo_labels_list = []
    masks = []
    for idx, bbox_l in enumerate(target_output):
        pred_bboxes = bbox_l.bbox.detach()

        labels = bbox_l.get_field('labels').detach()
        scores = bbox_l.get_field('scores').detach()

        # print(torch.max(scores))
        filtered_idx = scores>=threshold

        filtered_bboxes = pred_bboxes[filtered_idx]

        filtered_labels = labels[filtered_idx]

        new_bbox_list = BoxList(filtered_bboxes, bbox_l.size, mode=bbox_l.mode)

        new_bbox_list.add_field("labels", filtered_labels)

        domain_labels = torch.ones_like(filtered_labels, dtype=torch.uint8).to(filtered_labels.device)
        new_bbox_list.add_field("is_source", domain_labels)

        if len(new_bbox_list)>0:
            pseudo_labels_list.append(new_bbox_list)
            masks.append(idx)
    return pseudo_labels_list, masks