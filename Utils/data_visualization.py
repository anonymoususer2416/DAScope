import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import os

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# classes in the dataset 
CLASSES = ["N/A", "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"]

# from DETR.detr.engine import visualize

def rescale_bboxes(out_bbox, size, device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=device)
    return b

def save_tensor_img(tensor_img, name):
    # to convert tensor to PIL image 
    transform = T.ToPILImage()
    pil_img = transform(tensor_img)
    plt.imshow(pil_img)
    plt.savefig(name)
    plt.close('all')

def visualize(samples, outputs, targets, num_samples, device, output_dir, needs_transform=False, confidence_thresh=0.7):
    # to convert tensor to PIL image 
    if needs_transform:
        transform = T.ToPILImage()

    # separate output
    batch_pred_boxes = outputs['pred_boxes']
    batch_pred_logits = outputs['pred_logits']

    # iterate thru batch 
    for i in range(num_samples):
        target = targets[i]
        ax = plt.gca()

        # print(target)
        target_boxes = target['boxes']
        sample = samples[i]
        pred_boxes = batch_pred_boxes[i]
        pred_logits = batch_pred_logits[i]
        
        # transform to PIL if we need to
        pil_img = sample
        if needs_transform:
            pil_img = transform(sample)

        target_boxes = rescale_bboxes(target_boxes, pil_img.size, device)

        # keep only predictions over confidence thresh
        pred_logits = torch.unsqueeze(pred_logits, 0)
        probas = pred_logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > confidence_thresh

        # convert boxes from [0; 1] to image scales
        pred_boxes = torch.unsqueeze(pred_boxes, 0)
        bboxes_scaled = rescale_bboxes(pred_boxes[0, keep], pil_img.size, device)
        prob, boxes = probas[keep], bboxes_scaled

        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()

        # predicted boxes 
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        if 'image_id' in target:
            plt.savefig(os.path.join(output_dir, str(target['image_id'].item())))
        else:
            plt.savefig(os.path.join(output_dir, str(i)))

        # ground truth boxes 
        for cl, (xmin, ymin, xmax, ymax) in zip(target['labels'], target_boxes.tolist()):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color="r", linewidth=3))
            text = f'{CLASSES[cl]}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        if 'image_id' in target:
            plt.savefig(os.path.join(output_dir, str(target['image_id'].item())))
        else:
            plt.savefig(os.path.join(output_dir, str(i)))
    plt.close('all')