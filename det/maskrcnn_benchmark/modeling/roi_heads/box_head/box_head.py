# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
import torchvision.transforms as T

import os
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import torch.nn.functional as F

CLASSES = ["N/A", "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"]

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        # self.aug_feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.source_counts = {}
        self.target_counts = {}
        
        self.iq_cls_score = 0.05
        self.num_classes = 9
        self.con_queue_dir = "/opt/MIC/det/work_dirs/roi_feats/mic_rcnn"
        self.hq_pro_counts_thr = 8       
        self.hq_gt_aug_cfg=dict(
                     trans_range=[0.3, 0.5],
                     trans_num=2,
                     rescale_range=[0.97, 1.03],
                     rescale_num=2)
        self.lq_score = 0.25 
        self.hq_score = 0.0
        
        self.cfg = cfg
        self.con_sampler_cfg = dict(
            num=128,
            pos_fraction=[0.5, 0.25, 0.125])
        self.iq_loss_weights=[0.5, 0.1, 0.05]
        self.contrast_loss_weights = 0.25
        self.temperature = 0.65
        self.num_gpus = 1
        self.num_con_queue = 256
        self.con_sample_num = 128
        
        enc_input_dim = 256
        enc_output_dim = 512
        proj_output_dim = 128
        
        self.comp_convs = self._add_comp_convs(256, 7, None, act_cfg=None)
        self.fc_enc = self._init_fc_enc(enc_input_dim, enc_output_dim)
        self.fc_proj = nn.Linear(enc_output_dim, proj_output_dim)
        self.relu = nn.ReLU(inplace=False)
        self.aug_feature_extractor = make_roi_box_feature_extractor(cfg)        
        # self.post_processor_train = make_roi_box_post_processor(cfg, max_prop=1)
        
        self.feat_dist_pos = {}
        self.feat_dist_neg = {}
        self.gt_counts = {}
        
        
    def _add_comp_convs(self, in_channels, roi_feat_size, norm_cfg, act_cfg):
        comp_convs = nn.ModuleList()
        for i in range(roi_feat_size//2):
            comp_convs.append(nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    bias=False
                ), 
                nn.GroupNorm(32, 256)
                )                
            )
        return comp_convs

    def _init_fc_enc(self, enc_input_dim, enc_output_dim):
        fc_enc = nn.ModuleList()
        fc_enc.append(nn.Linear(enc_input_dim, enc_output_dim))
        fc_enc.append(nn.Linear(enc_output_dim, enc_output_dim))
        return fc_enc

    def save_features(self, features, logits, image, target=False):
        cats = []
        for feat, log in zip(features, logits):
            label = torch.argmax(torch.softmax(log, dim=0))
            if label in cats:
                continue 
            else:
                if target:
                    t = "target_"
                    counts = self.target_counts
                else:
                    t = "source_"
                    counts = self.source_counts
                
                cats.append(label)
                if int(label.cpu().numpy()) not in counts:
                    # print("label not in dict " + str(label))
                    counts[int(label.cpu().numpy())] = 0
                counts[int(label.cpu().numpy())] += 1
                count = counts[int(label.cpu().numpy())]

                # make image output path 
                od = "./exps/c2fc_papersmodel_plot_features/" + t + str(CLASSES[label]) + "_" + str(count)

                # if path is dir, and make dir if not 
                if not os.path.isdir(od):
                    os.makedirs(od)

                transform = T.ToPILImage()
                pil_img = transform(image)
                plt.imshow(pil_img)
                plt.savefig(od + "/img.png")

                print("saving feature maps for class: " + str(CLASSES[label]) + " count: " + str(count))
                for i in range(5):
                    # print("post pool feature map " + str(i) + ": " + str(feat[i].shape))
                    imgplot = plt.imshow(feat[i].cpu().detach().numpy())
                    plt.title("post_pool_featuremap_count_" + str(i), fontsize=30)
                    plt.savefig(od + "/post_pool_featuremap_count_" + str(i), bbox_inches='tight')

    def save_feat_source_target(self, source_features, source_logits, source_class, target_features, target_logits, target_class):
        for s_feat, s_log in zip(source_features, source_logits):
            s_label = torch.argmax(torch.softmax(s_log, dim=0))
            s_label = int(s_label.cpu().numpy())

            if s_label == source_class:
                for t_feat, t_log in zip(target_features, target_logits):
                    t_label = torch.argmax(torch.softmax(t_log, dim=0))
                    t_label = int(t_label.cpu().numpy())

                    if t_label == target_class:
                        # make image output path 
                        od = "./exps/c2fc_papersmodel_plot_report_features/" + str(CLASSES[source_class]) + "_" + str(CLASSES[target_class])

                        # if path is dir, and make dir if not 
                        if not os.path.isdir(od):
                            os.makedirs(od)

                        square = 5
                        ix = 1
                        for i in range(square):
                            # specify subplot and turn of axis
                            ax = plt.subplot(square, square, ix)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # plot filter channel in grayscale
                            plt.imshow(s_feat[i].cpu().detach().numpy())
                            ix += 1

                        for i in range(square):
                            # specify subplot and turn of axis
                            ax = plt.subplot(square, square, ix)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # plot filter channel in grayscale
                            plt.imshow(t_feat[i].cpu().detach().numpy())
                            ix += 1

                        # save figure 
                        # plt.title("Features for Source Class " + str(CLASSES[source_class]) + " and Target Class " + str(CLASSES[target_class]), fontsize=30)
                        plt.savefig(od + "/post_pool_featuremap_" + str(CLASSES[source_class]) + "_" + str(CLASSES[target_class]), bbox_inches='tight')
                        print("SUCCESSFULLY SAVED FIGURE for Source Class " + str(CLASSES[source_class]) + " and Target Class " + str(CLASSES[target_class]))
                        break
                    
    def get_gt_quality(self, iq_scores, aug_num_per_hq_gt, gt_labels, cur_sample_num):
        """ low-quality:  0;
            mid_qulity:   1;
            high-quality: 2;
        """
        with torch.no_grad():
            iq_signs = torch.zeros_like(iq_scores)  # low-quality
            iq_signs[iq_scores >= self.lq_score] = 1  # mid-quality
            iq_signs[iq_scores >= self.hq_score] = 2  # high-quality
            pos_fraction = self.con_sampler_cfg['pos_fraction']
            ex_pos_nums = gt_labels.new_ones(iq_scores.size(0))
            for val in range(2):
                ex_pos_nums[iq_signs == val] = int(cur_sample_num * pos_fraction[val])
            ex_pos_nums[iq_signs == 2] = aug_num_per_hq_gt
        return iq_signs, ex_pos_nums
                    
    def aug_hq_gt_bboxes(self, hq_gt_bboxes, img_w):
        with torch.no_grad():
            hq_gt_bboxes = hq_gt_bboxes.view(-1, 4)
            num_gts = hq_gt_bboxes.size(0)
            trans_range, rescale_range = \
                self.hq_gt_aug_cfg['trans_range'], self.hq_gt_aug_cfg['rescale_range']
            trans_num, rescale_num = \
                self.hq_gt_aug_cfg['trans_num'], self.hq_gt_aug_cfg['rescale_num']
            trans_ratios = torch.linspace(
                trans_range[0], trans_range[1], trans_num).view(-1).cuda()
            rescale_ratios = torch.linspace(
                rescale_range[0], rescale_range[1], rescale_num).view(-1).cuda()

            gt_bboxes = hq_gt_bboxes.unsqueeze(1)
            # gt box translation
            trans_candi = gt_bboxes.repeat(1, 4 * trans_num, 1)  # (num_gts, 4*trans_num, 4)
            w = hq_gt_bboxes[:, 3] - hq_gt_bboxes[:, 1]
            h = hq_gt_bboxes[:, 2] - hq_gt_bboxes[:, 0]
            wh = torch.cat([w.view(-1, 1), h.view(-1, 1)], dim=1).unsqueeze(1)  # (num_gts, 1, 2)
            inter_mat = torch.cat(
                [torch.eye(2), torch.eye(2) * (-1)], dim=0).cuda()  # (4, 2)
            wh_mat = wh * inter_mat  # (num_gts, 4, 2)
            scaled_wh = torch.cat(  # (num_gts, 4*trans_num, 2)
                [r * wh_mat for r in trans_ratios], dim=1)
            trans_wh = scaled_wh.repeat(1, 1, 2)  # (num_gts, 4*trans_num, 4)
            trans_gt_bboxes = trans_candi + trans_wh  # (num_gts, 4*trans_num, 4)
            trans_gt_bboxes = torch.clamp(trans_gt_bboxes, 0, img_w)

            # gt box rescale
            rescaled_gt_bboxes = self.rescale_gt_bboxes(
                hq_gt_bboxes, rescale_ratios)  # (num_gts, rescale_num, 4)
            rescaled_gt_bboxes = torch.clamp(rescaled_gt_bboxes, 0, img_w)
            aug_gt_bboxes = []
            for i in range(num_gts):
                aug_gt_bboxes.append(
                    torch.cat([trans_gt_bboxes[i], rescaled_gt_bboxes[i]],
                              dim=0))
            aug_gt_bboxes = torch.cat(aug_gt_bboxes, dim=0)  # (num_gts, 4*trans_num+rescale_num, 4)
            aug_num_per_hq_gt = 4 * trans_num + rescale_num
        return aug_gt_bboxes, aug_num_per_hq_gt


    def rescale_gt_bboxes(self, gt_bboxes, scale_factors):
        cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) * 0.5
        cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) * 0.5
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        rescaled_gt_bboxes = []
        for scale_factor in scale_factors:
            new_w = w * scale_factor
            new_h = h * scale_factor
            x1 = cx - new_w * 0.5
            x2 = cx + new_w * 0.5
            y1 = cy - new_h * 0.5
            y2 = cy + new_h * 0.5
            rescaled_gt_bboxes.append(
                torch.stack((x1, y1, x2, y2), dim=-1))
        rescaled_gt_bboxes = torch.cat(
            rescaled_gt_bboxes, dim=0).view(gt_bboxes.size(0), -1, 4)
        return rescaled_gt_bboxes
                    
    def load_hq_roi_feats(self, roi_feats, gt_labels, cat_ids):
        device_id = str(gt_labels.device.index)  # current GPU id
        # print("load hq feats cat ids: " + str(cat_ids))
        
        with torch.no_grad():
            hq_feats, hq_labels, q_weights = [], [], []
            for cat_id in range(self.num_classes):
                # print("cat id " + str(cat_id) + " in cat ids: " + str(cat_id not in cat_ids))
                if cat_id not in cat_ids:
                    continue

                cur_cat_feat_pth = os.path.join(
                    self.con_queue_dir, device_id, str(cat_id) + '.pt')
                
                print("======================Attemping to load hq ROI feat from: " + str(cur_cat_feat_pth) + "======================")
                # print(os.path.exists(cur_cat_feat_pth))
                if os.path.exists(cur_cat_feat_pth):
                    print("======================Successfully loaded hq ROI feat from: " + str(cur_cat_feat_pth) + "======================")
                    cur_cat_feat = torch.load(cur_cat_feat_pth)
                else:
                    cur_cat_feat = roi_feats.new_empty(0)
                    
                print("curr cat feat: " + str(cur_cat_feat.shape))
                    
                cur_cat_roi_feats = cur_cat_feat.to(roi_feats.device).view(-1, 256, 7, 7)
                cur_hq_labels = cat_id * gt_labels.new_ones(
                    cur_cat_roi_feats.size(0)).to(gt_labels.device)
                
                hq_feats.append(cur_cat_roi_feats)
                hq_labels.append(cur_hq_labels)

                cur_gt_weight_save_pth = os.path.join(self.con_queue_dir, device_id, str(cat_id) + '_weight.pt')
                if os.path.exists(cur_gt_weight_save_pth):
                    cur_weights = torch.load(cur_gt_weight_save_pth)

                q_weights.append(cur_weights)
                
            print("load hq roi feats hq feats: " + str(len(hq_feats)))
            print("hq labels: " + str(hq_labels))
                
            hq_feats = torch.as_tensor(
                torch.cat(hq_feats, dim=0),
                dtype=roi_feats.dtype).view(-1, 256, 7, 7)
            hq_labels = torch.as_tensor(
                torch.cat(hq_labels, dim=-1), dtype=gt_labels.dtype)
            q_weights = torch.as_tensor(
                torch.cat(q_weights, dim=-1))
                        
        return hq_feats, hq_labels, q_weights
    
    def update_iq_score_info(self, cat_id, cur_gt_roi_feat, cur_Q_weight):
        cur_gt_roi_feat = cur_gt_roi_feat.view(-1, 256, 7, 7)
        # update the iq_score queue and corresponding dict info
        device_dir = str(cur_gt_roi_feat.device.index)
        cur_gt_save_pth = os.path.join(
            self.con_queue_dir, device_dir, str(cat_id.item()) + '.pt')
        
        if os.path.exists(cur_gt_save_pth):
            cur_pt = torch.load(cur_gt_save_pth).view(-1, 256, 7, 7)
            os.remove(cur_gt_save_pth)
            cur_gt_roi_feat = torch.cat(
                [cur_pt.to(cur_gt_roi_feat.device), cur_gt_roi_feat], dim=0)
            
        cur_gt_roi_feat = cur_gt_roi_feat.view(-1, 256, 7, 7)
        dup_len = cur_gt_roi_feat.size(0) > int(self.num_con_queue // self.num_gpus)
        
        # if cat_id.item() == 3:
        #     with open('queue_test.txt', 'a') as file:
        #         file.write("dup len: " + str(dup_len) + "\n")
        #         file.write("cur gt roi feat: " + str(cur_gt_roi_feat.shape) + "\n")
        #         for i in range(cur_gt_roi_feat.size(0)):
        #             file.write(str(i) + ": " + str(cur_gt_roi_feat[i][0][0][0]) + "\n")
        
        if dup_len > 0:
            cur_gt_roi_feat = cur_gt_roi_feat[dup_len:, ...]

        # if cat_id.item() == 3:
        #     with open('queue_test.txt', 'a') as file:
        #         file.write("cur gt roi feat after dup len cutoff: " + str(cur_gt_roi_feat.shape) + "\n")
        #         file.write("cur gt roi feat: " + str(cur_gt_roi_feat.shape) + "\n")
        #         for i in range(cur_gt_roi_feat.size(0)):
        #             file.write(str(i) + ": " + str(cur_gt_roi_feat[i][0][0][0]) + "\n")
        #         file.write("saving ROI feat to: " + str(cur_gt_save_pth) + "================================================================\n")
        
        torch.save(cur_gt_roi_feat, cur_gt_save_pth, _use_new_zipfile_serialization=False)
        
        # for Q weight saving
        cur_gt_weight_save_pth = os.path.join(
            self.con_queue_dir, device_dir, str(cat_id.item()) + '_weight.pt')
        
        if os.path.exists(cur_gt_weight_save_pth):
            cur_weights = torch.load(cur_gt_weight_save_pth)
            os.remove(cur_gt_weight_save_pth)
            cur_Q_weight = torch.cat(
                [cur_weights.to(cur_gt_roi_feat.device), cur_Q_weight], dim=0)
            
        # if cat_id.item() == 3:
        #     with open('queue_test.txt', 'a') as file:
        #         file.write("cur q weight: " + str(cur_Q_weight) + "\n")
        #         for i in range(cur_Q_weight.size(0)):
        #             file.write(str(i) + ": " + str(cur_Q_weight[i]) + "\n")
            
        if dup_len > 0:
            cur_Q_weight = cur_Q_weight[dup_len:, ...]

        # if cat_id.item() == 3:
        #     with open('queue_test.txt', 'a') as file:
        #         file.write("q weights after dup len cutoff: " + str(cur_Q_weight.shape) + "\n")
        #         file.write("q weights: " + str(cur_Q_weight.shape) + "\n")
        #         for i in range(cur_Q_weight.size(0)):
        #             file.write(str(i) + ": " + str(cur_Q_weight[i]) + "\n")
        #         file.write("saving q weights to: " + str(cur_gt_weight_save_pth) + "\n")

        torch.save(cur_Q_weight, cur_gt_weight_save_pth, _use_new_zipfile_serialization=False)
    

    def ins_quality_assess(self, pos_proposals, class_logits, filtered_inds, target, eps=1e-6):
            """ Compute the quality of instances in a single image
                The quality of an instance is defined:
                    iq = 1 / N * (IoU * Score)_i (i: {1, 2, ..., N})
            """
            print("======================IN INS QUALITY==========================")
            gt_2_roi = []
            for i in range(1):
                # print("----------------------------")
                with torch.no_grad():
                    num_gts = len(target)
                    
                    # print("num gts: " + str(num_gts))                    
                    # print(filtered_inds)
                    # pos_proposals = proposals[i][filtered_inds]
                    # print(pos_proposals)
                    
                    pos_gt_labels = pos_proposals.extra_fields['labels']
                    pos_logits = class_logits[filtered_inds]
                    
                    scores = torch.softmax(pos_logits.detach(), dim=1)
                    scores = torch.gather(scores, dim=1, index=pos_gt_labels.view(-1, 1)).view(-1)  # (num_pos, )
                            
                iq_candi_inds = scores >= self.iq_cls_score
                if torch.sum(iq_candi_inds) == 0:
                    return scores.new_zeros(num_gts), scores.new_zeros(num_gts), []
                else:
                    pos_proposals = pos_proposals[iq_candi_inds]                    
                    scores = scores[iq_candi_inds]
                    
                    # CFINet paper: eq 3 
                    ins_q = []
                    ins_counts = []
                    for j in range(num_gts):
                        # print("for gt: " + str(j))
                        
                        with torch.no_grad():
                            ind_set = []
                            for k in range(len(pos_proposals)):
                                # print(torch.equal(target.bbox[j], pos_proposals.bbox[k]))
                                if torch.equal(target.bbox[j], pos_proposals.extra_fields['unweighted_targets'][k]):
                                    ind_set.append(k)                                    
                            
                            iou = boxlist_iou(target[[j]], pos_proposals[ind_set])
                            ins_q.append(torch.div(torch.sum(iou * scores[ind_set]), len(ind_set)))
                            ins_counts.append(len(ind_set))
                        
                        if len(ind_set) > 0:
                            best_ind = ind_set[torch.argmax(iou * scores[ind_set])]
                            gt_2_roi.append(pos_proposals[[best_ind]])
                            # print("ind set for target " + str(j) + ": " + str(ind_set))
                            # print("iou: " + str(iou) + " argmax iou: " + str(torch.argmax(iou)))
                            # print("scores: " + str(scores[ind_set]))
                        else:
                            gt_2_roi.append(-1)
                        
                        # print(ins_q) 
                        # print("num_pos: " + str(num_pos))
                        # print("pos proposals: " + str(pos_proposals.extra_fields))
                        # print(target.bbox)

            return torch.tensor(ins_q), torch.tensor(ins_counts), gt_2_roi

    def forward(self, features, proposals, targets=None, use_pseudo_labeling_weight='none', images=None, with_da_on=True):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        print("======================INPUT TO ROI BOX HEAD FORWARD==========================")

        # if self.training and with_da_on:
        if self.training:
            num_gts = [len(t) for t in targets]
            print("num gts: " + str(num_gts))
            
            # for target in targets:
            #     print("number of targets: " + str(len(target)))
            #     print(target.bbox.shape)
            #     print(target.extra_fields['labels'].shape)

            print("proposal len: " + str(len(proposals)) + " targets len: " + str(len(targets)))

            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals, filtered_pos_inds_imgs, filtered_neg_inds_imgs = self.loss_evaluator.subsample(proposals, targets, num_gts)
                
                for fp in filtered_pos_inds_imgs:
                    print(len(fp))
                    
                # if len(filtered_pos_inds_imgs) == 2:
                #     filtered_pos_inds_imgs = filtered_pos_inds_imgs[0]
                    
                # if len(filtered_neg_inds_imgs) == 2:
                #     filtered_neg_inds_imgs = filtered_neg_inds_imgs[0]
                    
                # if isinstance(filtered_pos_inds_imgs[0], list):
                #     filtered_pos_inds_imgs = filtered_pos_inds_imgs[0]
                    
                # if isinstance(filtered_neg_inds_imgs[0], list):
                #     filtered_neg_inds_imgs = filtered_neg_inds_imgs[0]

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        
        for f in features:
            print("features input to box head: " + str(f.shape))
            
        # for p in proposals:
        #     print(p.bbox)
        #     print(p.extra_fields)
        
        x, pre_fc_feats = self.feature_extractor(features, proposals)

        print("pre fc feats: " + str(pre_fc_feats.shape))
        
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if with_da_on:
            n = len(proposals) // 2
        else:
            n = len(proposals)
                
        # filter source features only 
        con_loss = 0
        if self.training:
            prev = 0
            for i in range(n):
                cat_ids = []
                for label in targets[i].extra_fields['labels']:
                    if label not in cat_ids:
                        cat_ids.append(label)
                    
                num_proposals = len(proposals[i])
                source_pre_fc_feats = pre_fc_feats[prev: prev + num_proposals]
                class_logits_source = class_logits[prev: prev + num_proposals]

                print("prev: " + str(prev) + " prev + num proposals: " + str(prev + num_proposals))
                prev = prev + num_proposals

                cur_sample_num = min(len(filtered_neg_inds_imgs[i]), self.con_sample_num)
                print("cur sample num: " + str(cur_sample_num) + " con sample num: " + str(self.con_sample_num) + " neg inds imgs: " + str(len(filtered_neg_inds_imgs[i])) + " pos inds imgs: " + str(len(filtered_pos_inds_imgs[i])))

                print(len(filtered_neg_inds_imgs[i]) == 0)
                if len(filtered_neg_inds_imgs[i]) == 0:
                    print("should be continuing!")
                    continue
                
                print(len(filtered_pos_inds_imgs[i]) == 0)
                if len(filtered_pos_inds_imgs[i]) == 0:
                    print("should be continuing!")
                    continue

                # 1) convolve features extracted from proposals
                print("source roi feats: " + str(source_pre_fc_feats.shape) + " proposals: " + str(proposals[i]))
                print(source_pre_fc_feats.requires_grad)

                if with_da_on == False:
                    comp_feats = source_pre_fc_feats.clone()
                    for conv in self.comp_convs:
                        comp_feats = conv(comp_feats)
                    proposals[i].add_field("comp_feats", comp_feats)

                    print("comp feats req grad: " + str(comp_feats.requires_grad))
                    print("convolved features extracted from roi proposals: " + str(comp_feats.shape))
                
                # associate assorted features with incoming proposals    
                proposals[i].add_field("roi_feats", source_pre_fc_feats)
                
                # print("comp feats req grad after adding to proposal: " + str(proposals[0].extra_fields["comp_feats"].requires_grad))
                con_losses = class_logits_source.new_zeros(1)

                gt_bboxes = proposals[i].extra_fields['unweighted_targets']
                gt_labels = targets[i].extra_fields['labels']
                proposal_labels = proposals[i].extra_fields['labels']

                print("gt labels: " + str(gt_labels))
                # print("proposal labels: " + str(len(proposal_labels)))

                pos_proposals = proposals[i][filtered_pos_inds_imgs[i]]
                pos_gt_labels = pos_proposals.extra_fields['labels']
                pos_gt_bboxes = pos_proposals.extra_fields['unweighted_targets']
                
                # print("filtered pos image inds: " + str(filtered_pos_inds_imgs))
                # print("num pos proposals: " + str(len(pos_proposals)))
                
                iq_scores, pro_counts, gt_2_roi = self.ins_quality_assess(pos_proposals, class_logits_source, filtered_pos_inds_imgs[i], targets[i])
                
                # print("comp feats req grad gt 2 roi: " + str(gt_2_roi[0].extra_fields["comp_feats"].requires_grad))
                print("iq scores: " + str(iq_scores))
                print("pro counts: " + str(pro_counts))
                
                # hq_inds = torch.nonzero((iq_scores >= self.hq_score) & \
                #                             (pro_counts >= self.hq_pro_counts_thr),
                #                             as_tuple=False).view(-1) # (N, )
                hq_inds = torch.nonzero(pro_counts >= self.hq_pro_counts_thr, as_tuple=False).view(-1) # (N, )
                
                print("gt indicies that have the proposals necessary for a high IQ score: " + str(hq_inds))
                print("hq thresh: " + str(self.hq_score) + " temp: " + str(self.temperature) + " cl: " + str(self.contrast_loss_weights))

                if with_da_on == False:
                    hq_feats, hq_labels, q_weights = self.load_hq_roi_feats(source_pre_fc_feats, gt_labels, cat_ids)
                    
                    print("hq roi feats loaded from database: " + str(hq_feats.shape))
                    print("hq labels: " + str(hq_labels.shape))
                    print("q weights loaded from database: " + str(q_weights.shape))
                    
                    with torch.no_grad():
                        for conv in self.comp_convs:
                            hq_feats = conv(hq_feats)  # [num_proposals, 256, 1, 1]
                            
                    print("hq feats after convolving: " + str(hq_feats.shape))
                    
                    con_roi_feats = torch.cat([comp_feats, hq_feats], dim=0)  # [num_proposals + num_hq, 256, 1, 1]
                    
                    print("con roi feats now has convolved roi feats from proposals, hq database: " + str(con_roi_feats.shape))
                    
                    if len(hq_inds) == 0:    # no high-quality gt in current image
                        aug_gt_ind = -1 * torch.ones(con_roi_feats.size(0))
                        aug_num_per_hq_gt = 0
                        aug_hq_gt_bboxes = gt_bboxes.new_empty(0)
                        aug_gt_labels = gt_labels.new_empty(0)
                        aug_q_scores = gt_labels.new_empty(0)
                    else:                
                        hq_gt_bboxes = pos_gt_bboxes[hq_inds]
                        img_size = targets[0].size  # use img_w only since img_w == img_h
                        aug_hq_gt_bboxes, aug_num_per_hq_gt = \
                            self.aug_hq_gt_bboxes(hq_gt_bboxes, img_size[0])
                        
                        # TODO: visualize aug_hq_gt_bboxes
                            
                        # aug_hq_gt_bboxes = BoxList(aug_hq_gt_bboxes, img_size)
                        # print("aug hq gt bboxes: " + str(aug_hq_gt_bboxes))
                        
                        aug_hq_gt_roi_feats = self.aug_feature_extractor(features, [BoxList(aug_hq_gt_bboxes, img_size)])[1]
                        
                        print("roi feats extracted from augmented gt bboxes corresponding to hq inds: " + str(aug_hq_gt_roi_feats.shape) + " " + str(aug_hq_gt_roi_feats.requires_grad))
                        with torch.no_grad():
                            for conv in self.comp_convs:
                                aug_hq_gt_roi_feats = conv(aug_hq_gt_roi_feats)
                        print("convolved aug hq gt roi feats: " + str(aug_hq_gt_roi_feats.shape))
                                
                        # defines which indicies of con_roi_feats correspond to features from augmented bboxes
                        aug_gt_ind = hq_inds.view(-1, 1).repeat(1, aug_num_per_hq_gt).view(1, -1).squeeze(0)
                        aug_gt_ind = torch.cat([-1 * aug_gt_ind.new_ones(con_roi_feats.size(0)), aug_gt_ind], dim=-1).to('cuda')
                        
                        # print("aug gt ind: " + str(aug_gt_ind))
                        
                        aug_gt_labels = pos_gt_labels[hq_inds].view(-1, 1).repeat(1, aug_num_per_hq_gt).view(1, -1).squeeze(0)
                        con_roi_feats = torch.cat([con_roi_feats, aug_hq_gt_roi_feats], dim=0)  # [num_proposals + num_hq + num_hq_aug, 256, 1, 1]

                        aug_q_scores = iq_scores[hq_inds].view(-1, 1).repeat(1, aug_num_per_hq_gt).to('cuda').squeeze(0)
                        aug_q_scores = aug_q_scores.reshape(-1, 1)
                        # print("aug q scores: " + str(aug_q_scores))

                    # format q scores so they can be masked 
                    q_weights = torch.cat([gt_labels.new_zeros(num_proposals), q_weights, aug_q_scores.view(-1)], dim=-1)

                    print("q weights: " + str(q_weights.shape))                        
                    print("con roi feats now has convolved roi feats from proposals, hq database, augmented gt bbox: " + str(con_roi_feats.shape))
                
                    iq_signs, ex_pos_nums = self.get_gt_quality(iq_scores, aug_num_per_hq_gt, gt_labels, cur_sample_num)
                    
                    print("iq signs: " + str(iq_signs))
                    print("ex pos nums: " + str(ex_pos_nums))
                    print("iq scores: " + str(iq_scores))
                    
                    is_hq = torch.cat(
                        [gt_labels.new_zeros(num_proposals),
                        torch.ones_like(hq_labels),
                        -gt_labels.new_ones(aug_hq_gt_bboxes.size(0))], dim=-1)
                    
                    # defines what the labels of the roi features are generally
                    roi_labels = torch.cat(
                        [proposal_labels, hq_labels, aug_gt_labels], dim=-1)
                                
                    assert roi_labels.size(0) == con_roi_feats.size(0)
                    
                    anchor_feature = []
                    gt_labels_w_rois = []
                    iq_signs_w_roi = []
                    i = 0
                    for gt in gt_2_roi:
                        if gt != -1:
                            # print("comp feats req grad: " + str(gt.extra_fields['comp_feats'].requires_grad))
                            anchor_feature.append(gt.extra_fields['comp_feats'])
                            gt_labels_w_rois.append(gt.extra_fields['labels'])
                            iq_signs_w_roi.append(iq_signs[i].item())
                            # print(anchor_feature[-1].requires_grad)
                        else:
                            gt_labels_w_rois.append(-1)
                        i += 1
                    iq_signs_w_roi = torch.tensor(iq_signs_w_roi)
                    
                    print("anchor feat len: " + str(len(anchor_feature)))
                    if len(anchor_feature) > 0:
                        anchor_feature = torch.cat(anchor_feature, dim=0)       
                        
                        sample_inds, pos_signs = self.sample(
                            iq_signs, ex_pos_nums, gt_labels_w_rois, roi_labels, is_hq, aug_gt_ind, cur_sample_num, q_weights)
                            
                        # print("pos inds to obtain anchor feature: " + str(filtered_pos_inds_imgs[:num_gts]) + " num gts: " + str(num_gts))
                        print("sample inds to obtain contrast feature: " + str(sample_inds.shape))
                        # print(pos_signs)
                            
                        contrast_feature = con_roi_feats[sample_inds]
                        
                        print(str(anchor_feature.shape) + " " + str(anchor_feature.requires_grad))
                        print(str(contrast_feature.shape) + " " + str(contrast_feature.requires_grad))

                        iq_loss_weights = torch.ones(len(anchor_feature)).to(device=anchor_feature.device)
                        for j, weight in enumerate(self.iq_loss_weights):
                            iq_loss_weights[iq_signs_w_roi == j] *= weight
                        
                        loss = self.contrast_forward(anchor_feature, contrast_feature,
                                                    pos_signs, iq_loss_weights, gt_labels_w_rois)
                        contrast_loss = self.contrast_loss_weights * loss
                        con_losses = con_losses + contrast_loss
                    
                        print("con losses: " + str(con_losses))
                
                # save high-quality features at last
                # high-quality proposals: high instance quality scores and
                # sufficient numbers of proposals
                if len(hq_inds) > 0:
                    hq_scores, hq_pro_counts = \
                        iq_scores[hq_inds], pro_counts[hq_inds]
                    print("hq scores: " + str(hq_scores))
                    for hq_score, hq_pro_count, hq_gt_ind in \
                            zip(hq_scores, hq_pro_counts, hq_inds):
                        cur_gt_cat_id = gt_2_roi[hq_gt_ind].extra_fields['labels']
                        cur_gt_roi_feat = gt_2_roi[hq_gt_ind].extra_fields['roi_feats'].clone()
                        cur_gt_roi_feat = torch.squeeze(cur_gt_roi_feat, 0)
                        self.update_iq_score_info(cur_gt_cat_id, cur_gt_roi_feat, torch.unsqueeze(hq_score, dim=0).to('cuda'))
                        
                if len(con_losses) > 0:
                    con_loss += con_losses 
        
        print("BOX REGRESS PREDICTOR OUTPUT INSIDE ROI BOX HEAD: " + str(box_regression.shape))

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)

            # print("OUTPUT OF ROI BOX HEAD: " + str(result))

            return x, result, {}, x, None, result

        loss_classifier, loss_box_reg, _ = self.loss_evaluator(
            [class_logits], [box_regression], use_pseudo_labeling_weight
        )

        if self.training:
            with torch.no_grad():
                da_proposals = self.loss_evaluator.subsample_for_da(proposals, targets)

        da_ins_feas, _ = self.feature_extractor(features, da_proposals)
        class_logits, box_regression = self.predictor(da_ins_feas)
        _, _, da_ins_labels = self.loss_evaluator(
            [class_logits], [box_regression]
        )

        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_contrast=con_loss[0] / n),
            da_ins_feas,
            da_ins_labels,
            da_proposals
        )
        
    def sample(self, iq_signs, ex_pos_nums, gt_labels, roi_labels,
                is_hq, aug_gt_ind, cur_sample_num, q_weights):
        """
        Returns:
            sample_inds : indices of pos and neg samples (num_gts, self.con_sample_num)
            pos_signs   : whether the sample of current index is positive
        """
        # print("aug gt ind: " + str(aug_gt_ind))
        # print("gt labels: " + str(gt_labels))
        # print("roi labels: " + str(roi_labels))
        # print(ex_pos_nums)
        
        sample_inds, pos_signs = [], []
        for gt_ind in range(len(gt_labels)):
            print("gt ind: " + str(gt_ind))
            if gt_labels[gt_ind] != -1:
                # ex_pos_num = ex_pos_nums[gt_ind]
                ex_pos_num = 10

                iq_sign = iq_signs[gt_ind]

                # sample positives first
                # if iq_sign == 2:
                #     pos_inds = torch.nonzero(aug_gt_ind == gt_ind, as_tuple=False).view(-1)
                #     print("pos inds from aug gt ind: " + str(pos_inds))
                # else:
                #     can_pos_inds = torch.nonzero(
                #         (is_hq == 1) & (roi_labels == gt_labels[gt_ind]),
                #         as_tuple=False).view(-1)
                #     if len(can_pos_inds) <= ex_pos_num:
                #         pos_inds = can_pos_inds
                #     else:
                #         pos_inds = self._random_choice(can_pos_inds, ex_pos_num)

                # print(is_hq)
                # print(aug_gt_ind)
                # print(gt_ind)
                # print(roi_labels)
                # print(gt_labels[gt_ind])
                        
                can_pos_inds = torch.nonzero(((is_hq == 1) | (aug_gt_ind.to('cuda') == gt_ind)) & (roi_labels == gt_labels[gt_ind]))
                # print("aug inds + hq inds w right label: " + str(can_pos_inds))
                # assert can_pos_inds[0, 0].item() > 255

                if len(can_pos_inds) <= ex_pos_num:
                    pos_inds = can_pos_inds
                else:
                    q_weights_sample = q_weights[can_pos_inds]
                    q_probs = F.softmax(q_weights_sample, dim=0)

                    # print("q weights sample: " + str(q_weights_sample))
                    # print("q weights sample softmax: " + str(q_probs))

                    sample_indices = torch.multinomial(q_probs.view(-1), num_samples=ex_pos_num, replacement=False)

                    # print("sample indicies: " + str(sample_indices))

                    pos_inds = can_pos_inds[sample_indices]

                    # print("pos inds: " + str(pos_inds))
                
                # sample negatives
                can_neg_inds = torch.nonzero(
                    (roi_labels != gt_labels[gt_ind]) & (is_hq == 0),
                    as_tuple=False).view(-1)
                
                # print("pos labels: " + str(roi_labels[pos_inds]))
                # print("gt ind: " + str(gt_ind) + " pos inds: " + str(pos_inds))
                # print("can neg inds: " + str(can_neg_inds.shape))
                # print("diff: " + str(cur_sample_num - len(pos_inds)) + " pos inds len: " + str(len(pos_inds)))
                
                neg_inds = self._random_choice(
                    can_neg_inds, cur_sample_num - len(pos_inds))
                sample_inds.append(
                    torch.cat([pos_inds.view(-1).cuda(), neg_inds.cuda()], dim=-1).view(1, -1))
                pos_signs.append(
                    torch.cat([torch.ones_like(pos_inds.view(-1).cuda()),
                            torch.zeros_like(neg_inds.cuda())], dim=-1).view(1, -1))
                
        sample_inds = torch.cat(sample_inds, dim=0)
        pos_signs = torch.cat(pos_signs, dim=0)
        return sample_inds, pos_signs
        
    def contrast_forward(self, anchor_feature, contrast_feature,
                         pos_signs, loss_weights, gt_labels, eps=1e-6):
        """
        Args:
            anchor_feature: ground-truth roi features in a single image
                (num_gts, 256, 1, 1)
            contrast_feature: pos/neg rois features fro training
                (num_gts, self.con_sample_num, 256, 1, 1)
            pos_signs: indicate whether the sample pos/neg (1/0)
                (num_gts, self.con_sample_num)
            loss_weights: loss weights of each gt (num_gts, )
        """
        anchor_feature = anchor_feature.view(anchor_feature.size()[:-2])  # [num_gts, 256]
        contrast_feature = contrast_feature.view(contrast_feature.size()[:-2])  # [num_gts, self.con_sample_num, 256]
        
        for fc in self.fc_enc:
            anchor_feature = self.relu(fc(anchor_feature))
            contrast_feature = self.relu(fc(contrast_feature))
            
        anchor_feature = self.fc_proj(anchor_feature)
        contrast_feature = self.fc_proj(contrast_feature)
        
        anchor_feats = F.normalize(anchor_feature, dim=-1)  # (num_gts, 128)
        contrast_feats = F.normalize(contrast_feature, dim=-1)  # (num_gts, self.con_sample_num, 128)
        
        print("encoded and projected anchor features: " + str(anchor_feats.shape))
        print("encoded and projected contrast features: " + str(contrast_feats.shape))
        print("anchor feats: " + str(anchor_feats.unsqueeze(1).shape) + " contrast feats: " + str(contrast_feats.transpose(2, 1).contiguous().shape) + " temp: " + str(self.temperature))
        
        cos_dist = torch.matmul(anchor_feats.unsqueeze(1), contrast_feats.transpose(2, 1).contiguous())
        print("cos dist shape: " + str(cos_dist.shape))
        sim_logits = torch.div(cos_dist, self.temperature).squeeze(1)
        print("sim logits shape: " + str(sim_logits.shape))
        
        pos_num = pos_signs.sum(dim=1).cuda()
        pos_num = pos_num + eps * (pos_num == 0)  # avoid dividing by zero
        
        # with torch.no_grad():
        #     cos_pos = (cos_dist.squeeze(1) * pos_signs).sum(dim=1) / pos_num
        #     cos_neg = (cos_dist.squeeze(1) * torch.flip(pos_signs, dims=(1,))).sum(dim=1) / pos_num
            
        #     print("cos_pos avg: " + str(cos_pos))
        #     print("cos_neg avg: " + str(cos_neg))
            
        #     gt_labels = [gt for gt in gt_labels if gt != -1]
        #     for i in range(len(gt_labels)):
        #         gt = gt_labels[i].item()
        
        #         if gt not in self.gt_counts:
        #             self.gt_counts[gt] = 1
        #         else:
        #             self.gt_counts[gt] += 1
                    
        #         if gt not in self.feat_dist_pos:
        #             self.feat_dist_pos[gt] = cos_pos[i]
        #         else:
        #             self.feat_dist_pos[gt] += cos_pos[i]
                
        #         if gt not in self.feat_dist_neg:
        #             self.feat_dist_neg[gt] = cos_neg[i]
        #         else:
        #             self.feat_dist_neg[gt] += cos_neg[i]
                    
        #     for gt in self.feat_dist_pos:
        #         count = self.gt_counts[gt]
        #         print("gt: " + str(gt) + " avg roi feat dist to positive: " + str(self.feat_dist_pos[gt] / count) + " avg roi feat dist to negative: " + str(self.feat_dist_neg[gt] / count))
        
        # for numerical stability
        sim_logits_max, _ = torch.max(sim_logits, dim=1, keepdim=True)
        logits = sim_logits - sim_logits_max.detach()  # (num_gts, self.con_sample_num)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # print("pos num: " + str(pos_num))
        # print("log prob: " + str(-(pos_signs * log_prob).sum(dim=1)))
        
        mean_log_prob_pos = -(pos_signs * log_prob).sum(dim=1) / pos_num
        weighted_loss = loss_weights * mean_log_prob_pos
        
        print("weighted loss: " + str(weighted_loss))
        
        loss = weighted_loss.mean()
        return loss
        
    def _random_choice(self, gallery, num):
        # fork from RandomSampler
        assert len(gallery) >= num
        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
