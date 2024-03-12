# Domain Adaptive Object Detection via Self-Supervised Feature Learning

## DAScope

Domain adaptive object detectors aim to classify and locate objects in a scene when domain gaps exist between the training (source) and testing (target) environments. 
Existing deep learning models for domain adaptive object detection suffer from poor detection performance with small and far-away objects. 
To address this, we propose DAScope, a novel, small-object-aware, domain adaptive object detection pipeline. 
The key novelty in our proposed pipeline is a self-supervised feature imitation learning module. 
Within a mean-teacher framework, it extracts object-level feature representations, stores them in a feature bank, and probabilistically samples them to compute a contrastive loss. 
This contrastive loss improves object-level feature representations, which is especially effective for detecting small objects. 
DAScope achieves competitive performance with the state-of-the-art detectors on location and weather adaptation tasks. 
Furthermore, DAScope also achieves superior performance on small object detection tasks with +0.8~+8.2 mAP improvement over SOTA methods. 

### Installation

Please follow the instruction in [det/INSTALL.md](det/INSTALL.md) to install and use this repo.

For installation problems, please consult issues in [ maskrcnn-benchmark
](https://github.com/facebookresearch/maskrcnn-benchmark).

This code is tested under Debian 11 with Python 3.9 and PyTorch 1.12.0.

### Datasets

The datasets used in the repository can be downloaded from the following links:

* [Cityscapes and Foggy Cityscapes](https://www.cityscapes-dataset.com/)
* [BDD100K](https://bdd-data.berkeley.edu/)

The datasets should be organized in the following structure.
```
datasets/
├── cityscapes
│   ├── annotations
│   ├── gtFine
│   └── leftImg8bit
├── foggy_cityscapes
│   ├── annotations
│   ├── gtFine
│   └── leftImg8bit
└── bdd_daytime
    ├── annotations
    └── images
```

All three datasets should be processed with [anno_convert.py](anno_convert.py) to:
* Convert annotations into the coco format
* Filter both datasets to contain only small objects

Filtering all three datasets with [anno_convert.py](anno_convert.py) to contain only small objects entails creating a second annotation file for each within their respective annotation folders using [anno_convert.py](anno_convert.py). The images in each dataset are not changed. More details on how to use [anno_convert.py](anno_convert.py) are described within the file itself. 

## Checkpoints

The model weights used to produce the results in our experiements section can be found through this google drive [link](https://drive.google.com/drive/u/3/folders/190p6_ZunhC1oIf8zdQ2G1h1ROEUMaPbr).

## Training

For experiments in our paper, we use the following script to run Cityscapes to Foggy Cityscapes adaptation task:

```shell
python det/tools/train_net.py --config-file configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml
```

## Testing

The trained model could be evaluated with the following script:
```shell
python det/tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml" MODEL.WEIGHT <path_to_store_weight>/model_final.pth
```

## Where to find DAScope in the code?

The most relevant files for DAScope are:

* [det/configs/da_faster_rcnn/](det/configs/da_faster_rcnn/):
  Definition of the experiment configurations in our paper.
* [det/tools/train_net.py](det/tools/train_net.py):
  Training script for UDA with DAScope(MIC + sa-da-faster).
* [det/maskrcnn_benchmark/engine/trainer.py](det/maskrcnn_benchmark/engine/trainer.py):
  Training process for UDA with DAScope(MIC + sa-da-faster).
* [det/maskrcnn_benchmark/modeling/roi_heads/box_head/box_head.py](det/maskrcnn_benchmark/modeling/roi_heads/box_head/box_head.py):
  Implementation of the DAScope feature imitation learning pipeline.

## Acknowledgements

DAScope is based on the following open-source projects. 
We thank their authors for making the source code publicly available.

* [MIC](https://github.com/lhoyer/MIC)
* [CFINet](https://github.com/shaunyuan22/cfinet)
* [sa-da-faster](https://github.com/yuhuayc/sa-da-faster)
* [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
