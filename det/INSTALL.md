## Installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

conda create --name mic-sada -y
source /opt/conda/etc/profile.d/conda.sh
conda activate mic-sada

# this installs the right pip and dependencies for the fresh python
conda install python=3.9.7 ipython=8.1.1 pip=21.2.4 -y
apt-get install graphviz

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 11.3
conda install pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3 -c pytorch -y

# maskrcnn_benchmark and coco api dependencies
pip install piqa ninja==1.10.2.3 yacs==0.1.8 cython==0.29.28 matplotlib==3.5.1 tqdm==4.63.0 opencv-python==4.5.5.64 numpy==1.22.3 torchfile pycocotools
python tools/train_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml" MODEL.WEIGHT ./exps/c2fc_imitation_target_saveboth_lw_temp0.6_hq0.7_batch4_lr0.0035_nofix/model_0025000.pth
# MIC dependencies
pip install timm==0.6.11 kornia==0.5.8 einops==0.4.1 torchviz

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install sa-da-faster
cd $INSTALL_DIR
git clone https://github.com/lhoyer/MIC
cd MIC/det
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR
```

To run [convert_cityscapes_to_coco.py](tools/cityscapes/convert_cityscapes_to_coco.py) and [convert_foggy_cityscapes_to_coco.py](tools/cityscapes/convert_foggy_cityscapes_to_coco.py), you also need to run:

```bash
# copy required file to cityscapesScripts toolkit
cp tool/cityscapes/instances2dict_with_polygons.py $INSTALL_DIR/cityscapesScripts/evaluation

# recompile cityscapesScripts
cd cityscapesScripts/
python setup.py build_ext install

pip install h5py==3.6.0 scipy==1.8.0
```
