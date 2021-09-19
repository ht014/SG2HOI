## SG2HOI
This repository is for our paper [Exploiting Scene Graphs for Human-Object Interaction Detection](https://arxiv.org/pdf/2108.08584) accepted by ICCV 2021.
![image]()
## Installation
#### Pytorch 1.7.1 
```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install tdqm sklearn panda Pillow
```
#### maskrcnn 
Check [INSTALL.md](https://github.com/facebookresearch/maskrcnn-benchmark/blob/main/INSTALL.md) to install maskrcnn. Then, adding the maskrcnn lib to your $PYTHONPATH, because our code uses the ROIAlign layer to extract the roi features.

#### Apex
If you want to use multiple gpus to train the model, you have to follow the [instructions](https://github.com/NVIDIA/apex) to install apex.

## Datasets
### HOI datasets
We use the off-the-shell object detection results of V-COCO and HICO from [VSGnet](https://github.com/ASMIftekhar/VSGNet), which can be downloaded from [here](https://drive.google.com/file/d/1XwLrv2_jEWvUBCAiANSMUyy_NLN2UeiW/view?usp=sharing).

### Scene graph datasets
The scene graph prediction results are generated by [TDE](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). Note that we use all the training and testing images of [Visual Genome](https://github.com/ranjaykrishna/visual_genome_python_driver) to train the SG model. Our pre-trained TDE model can be downloaded from [here](~).

## Training and testing
```
$ python main.py --gpu_id 0 --learning_rate 0.01 --batch_size 5 --num_epochs 50 
```
## Citations

If you find this project helps your research, please kindly consider citing our papers in your publications.

```
@InProceedings{he2021exploiting,
    author    = {He, Tao and Gao, Lianli and Song, Jingkuan and Li, Yuan-Fang},
    title     = {Exploiting Scene Graphs for Human-Object Interaction Detection},
    booktitle = {International Conference on Computer Vision(ICCV)},
    year      = {2021},
    url       = {https://arxiv.org/pdf/2108.08584}
}
```
## Acknowledgement

This repository is developed on top of the other two projects: TDE by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and VSGnet by [ASMIftekhar](https://github.com/ASMIftekhar/VSGNet). 
