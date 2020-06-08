# Attentive WaveBlock (AWB)
The official implementation for the "Attentive WaveBlock: Complementarity-enhanced Mutual Networks for Unsupervised Domain Adaptation in Person Re-identification". Many thanks for the framework offered by [**MMT**](https://github.com/yxgeee/MMT).

![image](https://github.com/WangWenhao0716/Attentive-WaveBlock/blob/master/feature_map.png)

## Performance
![image](https://github.com/WangWenhao0716/Attentive-WaveBlock/blob/master/performance.png)

## Requirement
* Python 3.6
* Pytorch 1.1.0
* Numpy 1.18.1
* Torchvision 0.2.0

## Reproduction Environment
* Test our models: 1 GTX 1080Ti GPU.
* Train new models: 4 GTX 1080Ti GPUs.

## Installation
You can directly get the codes by:
```
  git clone https://github.com/WangWenhao0716/Attentive-WaveBlock.git
```

## Note
In the following codes and statements, for simplicity, Pre-A denotes Pre-A (CBAM) and Post-A denotes Post-A (Non-local). You can find the details in our paper. 

## Preparation
1. Dataset

We evaluate our algorithm on [**DukeMTMC-reID**](https://arxiv.org/abs/1609.01775), [**Market-1501**](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) and [**MSMT17**](https://arxiv.org/abs/1711.08565). For convenience, we have prepared [**data**](https://drive.google.com/open?id=1aT_SZkAOQf9VuycXiSCTzPDDH2BOuMMT) for you. You should download them and prepare the directory structure like this:

```
*DATA_PATH
      *data
         *dukemtmc
             *DukeMTMC-reID
                 *bounding_box_test
                 ...
          
         *market1501
             *Market-1501-v15.09.15
                 *bounding_box_test
                 ...
         *msmt17
             *MSMT17_V1
                 *test
                 *train
                 ...
```


2. Pretrained Models

We use Resnet-50 as backbone, and the pretrained models will be downloaded automatically.

The pretrained parameters for CBAM module in our Pre-A can be downloaded from [**CBAM**](https://www.dropbox.com/s/bt6zty02h9ibufi/RESNET50_CBAM_new_name_wrap.pth?dl=0). It should be saved in ```./mmt/models_pre_a_s1``` .

3. Our Trained Models

We provide our trained Pre-A and Post-A models on Duke-to-Market, Market-to-Duke, Duke-to-MSMT, and Market-to-MSMT domain adaptation tasks.

Duke-to-Market:  [**Pre-A(78.0% mAP)**](https://drive.google.com/open?id=1c9JvTO45ltNlSYHAC99vB4CMmYfqED8V)    [**Post-A(80.6% mAP)**](https://drive.google.com/open?id=1hzgXCNhNQdfFn-_CiEzEVik_X7_W_CVT)

Market-to-Duke:  [**Pre-A(69.1% mAP)**](https://drive.google.com/open?id=1-k9p5MJyL0ToSRownFrDifbXMPNM9aY7)    [**Post-A(71.0% mAP)**](https://drive.google.com/open?id=1MBlafM2nlguXlH3pBMHPuX6gOsMOS6Pz)

Duke-to-MSMT:    [**Pre-A(27.9% mAP)**](https://drive.google.com/open?id=10qtC_KFAVYdVaVpSyRoQ78DFno9FivXB)    [**Post-A(30.7% mAP)**](https://drive.google.com/open?id=1ZMA_2KFAGF7Y74u-bGgJMCSELveT5FzA)

Market-to-MSMT:  [**Pre-A(25.5% mAP)**](https://drive.google.com/open?id=1MEKjWdlewpI4PXkRiP5BIfPMD4U9NHJi)    [**Post-A(30.6% mAP)**](https://drive.google.com/open?id=1XsT7X2sTcY6gUFbeTbckiYGjRcDZm4Zh)


## Train

We use Duke-to-Market as an example, other UDA tasks will follow similar pipelines.

### Step 1: Pre-train models on source domain
#### Pre-A
First Network:

`CUDA_VISIBLE_DEVICES=0,1,2,3 python source_pretrain_pre_a.py -ds dukemtmc -dt market1501 -a resnet50 --seed 1 --margin 0.0 
    --num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 80 --eval-step 40 
    --logs-dir logs/dukemtmcTOmarket1501/resnet50-pretrain-1-pre-a`

Second Network:

`CUDA_VISIBLE_DEVICES=0,1,2,3 python source_pretrain_pre_a.py -ds dukemtmc -dt market1501 -a resnet50 --seed 2 --margin 0.0 
    --num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 80 --eval-step 40 
    --logs-dir logs/dukemtmcTOmarket1501/resnet50-pretrain-2-pre-a`
    
#### Post-A
First Network:

`CUDA_VISIBLE_DEVICES=0,1,2,3 python source_pretrain_post_a.py -ds dukemtmc -dt market1501 -a resnet50 --seed 1 --margin 0.0 
    --num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 80 --eval-step 40 
    --logs-dir logs/dukemtmcTOmarket1501/resnet50-pretrain-1-post-a`

Second Network:

`CUDA_VISIBLE_DEVICES=0,1,2,3 python source_pretrain_post_a.py -ds dukemtmc -dt market1501 -a resnet50 --seed 2 --margin 0.0 
    --num-instances 4 -b 64 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 200 --epochs 80 --eval-step 40 
    --logs-dir logs/dukemtmcTOmarket1501/resnet50-pretrain-2-post-a`
    
### Step 2: Train our attention model on target domain (Stage 1)
#### Pre-A

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_pre_a_s1.py -dt market1501 -a resnet50 --num-clusters 500 --num-instances 4 --lr 0.00035 --iters 800 -b 64 --epochs 40 --soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --init-1 logs/dukemtmcTOmarket1501/resnet50-pretrain-1-pre-a/model_best.pth.tar --init-2 logs/dukemtmcTOmarket1501/resnet50-pretrain-2-pre-a/model_best.pth.tar --logs-dir logs/dukemtmcTOmarket1501/resnet50-train-pre-a-s1`

#### Post-A

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_post_a_s1.py -dt market1501 -a resnet50 --num-clusters 500 --num-instances 4 --lr 0.00035 --iters 800 -b 64 --epochs 10 --soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --init-1 logs/dukemtmcTOmarket1501/resnet50-pretrain-1-post-a/model_best.pth.tar --init-2 logs/dukemtmcTOmarket1501/resnet50-pretrain-2-post-a/model_best.pth.tar --logs-dir logs/dukemtmcTOmarket1501/resnet50-train-post-a-s1`

### Step 3: Train on target domain (Stage 2)
#### Pre-A

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_pre_a_s2.py -dt market1501 -a resnet50 --num-clusters 500 --num-instances 4 --lr 0.00035 --iters 800 -b 64 --epochs 80 --soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --init-1 logs/dukemtmcTOmarket1501/resnet50-train-pre-a-s1/model1_checkpoint.pth.tar --init-2 logs/dukemtmcTOmarket1501/resnet50-train-pre-a-s1/model2_checkpoint.pth.tar --logs-dir logs/dukemtmcTOmarket1501/resnet50-train-pre-a-s2`

#### Post-A

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_post_a_s2.py -dt market1501 -a resnet50 --num-clusters 500 --num-instances 4 --lr 0.00035 --iters 800 -b 64 --epochs 80 --soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --init-1 logs/dukemtmcTOmarket1501/resnet50-train-post-a-s1/model1_checkpoint.pth.tar --init-2 logs/dukemtmcTOmarket1501/resnet50-train-post-a-s1/model2_checkpoint.pth.tar --logs-dir logs/dukemtmcTOmarket1501/resnet50-train-post-a-s2`

## Test

We use Duke-to-Market as an example, other UDA tasks will follow similar pipelines.

### Pre-A
`CUDA_VISIBLE_DEVICES=0 python test_pre_a.py -b 256 -j 8 --dataset-target market1501 -a resnet50 --resume logs/dukemtmcTOmarket1501/resnet50-train-pre-a-s2/model_best.pth.tar`

### Post-A
`CUDA_VISIBLE_DEVICES=0 python test_post_a.py -b 256 -j 8 --dataset-target market1501 -a resnet50 --resume logs/dukemtmcTOmarket1501/resnet50-train-post-a-s2/model_best.pth.tar`



