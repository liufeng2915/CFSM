## Guided Face Synthesis for Face Recognition

Our pre-trained CFSMs could be a plug-in to any SoTA Face recognition model. Here, we provide a training example based on the [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) Pytorch repo. 

## Training

* Prepare the training data and add the data path to the configuration file.
* Train the model:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.0.0.1" \
--master_port=12348 train.py configs/arcface_ms1mv2_r50_ours
```

## Pretrained Models

| Method                                                       | Arch | Train Datasets | Link                                                         |
| ------------------------------------------------------------ | ---- | -------------- | ------------------------------------------------------------ |
| [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)+CFSM | R50  | MS1MV2         | [gdrive](https://drive.google.com/file/d/1RGKYYYZQ5lu_aagVwBV-Dso70wGru936/view?usp=sharing), [baidudrive](https://pan.baidu.com/s/17LRveDdXk7Zmdye-mrdc_g) (wyiv) |
| [AdaFace](https://github.com/mk-minchul/AdaFace)+CFSM        | R100 | WebFace12M     | [gdrive](https://drive.google.com/file/d/1QBk6oBIE5HlW5MDXsHNxcY0JTs_jDRx4/view?usp=sharing), [baidudrive](https://pan.baidu.com/s/1_2Tn0tlbfOmH-99qjKH-wA) (hatp) |

## Validation

* IJB-B dataset

  | Method       | Arch | Train Dataset | TAR@FAR=0.001% | TAR@FAR=0.01% | TAR@FAR=0.1% | Rank1 | Rank5 |
  | :----------- | ---- | ------------- | -------------- | ------------- | ------------ | ----- | ----- |
  | ArcFace      | R50  | MS1MV2        | 87.26          | 94.01         | 95.95        | 94.61 | 96.52 |
  | ArcFace+CFSM | R50  | MS1MV2        | 90.95          | 94.61         | 96.21        | 94.96 | 96.84 |

* IJB-C dataset

  | Method       | Arch | Train Dataset | TAR@FAR=0.0001% | TAR@FAR=0.001% | TAR@FAR=0.01% | Rank1 | Rank5 |
  | :----------- | ---- | ------------- | --------------- | -------------- | ------------- | ----- | ----- |
  | ArcFace      | R50  | MS1MV2        | 87.24           | 93.32          | 95.61         | 95.89 | 97.08 |
  | ArcFace+CFSM | R50  | MS1MV2        | 89.34           | 94.06          | 95.90         | 96.31 | 97.48 |

* TinyFace dataset

  | Method       | Arch | Train Dataset | Rank1 | Rank5 |
  | :----------- | ---- | ------------- | ----- | ----- |
  | ArcFace      | R50  | MS1MV2        | 62.21 | 66.85 |
  | ArcFace+CFSM | R50  | MS1MV2        | 63.01 | 68.21 |
  | AdaFace      | R100 | WebFace12M    | 72.29 | 74.97 |
  | AdaFace+CFSM | R100 | WebFace12M    | 73.87 | 76.77 |

* IJB-S dataset

| Method       | Arch | Train Dataset | V2S Rank1 | V2S Rank5 | V2S 1% | V2S 10% | V2B Rank1 | V2B Rank5 | V2B 1% | V2B 10% | V2V Rank1 | V2V Rank5 | V2V 1% | V2V 10% |
| ------------ | ---- | ------------- | --------- | --------- | ------ | ------- | --------- | --------- | ------ | ------- | --------- | --------- | ------ | ------- |
| ArcFace      | R50  | MS1MV2        | 58.78     | 66.40     | 40.99  | 50.45   | 60.66     | 67.43     | 43.12  | 51.38   | 14.81     | 26.72     | 2.51   | 5.72    |
| ArcFace+CFSM | R50  | MS1MV2        | 63.86     | 69.95     | 47.86  | 56.44   | 65.95     | 71.16     | 47.28  | 57.24   | 21.38     | 35.11     | 2.96   | 11.84   |
| AdaFace      | R100 | WebFace12M    | 71.35     | 76.24     | 59.40  | 66.34   | 71.93     | 76.56     | 59.37  | 66.68   | 36.71     | 50.03     | 4.62   | 11.84   |
| AdaFace+CFSM | R100 | WebFace12M    | 72.54     | 77.59     | 60.94  | 66.02   | 72.65     | 78.18     | 60.26  | 65.88   | 39.14     | 50.91     | 5.05   | 13.17   |
