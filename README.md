
## Description
This framework is helpful in training Deep learning based image classification task on general academic datasets. It currently supports features of training/validation accuracy and loss plotting, model weight storage and mutli GPU distributed training. Following models are supported as of now:

```
1] InceptionV3
2] Xception
3] VGG_19
4] Resnet18
5] Resnet34
6] Resnet50
7] Resnet101
8] Resnet152
9] DesneNet121
10] ResNeXt101-32
11] ResNeXt101-64
12] MobileNetV2
```

Following Image classification datasets are currently supported:

```
1] MNIST
2] CIFAR10
3] CIFAR100
4] Fashion-MNIST
5] SVHN
6] STL10
7] Caltech256
8] ImageNet
```

Data transforms are supported through torchvision.


## Usage

Firstly create `Model_storage` and `data` folders in the repo root directory. The former folder is used to store training results for a specific run id along with the training configs used. The latter folder is used to store the datasets. Use `tools/train.py` for training any of the above mentioned DL models. Create a desired config file under `configs` directory using the `configs/conig_template.py` provided. While attempting to run the script for a given dataset for the first time set the `download` config under `dataset_cfg/id_cfg` as `True` to download the dataset (except for Caltech256 and ImageNet).

## Dependencies 

- `Pytorch 1.0.0` or higher
- `Torchvision 0.4.0` or higher
- [`Sequential-imagenet-dataloader`](https://github.com/BayesWatch/sequential-imagenet-dataloader)  
- [`Tensorpack`](https://github.com/tensorpack/tensorpack)
- `OpenCV 4.0` or higher
- `Tqdm`
- `Lmdb`
- `TensorFlow 1.3` or higher and < 2
- `Matplotlib 3.1.1` or higher
- [`Pretrainedmodels`](https://github.com/Cadene/pretrained-models.pytorch)


