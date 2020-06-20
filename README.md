# SemGAN
Tensorflow version implementation of SemGAN


This repository contains the TensorFlow code of Semantics-aware GAN(SemGAN) method for performing semantic-aware domain adaptation tasks. The architecture of SemGAN network is:

![Architecture_SemGAN](/images/SemGAN.png)

## Getting started with the code

## Prepare for the datasets

For training sim2real domain adaptation SemGAN model, the selected datasets for training are: **GTA dataset**, **Cityscapes dataset**. The alternative datasets for testing are: **Virtual KITTI dataset**, **KITTI dataset** and **GIBSON**.

- Download the **GTA** images and labels:

```
$ wget https://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip
$ unzip 01_images.zip
$ wget https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip
$ unzip 01_labels.zip
```
Downloading the **Cityscapes** dataset requires registration on the website: https://www.cityscapes-dataset.com/dataset-overview/. After downloading the datasets(images and labels), put the images and the corresponding labels from each domain in four different folders, and then put these four folders in one folder named **GTA2Cityscapes**.

- Create the csv file as input to the data loader:
1. Edit the semgan_datasets.py file. For example: if the downloaded datasets contain 666 GTA images, 666 GTA labels, 888 Cityscapes images and 888 Cityscapes labels, then the file can be formulated as following:
  ```
DATASET_TO_SIZES = {
    'GTA2Cityscapes_train': 888,
    'GTA2Cityscapes_test': 888,
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'GTA2Cityscapes_train': '.jpg',
    'GTA2Cityscapes_test': '.jpg',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'GTA2Cityscapes_train': './input/GTA2Cityscapes/GTA2Cityscapes_train.csv',
    'GTA2Cityscapes_test': './input/GTA2Cityscapes/GTA2Cityscapes_test.csv',
}
```
2. Run create_semgan_datasets.py
  ```
  python -m create_semgan_datasets --image_path_a='./input/GTA2Cityscapes/GTA_imgs' --image_path_b='./input/GTA2Cityscapes/Cityscapes_imgs' --label_path_a='./input/GTA2Cityscapes/GTA_labels' --label_path_b='./input/GTA2Cityscapes/Cityscapes_labels'  --dataset_name="GTA2Cityscapes_train" --do_shuffle=0
  ```
  This will create a .csv file from the path of the dataset folders. This .csv file is the input of the data_loader.py file.
  
## Training
- Create the configuration file. The configuration file contains basic information for training/testing. An example of the configuration file could be found at configs/exp_01.json. Edit the critical items for training and testing the model.
``` 
  "pool_size": 50,
  "base_lr":0.0002,
  "max_step": 200,
  "dataset_name": "GTA2Cityscapes_train",
  "do_flipping": 1,
  "_LAMBDA_B": 10,
  "_LAMBDA_OBJ_A": 0,
  "_LAMBDA_OBJ_B": 0,
  "_LAMBDA_SEM_A": 10,
  "_LAMBDA_SEM_B": 10
```
- Start training:
``` 
 python main.py  --to_train=1 --log_dir=./output/SemGAN/exp_01 --config_filename=./configs/exp_01.json
 ``` 
