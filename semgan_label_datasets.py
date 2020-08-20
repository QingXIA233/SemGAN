"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'GTA2Cityscapes_labels_train': 500,
    'GTA2Cityscapes_labels_test': 10,
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'GTA2Cityscapes_labels_train': '.png',
    'GTA2Cityscapes_labels_test': '.png',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'GTA2Cityscapes_labels_train': './input/GTA2Cityscapes_labels/GTA2Cityscapes_labels_train.csv',
    'GTA2Cityscapes_labels_test': './input/GTA2Cityscapes_labels/GTA2Cityscapes_labels_test.csv',
}
