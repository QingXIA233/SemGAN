"""Create datasets for training and testing."""
import csv
import os
import random

import click

import semgan_datasets


def create_list(foldername, fulldir=True, suffix=".png"):
    """
    :param foldername: The full path of the folder.
    :param fulldir: Whether to return the full path or not.
    :param suffix: Filter by suffix.
    :return: The list of filenames in the folder with given suffix.
    """
    file_list_tmp = os.listdir(foldername)
    file_list = []
    if fulldir:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(item)
    return file_list



@click.command()
@click.option('--image_path_a',
              type=click.STRING,
              default='./input/GTA2Cityscapes/GTA_imgs',
              help='The path to the images from domain_a.')
@click.option('--image_path_b',
              type=click.STRING,
              default='./input/GTA2Cityscapes/Cityscapes_imgs',
              help='The path to the images from domain_b.')
@click.option('--dataset_name',
              type=click.STRING,
              default='GTA2Cityscapes_train',
              help='The name of the dataset for SemGAN.')
@click.option('--do_shuffle',
              type=click.BOOL,
              default=False,
              help='Whether to shuffle images when creating the dataset.')

def create_dataset(image_path_a, image_path_b, dataset_name, do_shuffle):
    list_img_a = create_list(image_path_a, True,
                         semgan_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    list_img_b = create_list(image_path_b, True,
                         semgan_datasets.DATASET_TO_IMAGETYPE[dataset_name])

    output_path = semgan_datasets.PATH_TO_CSV[dataset_name]

    num_rows = semgan_datasets.DATASET_TO_SIZES[dataset_name]
    all_data_tuples = []
    for i in range(num_rows):
        all_data_tuples.append((
            list_img_a[i % len(list_img_a)],
            list_img_b[i % len(list_img_b)]
        ))

    if do_shuffle is True:
        random.shuffle(all_data_tuples)
        
    with open(output_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for data_tuple in enumerate(all_data_tuples):
            csv_writer.writerow(list(data_tuple[1]))


if __name__ == '__main__':
    create_dataset()
