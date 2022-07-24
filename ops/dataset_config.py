# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os


ROOT_DATASET = None


def return_ucf101(modality):
    global ROOT_DATASET
    ROOT_DATASET = "path/datasets/"
    filename_categories = 101
    if modality == "RGB":
        root_data = ROOT_DATASET + "UCF101/UCF101_frames"
        filename_imglist_train = "UCF101/label/ucf101_rgb_train_split_3.txt"
        filename_imglist_val = "UCF101/label/ucf101_rgb_val_split_3.txt"
        prefix = "img_{:05d}.jpg"
    elif modality == "Flow":
        root_data = ROOT_DATASET + "UCF101/jpg"
        filename_imglist_train = "UCF101/file_list/ucf101_flow_train_split_1.txt"
        filename_imglist_val = "UCF101/file_list/ucf101_flow_val_split_1.txt"
        prefix = "flow_{}_{:05d}.jpg"
    else:
        raise NotImplementedError("no such modality:" + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    global ROOT_DATASET
    ROOT_DATASET = "path/datasets/"
    filename_categories = 51
    if modality == "RGB":
        root_data = ROOT_DATASET + "HMDB51/HMDB51_frames"
        filename_imglist_train = "HMDB51/splits/hmdb51_rgb_train_split_1.txt"
        filename_imglist_val = "HMDB51/splits/hmdb51_rgb_val_split_1.txt"
        prefix = "img_{:05d}.jpg"
    elif modality == "Flow":
        root_data = ROOT_DATASET + "HMDB51/images"
        filename_imglist_train = "HMDB51/splits/hmdb51_flow_train_split_1.txt"
        filename_imglist_val = "HMDB51/splits/hmdb51_flow_val_split_1.txt"
        prefix = "flow_{}_{:05d}.jpg"
    else:
        raise NotImplementedError("no such modality:" + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    global ROOT_DATASET
    ROOT_DATASET = "path/datasets/"
    filename_categories = 174
    if modality == "RGB":
        root_data = ROOT_DATASET + "sth-v1/20bn-something-something-v1"
        filename_imglist_train = "sth-v1/label/train.txt"
        filename_imglist_val = "sth-v1/label/val.txt"
        prefix = "{:05d}.jpg"
    elif modality == "Flow":
        root_data = ROOT_DATASET + "something/v1/20bn-something-something-v1-flow"
        filename_imglist_train = "something/v1/train_videofolder_flow.txt"
        filename_imglist_val = "something/v1/val_videofolder_flow.txt"
        prefix = "{:06d}-{}_{:05d}.jpg"
    else:
        print("no such modality:" + modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_somethingv2(modality):
    global ROOT_DATASET
    ROOT_DATASET = "path/datasets/"
    filename_categories = 64
    if modality == "RGB":
        root_data = ROOT_DATASET + ""
        filename_imglist_train = "train.txt"
        filename_imglist_val = "test.txt"
        prefix = "img_{:05d}.png"
    elif modality == "Flow":
        root_data = ROOT_DATASET + "20bn-something-something-v2-flow"
        filename_imglist_train = "data/tsm_labels/train_videofolder.txt"
        filename_imglist_val = "data/tsm_labels/val_videofolder.txt"
        prefix = "{:05d}.jpg"
    else:
        raise NotImplementedError("no such modality:" + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester(modality):
    global ROOT_DATASET
    ROOT_DATASET = "path/datasets/"
    filename_categories = 100
    if modality == "RGB":
        prefix = "{:05d}.jpg"
        root_data = ROOT_DATASET + "jester/20bn-jester-v1"
        filename_imglist_train = "jester/train_videofolder.txt"
        filename_imglist_val = "jester/val_videofolder.txt"
    else:
        raise NotImplementedError("no such modality:" + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    global ROOT_DATASET
    ROOT_DATASET = "path/datasets/"
    filename_categories = 400
    if modality == "RGB":
        root_data = ROOT_DATASET + "k400/"
        filename_imglist_train = "k400/label/k400_train_list.txt"
        filename_imglist_val = "k400/label/k400_val_list.txt"
        prefix = "img_{:05d}.png"
    else:
        raise NotImplementedError("no such modality:" + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics600(modality):
    global ROOT_DATASET
    ROOT_DATASET = "path/datasets/"
    filename_categories = 600
    if modality == "RGB":
        root_data = ROOT_DATASET + "k600/"
        filename_imglist_train = "k600/label/k600_train.txt"
        filename_imglist_val = "k600/label/k600_val.txt"
        prefix = "img_{:05d}.jpg"
    else:
        raise NotImplementedError("no such modality:" + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {
        "jester": return_jester,
        "something": return_something,
        "somethingv2": return_somethingv2,
        "ucf101": return_ucf101,
        "hmdb51": return_hmdb51,
        "kinetics": return_kinetics,
        "kinetics600": return_kinetics600,
    }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError("Unknown dataset " + dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print("{}: {} classes".format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix