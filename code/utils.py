import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


# get training data
def get_train_data(root):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # Iterate through folders, one folder corresponds to one category

    cag_root = os.path.join(root, "CAG")
    cnag_root = os.path.join(root, "CNAG")

    cagfiles = [name for name in os.listdir(cag_root)
               if name.endswith('.bmp')]
    cnagfiles = [name for name in os.listdir(cnag_root)
                if name.endswith('.bmp')]
    allfiles = []
    for f in cagfiles:
        real_url = os.path.join(cag_root, f)
        allfiles.append(real_url)
    for f in cnagfiles:
        real_url = os.path.join(cnag_root, f)
        allfiles.append(real_url)

    return allfiles
    
    
# Get the split data
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  #  Guaranteed reproducible random results
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    lesion_class = ["CAG","CNAG"]
    # sort， to ensure consistent order
    lesion_class.sort()
    # Generate category names and corresponding numeric indexes
    class_indices = dict((k, v) for v, k in enumerate(lesion_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # Store all image paths of the train set
    train_images_label = []  # Store all image label of the train set
    val_images_path = []  # Store all image paths of the Validation set
    val_images_label = []  #  Store all image label of the Validation set
    every_class_num = []  # Store the total number of samples for each category
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".BMP"]  # Supported file suffix types

    train_path = os.path.join(root, 'train')
    val_path = os.path.join(root, 'val')

    # Processing the training set
    for cla in lesion_class:
        cla_path = os.path.join(train_path, cla)

        # Iterate through to get the paths of all files of the supported types
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        # Get the index corresponding to the category
        image_class = class_indices[cla]
        # Record the number of samples in the category
        every_class_num.append(len(images))

        train_images_path = train_images_path+images
        train_images_label =train_images_label+  [image_class] * len(images)

    # Processing the Validation set
    for cla in lesion_class:
        cla_path = os.path.join(val_path, cla)

        # Iterate through to get the paths of all files of the supported types
        images2 = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        # Get the index corresponding to the category
        image_class2 = class_indices[cla]
        # Record the number of samples in the category
        every_class_num.append(len(images2))

        val_images_path = val_images_path + images2
        val_images_label = val_images_label + [image_class2] * len(images2)


    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    return train_images_path, train_images_label, val_images_path, val_images_label

  


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # Anti-Normalize
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # remove the x
            plt.yticks([])  # remove the y
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # accumulated losses
    accu_num = torch.zeros(1).to(device)   #  cumulative number of correctly predicted samples
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # cumulative number of correctly predicted samples
    accu_loss = torch.zeros(1).to(device)  # accumulated losses

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
