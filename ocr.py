#!/usr/bin/env python3
import cv2
import numpy as np
import os
import random

print('OpenCV Version ' + cv2.__version__)


def sizer(shape, factor):
    """Scales image dimensions equally."""
    return tuple((int(factor * a) for a in shape[:2]))


def get_image(fname):
    """Loads an image by filename, resizes it, converts to greyscale."""
    image = cv2.imread(fname)
    image = cv2.resize(image, sizer(image.shape, 0.3))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.reshape(image.size, 1)


def load_images(img_file_list):
    """Load a bunch of images into a numpy array."""
    gen = (get_image(fname) for fname in img_file_list)
    return np.array(list(gen)).astype(np.float32)


def shuffle_data(labels, data):
    """Randomizes sample order such that labels[i] still corresponds
    to data[i] for every i."""
    together = list(zip(labels, data))
    random.shuffle(together)
    return zip(*together)


def load_data(dirpath):
    """Loads images in the /data_{train,test} folders."""
    files = os.listdir(dirpath)
    img_labels = []
    img_file_list = []
    for f in files:
        if f == 'random':
            continue
        path = os.path.join(dirpath, f)
        if os.path.isdir(path):
            imgs = os.listdir(path)
            for i in imgs:
                if 'png' not in i:
                    continue
                img_labels.append(int(f))
                img_file_list.append(os.path.join(path, i))

    img_labels, img_file_list = shuffle_data(img_labels, img_file_list)
    images = load_images(img_file_list)
    return images, np.array(img_labels), img_file_list



def train():
    """Trains the k-Nearest Neighbour."""
    print('Loading images for training...')
    train_data, train_labels, _ = load_data('data_train')
    assert len(train_data) == len(train_labels)
    print(f'Loaded {len(train_data)} images for training')

    knn = cv2.ml.KNearest_create()
    print('Training the model...')
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    print('Training finished...')

    return knn


def test(knn):
    """Tests the accuracy of supplied k-Nearest Neighbour."""
    print('Loading images for testing...')
    test_data, test_labels, test_flist = load_data('data_test')
    assert len(test_data) == len(test_labels)
    print(f'Loaded {len(test_data)} images for testing')

    # k=4 seems to yield the best accuracy
    ret, result, neighbours, dist = knn.findNearest(test_data, k=4)
    assert len(result) == len(test_labels)

    total_tests = len(result)
    correct = 0
    for i in range(total_tests):
        if float(test_labels[i]) == result[i][0]:
            correct += 1

    percentage = 100.0 * correct / total_tests
    print(f'{correct}/{total_tests} correct, {percentage} %')


def main():
    knn = train()
    test(knn)


if '__main__' == __name__:
    main()
