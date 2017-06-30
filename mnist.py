from PIL import Image
from scipy.ndimage import rotate
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def get_mnist():
    return input_data.read_data_sets('MNIST_data')


def to_img(array, degree):
    return Image.fromarray(np.array(rotate(np.reshape(array,
                                                      [28, 28]),
                                           degree,
                                           reshape=False)
                                    *255, np.int32)).convert('L')


def rotate_data(data, degree):
    n, _ = data.shape
    steps = 360/degree
    string = './data/mnist/train/{:06d}.png'
    for j in range(n):
        if j/n % 10 == 0:
            print str(j/float(n))
        for k in range(steps):
            img = to_img(data[j], degree*k)
            img.save(string.format(j*steps + k))


def save_data(data):
    n, _ = data.shape
    string = './data/mnist/train_norotate/{:06d}.png'
    for j in range(n):
        if j/n % 10 == 0:
            print str(j/float(n))
            img = to_img(data[j], 0)
            img.save(string.format(j))
