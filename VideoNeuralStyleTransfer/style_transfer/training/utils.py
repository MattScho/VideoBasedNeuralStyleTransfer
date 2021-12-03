'''
Utilities for training the style transfer network

Adapted from: https://github.com/pytorch/examples/tree/master/fast_neural_style/
'''
from PIL import Image

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size), Image.ANTIALIAS)
    return img

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std