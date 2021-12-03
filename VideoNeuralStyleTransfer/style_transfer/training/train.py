'''
Train a custom style transfer network

:author: Matthew Schofield
Based on: https://github.com/pytorch/examples/tree/master/fast_neural_style/neural_style
'''
import os
import sys
import time

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch

import style_transfer.training.utils as utils
from style_transfer.transformer_net import TransformerNet
from style_transfer.training.vgg import Vgg16

if __name__ == "__main__":
    # Path to a COCO dataset of base images to train from
    dataset_path = "../../raw_data/COCO_2014"
    # Whether to use GPU
    cuda = True
    # Style image to fit the network to
    style_image = "../../raw_data/style_transfer_training_images/glacier1.jpg"
    # Number of epochs, warning a single epoch can take over 3 hours on a regular desktop
    epochs = 2
    # Directory to save checkpoints and model to
    checkpoint_model_dir = "chckpt_models/"

    checkpoint_interval = 2000
    log_interval = 100

    # Avoid changing these, but these tweak the performance of the network
    image_size = 256
    style_size = 256

    # Weight to apply to the content and style image in the loss function
    content_weight = 1e5
    style_weight = 1e10
    # Images are trained in batches of 4
    batch_size = 4
    # standard learning rate
    lr = 1e-3

    try:
        if checkpoint_model_dir is not None and not (os.path.exists(checkpoint_model_dir)):
            os.makedirs(checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

    if cuda:
        if not torch.cuda.is_available():
            print("ERROR: cuda is not available, try running on CPU")
            sys.exit(1)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(dataset_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    transformer = TransformerNet(None, None, False).to(device)
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(style_image, size=style_size)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(epochs):
        print(e)
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if checkpoint_model_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

        # save model
        transformer.eval().cpu()
        save_model_filename = "epoch_" + str(e) + ".model"
        save_model_path = os.path.join(checkpoint_model_dir, save_model_filename)
        torch.save(transformer.state_dict(), save_model_path)

        print("\nDone, trained model saved at", save_model_path)
