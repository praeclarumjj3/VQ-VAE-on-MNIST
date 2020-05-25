import numpy as np
import torch
import torch.nn.functional as F
import json
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import torchvision.utils as vutils
from datetime import datetime
from modules import VectorQuantizedVAE, GatedPixelCNN
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def train(data_loader, model, prior, optimizer, args, writer):
    for images, labels in data_loader:
        with torch.no_grad():
            images = images.to(args.device)
            latents = model.encode(images)
            latents = latents.detach()

        labels = labels.to(args.device)
        logits = prior(latents, labels)
        if args.steps == 1:
            writer.add_graph(model=prior, input_to_model=(latents, labels))  # get model structure on tensorboard
        logits = logits.permute(0, 2, 3, 1).contiguous()

        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, args.k),
                               latents.view(-1))
        loss.backward()

        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)

        if args.steps % 1000 == 0:
            print('loss/train: {:f} at step {:f}'.format(loss.item(), args.steps))

        optimizer.step()
        args.steps += 1


def test(data_loader, model, prior, args, writer):
    with torch.no_grad():
        loss = 0.
        for images, labels in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            latents = model.encode(images)
            latents = latents.detach()
            logits = prior(latents, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss += F.cross_entropy(logits.view(-1, args.k),
                                    latents.view(-1))

        loss /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test', loss.item(), args.steps)

    if args.steps % 1000 == 0:
        print('loss/test: {:f} at step {:f}'.format(loss.item(), args.steps))

    return loss.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde


def main(args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    writer = SummaryWriter('./logs_pixelCNN/{0}'.format(args.output_folder))
    save_filename = './models_pixelCNN/{0}'.format(args.output_folder)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    # Define the train & test dataSets
    train_set = datasets.MNIST(args.data_folder, train=True,
                               download=True, transform=transform)
    test_set = datasets.MNIST(args.data_folder, train=False,
                              transform=transform)
    num_channels = 1

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=16, shuffle=True)

    # Save the label encoder
    with open('./models_pixelCNN/{0}/labels.json'.format(args.output_folder), 'w') as f:
        json.dump(train_set.class_to_idx, f)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(num_channels, args.hidden_size_vae, args.k).to(args.device)
    with open(args.model, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    model.eval()

    prior = GatedPixelCNN(args.k, args.hidden_size_prior,
                          args.num_layers, n_classes=10).to(args.device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr)

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction at start', grid, 0)

    best_loss = -1.
    img_list = []
    for epoch in range(args.num_epochs):
        train(train_loader, model, prior, optimizer, args, writer)

        loss = test(test_loader, model, prior, args, writer)

        reconstruction = generate_samples(fixed_images, model, args)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image('reconstruction at epoch {:f}'.format(epoch + 1), grid, epoch + 1)
        print("loss = {:f} at epoch {:f}".format(loss, epoch + 1))
        writer.add_scalar('loss/testing_loss', loss, epoch + 1)
        img_list.append(grid)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(prior.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(prior.state_dict(), f)

    real_batch = next(iter(train_loader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Original Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(args.device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='PixelCNN Prior for VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
                        help='name of the data folder')

    parser.add_argument('--model', type=str,
                        help='filename containing the model')

    # Latent space
    parser.add_argument('--hidden-size-vae', type=int, default=256,
                        help='size of the latent vectors (default: 256)')
    parser.add_argument('--hidden-size-prior', type=int, default=64,
                        help='hidden size for the PixelCNN prior (default: 64)')
    parser.add_argument('--k', type=int, default=512,
                        help='number of latent vectors (default: 512)')
    parser.add_argument('--num-layers', type=int, default=15,
                        help='number of layers for the PixelCNN prior (default: 15)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate for Adam optimizer (default: 3e-4)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='prior',
                        help='name of the output folder (default: prior)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
                        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs_pixelCNN'):
        os.makedirs('./logs_pixelCNN')
    if not os.path.exists('./models_pixelCNN'):
        os.makedirs('./models_pixelCNN')
    # Device
    args.device = torch.device(args.device
                               if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models_pixelCNN/{0}'.format(args.output_folder)):
        os.makedirs('./models_pixelCNN/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
