import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from modules import VectorQuantizedVAE
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def train(data_loader, model, optimizer, args, writer):
    for images, _ in data_loader:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        if (args.steps % 1000 == 0):
            print('loss/training_loss: {:f} at step {:f}'.format(loss.item(), args.steps))

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)
        writer.add_scalar('loss/train/training_loss', loss.item(), args.steps)

        optimizer.step()
        args.steps += 1


def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde


def main(args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = './models/{0}'.format(args.output_folder)

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # Define the train & test dataSets
    train_set = datasets.MNIST(args.data_folder, train=True,
                               download=True, transform=transform)
    test_set = datasets.MNIST(args.data_folder, train=False,
                              download=True, transform=transform)
    num_channels = 1

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=16, shuffle=False)

    # Fixed images for TensorBoard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer.add_graph(model, fixed_images.to(args.device))  # get model structure on tensorboard

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction at start', grid, 0)

    img_list = []
    best_loss = -1.
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args, writer)
        loss, _ = test(test_loader, model, args, writer)

        reconstruction = generate_samples(fixed_images, model, args)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)

        writer.add_image('reconstruction at epoch {:f}'.format(epoch + 1), grid, epoch + 1)
        print("loss = {:f} at epoch {:f}".format(loss, epoch + 1))
        writer.add_scalar('loss/testing_loss', loss, epoch + 1)
        img_list.append(grid)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)

    # Grab a batch of real images from the dataloader
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

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str, default="VQ-VAE",
                        help='name of the data folder')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512,
                        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
                        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device
                               if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
