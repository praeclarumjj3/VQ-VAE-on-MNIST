import torch
from torchvision.utils import make_grid, save_image
from datetime import datetime
from modules import VectorQuantizedVAE
from torchvision import transforms, datasets


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde


def main(args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    if args.input == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        # Define the train & test dataSets
        test_set = datasets.MNIST("MNIST", train=False,
                                  download=True, transform=transform)

        # Define the data loaders
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=100, shuffle=True,
                                                  pin_memory=True)

        real_batch, _ = next(iter(test_loader))

    else:
        real_batch = torch.normal(mean=0, std=1, size=(100, 1, 28, 28))

    model = VectorQuantizedVAE(1, args.hidden_size_vae, args.k).to(args.device)
    with open(args.model, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    generated = generate_samples(real_batch, model, args)
    save_image(make_grid(generated, nrow=10), './generatedImages/{0}.png'.format(args.filename))

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Generated Image')

    parser.add_argument('--model', type=str,
                        help='filename containing the model')

    parser.add_argument('--input', type=str, default="MNIST",
                        help='MNIST or random')

    # Latent space
    parser.add_argument('--hidden-size-vae', type=int, default=40,
                        help='size of the latent vectors')
    parser.add_argument('--k', type=int, default=512,
                        help='number of latent vectors (default: 512)')

    # Miscellaneous
    parser.add_argument('--filename', type=str,
                        help='name with which file is to be saved')
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
                               if torch.cuda.is_available() else 'cpu')

    if not os.path.exists('./generatedImages'):
        os.makedirs('./generatedImages')

    main(args)

