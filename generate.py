import torch
from torchvision.utils import make_grid, save_image
from datetime import datetime
from modules import VectorQuantizedVAE, GatedPixelCNN

N_LAYERS = 15
IMAGE_SHAPE = (28, 28)
K = 512
DIM = 64


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde


def generate_samplesWithPixelCNN(model):
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.long().cuda()

    x_tilde = model.generate(label, shape=IMAGE_SHAPE, batch_size=100)
    images = x_tilde.cpu().data.float() / (K - 1)

    save_image(
        images[:, None],
        './generatedImages/{0}.png'.format(args.filename),
        nrow=10
    )


def post_process(images):
    images = images.numpy()
    return images


def main(args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    random_images = torch.normal(mean=0.5, std=0.5, size=(100, 1, 28, 28))

    if args.name_of_model == "VQ-VAE":
        model = VectorQuantizedVAE(1, args.hidden_size_vae, args.k).to(args.device)
        with open(args.model, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
        generated = generate_samples(random_images, model, args)
        save_image(make_grid(generated, nrow=10), './generatedImages/{0}.png'.format(args.filename))

    else:
        model = GatedPixelCNN(K, DIM, N_LAYERS).cuda()
        with open(args.model, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
        generate_samplesWithPixelCNN(model)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Generated Image')

    parser.add_argument('--model', type=str,
                        help='filename containing the model')

    parser.add_argument('--name_of_model', type=str, default="VQ-VAE",
                        help='VQ-VAE or PixelCNN')

    # Latent space
    parser.add_argument('--hidden-size-vae', type=int, default=256,
                        help='size of the latent vectors (default: 256)')
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

