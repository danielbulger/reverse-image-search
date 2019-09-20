from argparse import ArgumentParser

import torch
from data.dataset import get_images
from PIL import Image

from torch.autograd import Variable
from torchvision.transforms import ToTensor
from annoy import AnnoyIndex


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--cuda', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to train using CUDA")
    parser.add_argument('--model', type=str, help='The torch model to use')
    parser.add_argument('--data', type=str, help='The folder of images to read')

    return parser.parse_args()


# Create the ANNOY indexes from a given directory
def main():
    args = parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise Exception('CUDA Device not found')

    device = torch.device('cuda' if args.cuda else 'cpu')
    model = torch.load(args.model)
    model.eval()

    annoy = AnnoyIndex(4096)

    for image in get_images(args.data):
        input_image = Image.open(image).convert('RGB')
        input_tensor = Variable(ToTensor()(input_image))
        input_tensor = input_tensor.view(1, -1, input_image.size[1], input_image.size[0])

        if args.cuda:
            input_tensor = input_tensor.to(device)
            model = model.to(device)

        output = model(input_tensor)
        if args.cuda:
            output = output.cpu()

        # Add the embedding to the index.
        annoy.add_item(output)

    annoy.build(20)
    annoy.save('embeddings.ann')


if __name__ == '__main__':
    main()
