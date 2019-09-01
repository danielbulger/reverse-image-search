import torch
from model.DeepRank import DeepRank
from PIL import Image

from torch.autograd import Variable
from torchvision.transforms import ToTensor


def main():
    device = torch.device('cuda')
    model = DeepRank().cuda(device)

    input_image = Image.open('test.jpg').convert('RGB')
    input_image = input_image.resize((224, 224), Image.BICUBIC)
    input_tensor = Variable(ToTensor()(input_image))
    input_tensor = input_tensor.view(1, -1, input_image.size[1], input_image.size[0])

    model(input_tensor.cuda())


if __name__ == '__main__':
    main()
