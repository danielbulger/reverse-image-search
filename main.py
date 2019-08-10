import torch
from model.DeepRank import DeepRank


def main():
    device = torch.device('cuda')
    model = DeepRank().cuda(device)


if __name__ == '__main__':
    main()
