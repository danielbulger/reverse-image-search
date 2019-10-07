from argparse import ArgumentParser

from model.DeepRank import DeepRank
from model.TripletLoss import TripletLoss
from data import dataset
import torch
import torch.nn
import torch.optim
import torch.utils.data
import os


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--epochs', type=int, required=True, help='Number of iterations to run over the dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='The Learning Rate')
    parser.add_argument('--log-dir', default="./log/", type=str, help="Directory to save checkpoint logs")
    parser.add_argument('--checkpoint', default=50, type=int, help='Number of iterations between each checkpoint')
    parser.add_argument('--dataset', type=str, required=True, help="Path containing the Image data set")
    parser.add_argument('--workers', type=int, help='Number of worker threads to use')
    parser.add_argument('--batch-size', type=int, required=True, help='Number of samples per batch')
    parser.add_argument('--crop-size', type=int, default=256, help='Dimensions to crop the images to')
    parser.add_argument('--cuda', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to train using CUDA")

    return parser.parse_args()


def checkpoint(model, directory, file):
    path = os.path.join(directory, file)
    torch.save(model, path)


def main():
    args = parse_args()

    # If the number of workers wasn't set, use all available ones
    if args.workers is None:
        import multiprocessing
        args.workers = multiprocessing.cpu_count()

    if args.cuda and not torch.cuda.is_available():
        raise Exception('CUDA Device not found')

    device = torch.device('cuda' if args.cuda else 'cpu')

    model = DeepRank()
    if args.cuda:
        model = model.cuda(device)

    triplet_loss = TripletLoss(args.batch_size, device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    data_loader = torch.utils.data.DataLoader(
        dataset.TripletDataSet(
            args.dataset,
            1000,
            args.crop_size
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    for epoch in range(args.epochs):
        running_loss = 0.0

        for index, data in enumerate(data_loader):

            # Move this to the GPU if available
            if args.cuda:
                data = data.to(device)

            optimiser.zero_grad()
            prediction = model(data)

            # Calculate the triplet loss, we don't have a target so set as none.
            loss = triplet_loss(prediction, None)
            loss.backward()

            optimiser.step()

            # Sum the loss from this batch into the checkpoint total
            running_loss += loss.item()

            if index != 0 and index % args.checkpoint == 0:
                checkpoint(model, args.log_dir, "checkpoint-{}-{}.pth".format(epoch, index))
                print('[%d,%d] loss: %.6f' % (epoch, index, running_loss / args.batch_size))
                running_loss = 0.0


if __name__ == '__main__':
    main()
