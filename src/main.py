import torch.optim as optim
from torchsummary import summary
import json
import argparse
from tensorboardX import SummaryWriter

from load_data import *
from model.SphereFace import SphereFace
from model.helper import train
from losses.AngularSoftmaxWithLoss import AngularSoftmaxWithLoss


class Options:
    """Configure train model parameters."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SphereFace')
        self.parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                                 help='input batch size for training (default: 32)')
        self.parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                                 help='input batch size for testing (default: 32)')
        self.parser.add_argument('--print-every-batch', type=int, default=50, metavar='N',
                                 help='how many batches to wait before logging training status')
        self.parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                                 help='learning rate (default: 1e-2)')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                 help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--weight-decay', type=float, default=1e-5, metavar='W',
                                 help='SGD weight decay (default: 1e-5)')
        self.parser.add_argument('--log-dir', default='./runs/exp-0',
                                 help='path of data for save log.')
        self.parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                 help='number of epochs to train (default: 10)')
        self.parser.add_argument('--save-model-folder', default='./save',
                                 help='save folder.')

    def parse(self):
        return vars(self.parser.parse_args())


def main():
    # data = load_fetch_olivetti_faces()
    # knn = KNN(3, pca())
    # knn.training(dataset=data['images'], labels=data['labels'], input_shape=256)
    # predict = knn.predict(data['images'][-1, :].reshape(1, -1))
    # print(predict)
    op = Options()
    args = op.parse()

    with open('./model.json') as json_file:
        model_json = json.load(json_file)

        # train_data, test_data = load_MNIST_data(**args)
        train_data = load_images_from_folder(f'D:\CelebA\splitted_img_celeba', **args)
        test_data = None
        input_shape = train_data.dataset.__getitem__(0)[0].size()
        n_classes = len(train_data.dataset.classes)
        model = SphereFace(model_json, input_shape=input_shape, n_classes=n_classes, **args)
        criterion = AngularSoftmaxWithLoss()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(model)
        summary(model, input_data=input_shape, device='cpu', depth=6)

        optimizer = optim.SGD(model.parameters(),
                              lr=args['lr'],
                              momentum=args['momentum'],
                              weight_decay=args['weight_decay'],
                              nesterov=True)
        with SummaryWriter('runs/exp-0') as writer:
            input_data = torch.autograd.Variable(torch.rand(32, *input_shape))
            writer.add_graph(model, (input_data,))
            train(model, criterion, optimizer, train_data, test_data, **args)


if __name__ == "__main__":
    main()
