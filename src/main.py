import torch.optim as optim
from torchsummary import summary
import json
import argparse
from tensorboardX import SummaryWriter

from load_data import *
from model.SphereFace import SphereFace, sphere20a
from model.helper import train
from losses.AngularSoftmaxWithLoss import AngularSoftmaxWithLoss
import warnings
warnings.filterwarnings("ignore")
import torchvision.models as models


class Options:
    """Configure train model parameters."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SphereFace')
        self.parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                                 help='input batch size for training (default: 32)')
        self.parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                                 help='input batch size for testing (default: 32)')
        self.parser.add_argument('--print-every-batch', type=int, default=50, metavar='N',
                                 help='how many batches to wait before logging training status')
        self.parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                                 help='learning rate (default: 1e-2)')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                 help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--weight-decay', type=float, default=0.05, metavar='W',
                                 help='SGD weight decay (default: 1e-5)')
        self.parser.add_argument('--log-dir', default='./runs/exp-0',
                                 help='path of data for save log.')
        self.parser.add_argument('--epochs', type=int, default=10, metavar='N',
                                 help='number of epochs to train (default: 10)')
        self.parser.add_argument('--save-model-folder', default='/content/drive/MyDrive/face_classification/save',
                                 help='save folder.')
        self.parser.add_argument('--cont-training', default=None,
                                 help='save folder.')
        self.parser.add_argument('--model-file', default='./model.json',
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

    with open(args['model_file']) as json_file:
        model_json = json.load(json_file)

        # train_data, test_data = load_MNIST_data(**args)
        train_data = load_images_from_folder('/content/drive/MyDrive/celeb_data/train', **args)
        test_data = load_images_from_folder('/content/drive/MyDrive/celeb_data/test', train=False, **args)
        n_classes = 1000

        # train_data, test_data, n_classes = load_fetch_olivetti_faces(**args)

        print('n_classes = ', n_classes)
        print(train_data.dataset.__getitem__(0)[1])

        input_shape = train_data.dataset.__getitem__(0)[0].size()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model = SphereFace(model_json, input_shape=input_shape, n_classes=n_classes, **args).to(device)
        # model = models.googlenet().to(device)  
        model = sphere20a(n_classes).to(device)
        criterion = AngularSoftmaxWithLoss().to(device)
        # criterion= torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(),
                              lr=args['lr'],
                              momentum=args['momentum'],
                              weight_decay=args['weight_decay'],
                              nesterov=True)
        print(model)
        # summary(model, input_data=input_shape, device=device, depth=6)
        if not args['cont_training'] is None:
            checkpoint = torch.load(args['cont_training'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        with SummaryWriter('runs/exp-0') as writer:
            # input_data = torch.autograd.Variable(torch.rand(32, *input_shape)).to(device)
            # writer.add_graph(model, (input_data,))
            if args['cont_training'] is None:
              train(model, criterion, optimizer, train_data, test_data, device=device, **args)
            else:
              train(model, criterion, optimizer, train_data, test_data, start_epoch=start_epoch, device=device, **args)



if __name__ == "__main__":
    main()
