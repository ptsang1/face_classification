import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from time import time
from evaluations.AverageMeter import AverageMeter


def train(model: nn.Module,
          criterion,
          optimizer,
          train_loader,
          test_loader,
          epochs=1,
          print_every_batch=1,
          device='cpu', **kwangs):

    for epoch in range(1, epochs+1):
        adjust_learning_rate(optimizer, epoch, **kwangs)
        train_one_epoch(model, criterion, optimizer, train_loader, epoch, epochs, device, print_every_batch)
        print(f"Epoch=[{epoch}/{epochs}]", end='\t')
        if test_loader:
            validation(model, criterion, test_loader)
        torch.save(model.state_dict(), f"{kwangs['save_model_folder']}/model_{epoch}.pt")


def train_one_epoch(model: nn.Module,
                    criterion,
                    optimizer: optim.Optimizer,
                    train_loader: DataLoader,
                    epoch,
                    n_epochs,
                    device,
                    print_every_batch=1):

    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    n_train_batches = len(train_loader)

    model.train()
    for batch_id, (inputs, labels) in enumerate(train_loader):
        start = time()
        n_samples = inputs.size(0)
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()  # clear the gradients of all optimized variables
        features, outputs = model(inputs)  # forward pass
        loss = criterion(outputs, labels)  # calculate loss
        loss.backward()  # backward pass
        optimizer.step()  # perform a single optimization step (update params)

        batch_time.update(time() - start)
        losses.update(loss.item(), n_samples)

        acc = accuracy(outputs[0], labels)
        accuracies.update(acc*100, n_samples)

        if print_every_batch > 0 and (batch_id % print_every_batch == 0 or batch_id == n_train_batches):
            print(f"Epoch=[{epoch}/{n_epochs}][{batch_id}/{n_train_batches}]\t"
                  f"Training loss={losses.avg:.3f}\t"
                  f"Acc={accuracies.avg:.3f}\t"
                  f"Training time={batch_time.val:.3f}, {batch_time.avg:.3f}")


def validation(model, criterion, test_loader):
    losses = AverageMeter()
    accuracies = AverageMeter()
    model.eval()
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = Variable(test_inputs), Variable(test_labels)

            features, outputs = model(test_inputs)
            batch_loss = criterion(outputs, test_labels)

            losses.update(batch_loss.item(), test_inputs.size(0))

            acc = accuracy(outputs[0], test_labels)
            accuracies.update(acc, test_inputs.size(0))

        print(f"Validation loss={losses.avg:.3f}\t"
              f"Acc={accuracies.avg:.3f}\t")


def accuracy(outputs, targets):
    with torch.no_grad():
        top_p, top_class = outputs.topk(1, 1, True, True)
        equals = top_class == targets.view(*top_class.shape)
        return torch.mean(equals.type(torch.FloatTensor))


def adjust_learning_rate(optimizer, epoch, **kwangs):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    m = epoch // 5
    lr = kwangs['lr'] * (0.1 ** m)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
