import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from time import time
from evaluations.AverageMeter import AverageMeter
from os import path, mkdir

def train(model: nn.Module,
          criterion,
          optimizer,
          train_loader,
          test_loader,
          epochs=1,
          print_every_batch=1,
          start_epoch=1,
          device="cuda:0", **kwangs):

    for epoch in range(start_epoch, epochs+1):
        adjust_learning_rate(optimizer, epoch, **kwangs)
        print(optimizer.param_groups[0]['lr'])
        train_one_epoch(model, criterion, optimizer, train_loader, epoch, epochs, device, print_every_batch)
        if not path.exists(f"{kwangs['save_model_folder']}"):
          mkdir(f"{kwangs['save_model_folder']}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, f"{kwangs['save_model_folder']}/latest.pt")
        print(f"Epoch=[{epoch}/{epochs}]", end='\t')
        if test_loader:
            validation(model, criterion, test_loader, device)


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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        n_samples = inputs.size(0)
        inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
        optimizer.zero_grad()  # clear the gradients of all optimized variables

        start.record()
        _, outputs = model(inputs)  # forward pass
        # outputs = model(inputs)
        loss = criterion(outputs, labels)  # calculate loss
        loss.backward()  # backward pass
        optimizer.step()  # perform a single optimization step (update params)
        end.record()
    
        losses.update(loss.item(), n_samples)
        acc = accuracy(outputs[0], labels)
        # acc = accuracy(outputs, labels)
        accuracies.update(acc*100, n_samples)

        torch.cuda.synchronize()

        batch_time.update(start.elapsed_time(end))
        
        if print_every_batch > 0 and (batch_id % print_every_batch == 0 or batch_id == n_train_batches):
            print(f"Epoch=[{epoch}/{n_epochs}][{batch_id}/{n_train_batches}]\t"
                  f"Training loss={losses.avg:.3f}\t"
                  f"Acc={accuracies.avg:.3f}\t"
                  f"Training time={batch_time.val:.3f}, {batch_time.avg:.3f}")


def validation(model, criterion, test_loader, device):
    losses = AverageMeter()
    accuracies = AverageMeter()
    model.eval()
    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = Variable(test_inputs).to(device), Variable(test_labels).to(device)

            features, outputs = model(test_inputs)
            # outputs = model(test_inputs)
            batch_loss = criterion(outputs, test_labels)

            losses.update(batch_loss.item(), test_inputs.size(0))

            acc = accuracy(outputs[0], test_labels)
            # acc = accuracy(outputs, test_labels)*100
            accuracies.update(acc*100, test_inputs.size(0))

        print(f"Validation loss={losses.avg:.3f}\t"
              f"Acc={accuracies.avg:.3f}\t")


def accuracy(outputs, targets):
    with torch.no_grad():
        _, preds = torch.max(outputs, 1)
        # _, top_class = outputs.topk(1, dim=1)
        # equals = targets.eq(top_class.view(*targets.shape))
        # return torch.mean(equals.type(torch.FloatTensor))
        return torch.mean((preds == targets.data).type(torch.FloatTensor))
        # _, predicted = torch.max(outputs.data, 1)
        # return predicted.eq(targets.data).cpu().type(torch.FloatTensor).mean()


def adjust_learning_rate(optimizer, epoch, **kwangs):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    m = epoch // 50
    lr = kwangs['lr'] * (0.1 ** m)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr