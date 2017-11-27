import torch
from torch.autograd import Variable
import argparse
import os
import model
import data_loader
from time import time


def test_model(loader, model):
    """
    Help function that tests the models's performance on a dataset
    :param: loader: data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, labels in loader:
        if args.cuda:
            data, labels = data.cuda(), labels.cuda()
        data = Variable(data)
        outputs = model(data)
        predicted = (outputs.max(1)[1].data.long()).view(-1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        break
    model.train()
    return 100 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch model on Quora Paraphrase Detection")
    parser.add_argument('--train-dir', type=str, default='./train/frames', help='location of train data')
    parser.add_argument('--valid-dir', type=str, default='./validation/frames', help='location of dev data')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--shuffle', action='store_true', help='whether or not to shuffle data')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--save', type=str, default='./models/', help='path to store models')
    parser.add_argument('--interval', type=int, default=50, help='batch interval to report accuracy')
    parser.add_argument('--batch-size', type=int, default=16)

    args = parser.parse_args()
    print(args)

    # set random seed
    torch.manual_seed(args.seed)

    num_epochs = args.epochs

    print("Preparing data loader...")
    train_loader = data_loader.make_dataloader(args.train_dir, batch_size=args.batch_size)
    valid_loader = data_loader.make_dataloader(args.valid_dir, batch_size=args.batch_size)

    print("Defining models, loss function, optimizer...")
    # define models, loss, optimizer
    model = model.CNN()

    if args.cuda:
        model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    print("beginning training...")
    # training the Model
    train_acc_history = []
    validation_acc_history = []
    for epoch in range(num_epochs):
        start = time()
        for i, (data, labels) in enumerate(train_loader):
            if args.cuda:
                data, labels = data.cuda(), labels.cuda()
            data, labels = Variable(data), Variable(labels)

            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # report performance
            if i % args.interval == 0:
                train_acc = test_model(train_loader, model)
                val_acc = test_model(valid_loader, model)
                print('Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Train Acc: {5}, Validation Acc:{6}'.format(
                    epoch + 1, num_epochs, i + 1, len(train_loader), loss.data[0],
                    train_acc, val_acc))
                train_acc_history.append(train_acc)
                validation_acc_history.append(val_acc)
        print("Epoch %d, time taken = %.4f" % (epoch + 1, time() - start))
        torch.save(model.state_dict(), os.path.join(args.save, "CNN_%d.pth" % epoch))
    print("Train Accuracy:")
    print(train_acc_history)
    print("Validation Accuracy:")
    print(validation_acc_history)
