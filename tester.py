import torch
from torch.autograd import Variable
import argparse
import trainer
import model
import data_loader


def scoring_criterion(loader, model, criterion):
    score = 0
    model.eval()
    for data, labels in loader:
        data = Variable(data)
        outputs = model(data)
        score += criterion(outputs, labels)
    model.train()
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for JDD pig identification")
    parser.add_argument('--model-path', type=str, default='./models/CNN_1.pth', help='location of train model')
    parser.add_argument('--test-dir', type=str, default='./test/frames', help='location of test data')
    parser.add_argument('--batch-size', type=int, default=64)

    args = parser.parse_args()
    print("Preparing test data loader...")
    test_loader = data_loader.make_dataloader(args.test_dir, batch_size=args.batch_size)
    print("Loading trained model parameters")

    model = model.CNN()
    model.load_state_dict(args.model_path)
    test_acc = trainer.test_model(test_loader, model)
    criterion = torch.nn.NLLLoss()
    loss = scoring_criterion(test_loader, model, criterion)
    print("Test Accuracy:")
    print(test_acc)
    print("Scoring Criteria:")
    print(loss / len(test_loader.dataset))

