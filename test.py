import torch
from torch.autograd import Variable
import argparse
import os
import model
import data_loader
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pig identification")
    parser.add_argument('--model', type=str, default='./models/CNN_1.pth', help='location of model')
    parser.add_argument('--test-dir', type=str, default='./test_A', help='location of test data')
    parser.add_argument('--output', type=str, default='./output.csv')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)

    # read test data
    print("Loading test data...")
    test_loader = data_loader.make_dataloader(args.test_dir, batch_size=1)
    file_names = sorted([s.split('.')[0] for s in os.listdir(os.path.join(args.test_dir, 'frames'))])

    # read model
    print("Loading trained model...")
    model = model.CNN()
    model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    model.eval()

    print("Generating predictions...")
    for i, (data, labels) in tqdm(enumerate(test_loader)):
        print(file_names[i])
        data = Variable(data)
        output = torch.exp(model(data)).data.squeeze()

        with open(args.output, 'a') as f:
            for j in range(30):
                f.write("%s, %d, %.8f\n" % (file_names[i], j + 1, output[j]))
