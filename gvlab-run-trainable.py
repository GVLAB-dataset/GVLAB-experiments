import argparse
import json
import os

import torch
from torch import nn
torch.autograd.set_detect_anomaly(True)

from config import TRAIN, TRAIN_RESULTS_PATH, MODEL_RESULTS_PATH
from models.gvlab_backend import BackendModel
from models.gvlab_trainable import BaselineModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import save_model, dump_train_info, get_split

device_ids = [0, 1, 2, 3]

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--lr', help='learning rate', default=0.001, type=float)
    # parser.add_argument('-bz', '--batch_size', default=128, type=int)
    parser.add_argument('-bz', '--batch_size', default=4, type=int)
    parser.add_argument('-ne', '--n_epochs', default=3, type=int)
    parser.add_argument('-s', '--split', default='train') # Train, Dev, Test.
    parser.add_argument('-rs', '--result_suffix', default="", required=False, help='suffix to add to results name')
    parser.add_argument("--debug", action='store_const', default=False, const=True)
    parser.add_argument("--multi_gpu", action='store_const', default=False, const=True)
    parser.add_argument("--test_model", action='store_const', default=False, const=True)
    parser.add_argument('--load_epoch', default=2)
    parser.add_argument('--model_backend_type', default='vit', help="vit", required=False)
    parser.add_argument('--backend_version', default='1.0.0', help="version", required=False)
    args = parser.parse_args()

    if args.multi_gpu:
        print(f"Multiplying batch_size by # GPUs: {len(device_ids)}")
        initial_batch_size = args.batch_size
        args.batch_size *= len(device_ids)
        print(f"initial_batch_size: {initial_batch_size}, new batch_size: {args.batch_size}")
    return args


class Loader(Dataset):
    def __init__(self, data, backend_model):
        self.data = data
        self.backend_model = backend_model

    def __getitem__(self, index):
        row = self.data[index]
        input_image_vector = self.backend_model.load_and_encode_img(row['image'])
        text_vector = self.backend_model.encode_text(row['cue'])

        return input_image_vector, text_vector, row['label']

    def __len__(self):
        return len(self.data)

def main(args):
    data = get_split(args.split)
    backend_model = BackendModel()
    baseline_model = BaselineModel(backend_model).to(device)
    print(f"Checking baseline model cuda: {next(baseline_model.parameters()).is_cuda}")
    if args.multi_gpu:
        baseline_model = nn.DataParallel(baseline_model)
    # class_weights = torch.FloatTensor([4.0]).to(device)
    # loss_fn = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if args.test_model is False:
        train(backend_model, baseline_model, data, loss_fn)


def get_experiment_dir(args):
    if not os.path.exists(MODEL_RESULTS_PATH):
        os.makedirs(MODEL_RESULTS_PATH)
    if not os.path.exists(TRAIN_RESULTS_PATH):
        os.makedirs(TRAIN_RESULTS_PATH)

    model_dir_path = os.path.join(TRAIN_RESULTS_PATH, f"model_backend_{args.model_backend_type}_{args.backend_version.replace('/', '-')}_{args.split}")

    if args.debug:
        model_dir_path += "_DEBUG"
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    json.dump(args.__dict__, open(os.path.join(model_dir_path, 'args.json'), 'w'))
    return model_dir_path

def train(backend_model, baseline_model, data, loss_fn):
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=args.lr)
    train_dataset = Loader(data, backend_model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    train_loop(args=args, model=baseline_model, optimizer=optimizer, train_loader=train_loader, loss_fn=loss_fn,
               n_epoch=args.n_epochs)


def train_loop(args, model, optimizer, train_loader, loss_fn, n_epoch):
    """
    Parameters
    ----------
    args : (argparse.Namespace) arguments
    model :(nn.Module) The baseline model
    optimizer :(torch.optim.Optimizer)
    train_loader :(DataLoader) for the train set
    loss_fn : Loss function
    n_epoch :(int) epoch number
    """
    all_losses = {TRAIN: []}
    model_dir_path = get_experiment_dir(args)
    print(f"model_dir_path: {model_dir_path}")

    for epoch in tqdm(range(n_epoch)):
        epoch_train_losses = train_epoch(loss_fn, model, optimizer, train_loader, epoch)
        all_losses[TRAIN].append(epoch_train_losses)

        dev_accuracy_list = dump_train_info(args, model_dir_path, all_losses, epoch=epoch)
        save_model(model_dir_path, epoch, model, dev_accuracy_list)


def train_epoch(loss_fn, model, optimizer, train_loader, epoch):
    """
    Runs training on a single epoch
    Parameters
    ----------
    loss_fn : Loss function
    model : (nn.Module) The baseline model
    optimizer :(torch.optim.Optimizer)
    train_loader : (DataLoader)
    epoch : (int) epoch number
    Returns
    -------
    The epoch losses
    """
    model.train()
    epoch_train_losses = []

    with tqdm(enumerate(train_loader), total=len(train_loader)) as epochs:
        epochs.set_description(f'Training epoch {epoch}, split: {args.split}')

        for batch_idx, batch_data in epochs:

            # Forward pass
            input_img, input_text, label = batch_data
            label = label.to(device)
            out = model(input_img, input_text).squeeze()

            y = label.squeeze()
            optimizer.zero_grad()

            # Compute Loss
            loss = loss_fn(out.double(), y.double())
            epoch_train_losses.append(loss.item())
            # Backward pass
            loss.backward()
            optimizer.step()

    return epoch_train_losses

if __name__ == '__main__':
    args = get_args()
    main(args)
