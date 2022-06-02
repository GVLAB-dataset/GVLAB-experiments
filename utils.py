# ------------------------------Imports--------------------------------
import json

import torch
import numpy as np
import os
import pandas as pd
import pickle

# ------------------------------Code--------------------------------
from config import TEST, DEV, TRAIN, SWOW_SPLIT_PATH


def calculate_accuracy(out_prob, y):
    prob = torch.softmax(out_prob, dim=1)
    out_np = prob.detach().cpu().numpy()
    labels_np = y.detach().cpu().numpy()
    accuracy = (np.argmax(out_np, 1) == labels_np).mean()
    predictions = [float(x) for x in np.argmax(out_np, 1)]
    labels = [float(x) for x in labels_np]
    return accuracy, predictions, labels


def save_model(model_dir_path, epoch, model, dev_accuracy_list):
    out_p = os.path.join(model_dir_path, f"epoch_{epoch}.pth")
    print(f"Saving model path to... {out_p}")
    torch.save(model.state_dict(), out_p)
    # if dev_accuracy_list[-1] == max(dev_accuracy_list):
    #     out_p = os.path.join(model_dir_path, f"epoch_BEST.pth")
    #     print(f"Saving BEST model path to... {out_p}")
    #     print(dev_accuracy_list)
    #     torch.save(model.state_dict(), out_p)


# def dump_test_info(args, model_dir_path, all_losses, all_test_accuracy, test_df, epoch):
#     test_losses_mean = {i: np.mean(v) for i, v in enumerate(all_losses['test'])}
#     test_accuracy_mean = {i: np.mean(v) for i, v in enumerate(all_test_accuracy)}
#     test_info = pd.concat(
#         [pd.Series(test_losses_mean, name='test loss'), pd.Series(test_accuracy_mean, name='test accuracy')], axis=1)
#     out_p = os.path.join(model_dir_path, f'epoch_{epoch}_test')
#     if args.result_suffix != '':
#         out_p += "_" + args.result_suffix
#     all_losses_out_p = out_p + '_all_losses_test.pickle'
#     out_p_test_df = out_p + "_test_df.csv"
#     out_p += ".csv"
#     test_info.to_csv(out_p)
#     test_df.to_csv(out_p_test_df)
#     all_losses_and_acc_d = {'all_losses': all_losses, 'all_test_accuracy': all_test_accuracy}
#     with open(all_losses_out_p, 'wb') as f:
#         pickle.dump(all_losses_and_acc_d, f)
#     print(f'Dumping losses {len(test_info)} to {all_losses_out_p}')
#     print(test_info)
#     print(f'Dumping df {len(test_info)} to {out_p}, and {len(test_df)} to {out_p_test_df}')


def dump_train_info(args, model_dir_path, all_losses, epoch):
    train_losses_mean = {i: np.mean(v) for i, v in enumerate(all_losses['train'])}
    train_info = pd.concat([pd.Series(train_losses_mean, name='train loss')], axis=1)
    out_p = os.path.join(model_dir_path, f'epoch_{epoch}')
    if args.result_suffix != '':
        out_p += "_" + args.result_suffix
    all_losses_out_p = out_p + '_all_losses.pickle'
    out_p += ".csv"
    train_info.to_csv(out_p)
    all_losses_and_acc_d = {'all_losses': all_losses}
    with open(all_losses_out_p, 'wb') as f:
        pickle.dump(all_losses_and_acc_d, f)
    print(f'Dumping losses {len(train_info)} to {all_losses_out_p}')
    print(train_info)
    print(f'Dumping df {len(train_info)} to {out_p}')
    return list(train_info['train loss'].values)



def get_split(args):
    f = open(SWOW_SPLIT_PATH)
    split = json.load(f)
    return split
