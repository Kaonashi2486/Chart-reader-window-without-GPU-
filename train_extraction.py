import os
import json
import torch
import argparse
from sampling_function import sample_data
import traceback
import re
import numpy as np
from tqdm import tqdm
from config import system_configs
from model_factory import Network
from db.datasets import datasets
import time
from torch.multiprocessing import Process, Queue

# Remove GPU-related configurations
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False

import wandb

# Prefetching function for loading data asynchronously
def prefetch_data(db, queue, sample_data, data_aug):
    ind = 0
    print("Starting data prefetching process...")
    np.random.seed(os.getpid())  # Use process ID as the random seed
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            print(f'An error occurred during data prefetching: {e}')
            traceback.print_exc()

# Pin memory is used for faster CPU-GPU transfer, skip in CPU-only mode
def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        try:
            data = data_queue.get()
            data["xs"] = [x for x in data["xs"]]  # No pin_memory
            data["ys"] = [y for y in data["ys"]]  # No pin_memory
            pinned_data_queue.put(data)
            if sema.acquire(blocking=False):
                return
        except Exception as e:
            print(f"Error in pin_memory: {e}")
            pass

# Initialize multiprocessing for prefetching
def init_parallel_jobs(dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

# Main training loop
def train(training_db, validation_db, start_iter=0):
    learning_rate = system_configs.learning_rate
    max_iter = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    val_iter = system_configs.val_iter
    decay_rate = system_configs.decay_rate
    stepsize = system_configs.stepsize
    val_ind = 0

    print("Initializing model...")
    nnet = Network()

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("The requested pretrained model does not exist.")
        print("Loading pretrained model...")
        nnet.load_pretrained_model(pretrained_model)

    if start_iter == -1:
        save_list = os.listdir(system_configs.snapshot_dir)
        save_list = [f for f in save_list if f.endswith('.pkl')]
        save_list.sort(reverse=True, key=lambda x: int(x.split('_')[1][:-4]))
        if len(save_list) > 0:
            target_save = save_list[0]
            start_iter = int(re.findall(r'\d+', target_save)[0])
            learning_rate /= (decay_rate ** (start_iter // stepsize))
            nnet.load_model(start_iter)
        else:
            start_iter = 0
        nnet.set_lr(learning_rate)
    else:
        nnet.set_lr(learning_rate)

    print(f"Starting training from iter {start_iter + 1}, LR: {learning_rate}...")
    total_training_loss = []
    ind = 0
    error_count = 0

    optimizer = nnet.optimizer
    device = "cpu"  # Force CPU mode
    nnet.to(device)

    best_val_loss = float('inf')
    for iteration in tqdm(range(start_iter + 1, max_iter + 1)):
        try:
            training, ind = sample_data(training_db, ind)
            training_data = [d.to(device) if isinstance(d, torch.Tensor) else d for d in training.values()]

            optimizer.zero_grad()
            training_loss = nnet.train_step(*training_data)
            training_loss.backward()  # No mixed precision; directly backward pass

            optimizer.step()
            total_training_loss.append(training_loss.item())
        except:
            print('Data extraction error occurred.')
            traceback.print_exc()
            error_count += 1
            if error_count > 10:
                print('Too many extraction errors. Terminating...')
                time.sleep(1)
                break
            continue

        if iteration % 500 == 0:
            avg_training_loss = sum(total_training_loss) / len(total_training_loss)
            print(f"Training loss at iter {iteration}: {avg_training_loss}")
            wandb.log({"train_loss": avg_training_loss})
            total_training_loss = []

        if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
            validation, val_ind = sample_data(validation_db, val_ind)
            validation_data = [d.to(device) if isinstance(d, torch.Tensor) else d for d in validation.values()]
            validation_loss = nnet.validate_step(*validation_data)
            wandb.log({"val_loss": validation_loss.item()})
            print(f"Validation loss at iter {iteration}: {validation_loss.item()}")
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                print(f"New best validation loss: {best_val_loss}. Saving model...")
                nnet.save_model("best")

        if iteration % stepsize == 0:
            learning_rate /= decay_rate
            nnet.set_lr(learning_rate)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model with the given configs.")
    parser.add_argument("--cfg_file", dest="cfg_file", default="KPDetection", type=str)
    parser.add_argument("--start_iter", dest="start_iter", default=0, type=int)
    parser.add_argument("--pretrained_model", dest="pretrained_model", default="KPDetection.pkl", type=str)
    parser.add_argument("--threads", dest="threads", default=1, type=int)
    parser.add_argument("--cache_path", dest="cache_path", default="./data/cache/", type=str)
    parser.add_argument("--data_dir", dest="data_dir", default="./data", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    wandb.init(
        project="ChartLLM-Extraction",
        name="bar only",
        group="grouping",
        notes="Test KP Grouping with Only Bars-No GPU",
        tags=["ChartLLM", "KP Grouping"],
        config=args
    )

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["data_dir"] = args.data_dir
    configs["system"]["cache_dir"] = args.cache_path
    configs["system"]["dataset"] = "Chart"
    configs["system"]["snapshot_name"] = args.cfg_file

    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split

    dataset = system_configs.dataset
    threads = args.threads
    training_db = datasets[dataset](configs["db"], train_split)
    validation_db = datasets[dataset](configs["db"], val_split)

    print("Current system configuration:")
    print(system_configs.full)

    train(training_db, validation_db, args.start_iter)