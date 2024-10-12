import os
import json
import torch
import argparse
import matplotlib
matplotlib.use("Agg")
import cv2
from tqdm import tqdm
from config import system_configs
from model_factory import Network
from db.datasets import datasets
from test_model import testing
import json

# Ensure that cuDNN benchmarks are not used for consistent CPU-based execution
torch.backends.cudnn.benchmark = False

def load_net(test_iter, config_name, data_dir, cache_dir):
    print(f"Loading {config_name} model")
    config_file = os.path.join(system_configs.config_dir, config_name + ".json")
    with open(config_file, "r") as f:
        configs = json.load(f)
    print(f"Configuration file loading complete")
    configs["system"]["snapshot_name"] = config_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["dataset"] = "Chart"
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split
    print("Config loading complete")
    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }["validation"]

    test_iter = system_configs.max_iter if test_iter is None else test_iter
    print(f"Loading parameters at iteration: {test_iter}")
    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)
    print("Building neural network...")
    nnet = Network()
    print("Loading parameters...")
    nnet.load_model(test_iter)
    # Always run on CPU
    print("Running on CPU...")
    return db, nnet

def pre_load_nets(model_type, data_dir, cache_dir, iteration):
    methods = {}
    print(f"Preloading {model_type} model")
    db, nnet = load_net(iteration, model_type, data_dir, cache_dir)
    methods[model_type] = [db, nnet, testing]
    return methods

def get_groups(keys, cens, group_scores):
    thres = 0.4
    groups = []
    group_scores_ = group_scores
    group_thres = 0.4
    for category in range(3):
        keys_trim = [p for p in keys[category] if p[0] > thres]
        cens_trim = [p for p in cens[category] if p[0] > thres]
        group_scores = group_scores_[:len(cens_trim), len(cens_trim):len(keys_trim) + len(cens_trim)]
        if len(cens_trim) == 0 or len(keys_trim) < 2: continue
        if category == 1:
            for i in range(len(cens_trim)):
                group = []
                vals = []
                cen = cens_trim[i]
                group += [cen[2], cen[3]]
                for j in range(len(keys_trim)):
                    val = group_scores[i][j].item()
                    if val > group_thres:
                        key = keys_trim[j]
                        group += [key[2], key[3]]
                        vals.append(val)
                if len(vals) == 0: continue
                group.append(sum(vals) / len(vals))
                group.append(category)
                groups.append(group)
            continue
        if category == 0:
            vals, inds = torch.topk(group_scores, 2)
        elif category == 2:
            vals, inds = torch.topk(group_scores, 3)
            group_thres = 0.1
        for i in range(len(cens_trim)):
            if (vals[i] > group_thres).sum().item() == vals.size(1):
                group = []
                cen = cens_trim[i]
                group += [cen[2], cen[3]]
                for ind in inds[i]:
                    key = keys_trim[ind]
                    group += [key[2], key[3]]
                group.append(vals[i].mean().item())
                group.append(category)
                groups.append(group)
    return groups

def test(image_path, model_type):
    image = cv2.imread(image_path)
    with torch.no_grad():
        results = methods[model_type][2](image, methods[model_type][0], methods[model_type][1])
        if model_type == 'KPDetection':
            keys, centers = results[0], results[1]
            thres = 0.
            keys = {k: [p for p in v.tolist() if p[0] > thres] for k, v in keys.items()}
            centers = {k: [p for p in v.tolist() if p[0] > thres] for k, v in centers.items()}
            return (keys, centers)
        if model_type == 'KPGrouping':
            keys, centers, group_scores = results
            keys = {k: [p for p in v.tolist()] for k, v in keys.items()}
            centers = {k: [p for p in v.tolist()] for k, v in centers.items()}
            groups = get_groups(keys, centers, group_scores)
            return (keys, centers, groups)

def parse_args():
    parser = argparse.ArgumentParser(description="Test the ChartReader extraction part.")
    parser.add_argument("--save_path", default="tmp/", type=str)
    parser.add_argument("--model_type", default="Grouping", type=str)
    parser.add_argument("--trained_model_iter", default='50000', type=str)
    parser.add_argument("--data_dir", default="data/extraction_data", type=str)
    parser.add_argument('--cache_path', default="data/cache/", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    methods = pre_load_nets(args.model_type, args.data_dir, args.cache_path, args.trained_model_iter)
    save_path = os.path.join(args.save_path, args.model_type + args.trained_model_iter + '.json')
    rs_dict = {}
    images = os.listdir(methods[args.model_type][0]._image_dir)
    print(f"Predicting with {args.model_type} net")
    for img in tqdm(images):
        path = os.path.join(methods[args.model_type][0]._image_dir, img)
        if cv2.imread(path) is not None:
            data = test(path, args.model_type)
            rs_dict[img] = data
    json.dump(rs_dict, open(save_path, 'w'))
    print(f'Results saved at: {save_path}')
