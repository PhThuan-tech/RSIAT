import sys
import logging
import copy
import torch
import glob
import re
from utils import model_factory
from data.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random


def RSIAT_train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    sum_seed = 0.0
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        sum_seed += _train(args)
    avg_seed = sum_seed / len(seed_list)
    print('Average Seed Accuracy (CNN):', avg_seed)
    logging.info("Average Seed Accuracy (CNN): {}".format(avg_seed))

def find_latest_checkpoint(checkpoint_dir):
    """
    Tìm file task_N.pkl có N lớn nhất.
    """
    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, "task_*.pkl")
    )

    if not checkpoint_files:
        return None

    def get_task_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"task_(\d+)\.pkl$", filename)

        if match is None:
            return -1

        return int(match.group(1))

    checkpoint_files.sort(key=get_task_number)
    return checkpoint_files[-1]

def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = model_factory.get_model(args["model_name"], args)

    checkpoint_dir = os.path.join(
        "ckpt",
        str(args["prefix"]),
        str(args["dataset"]),
        "{}_{}".format(args["init_cls"], args["increment"]),
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    start_task = 0

    # Có thể bật/tắt bằng "resume": true/false trong file config.
    resume_enabled = args.get("resume", False)

    if resume_enabled:
        resume_path = args.get("resume_path")

        # Nếu không chỉ định file cụ thể, tự tìm task_N.pkl mới nhất.
        if not resume_path:
            resume_path = find_latest_checkpoint(checkpoint_dir)

        if resume_path and os.path.isfile(resume_path):
            completed_task = model.load_checkpoint(resume_path)
            start_task = completed_task + 1

            logging.info(
                "Resuming experiment from task %d",
                start_task,
            )
        else:
            logging.info(
                "Resume enabled but no checkpoint was found. "
                "Starting from task 0."
            )

    print()

    cnn_curve = getattr(model, "cnn_curve", {"top1": [], "top5": []})
    nme_curve = getattr(model, "nme_curve", {"top1": [], "top5": []})

    if start_task >= data_manager.nb_tasks:
        if cnn_curve["top1"]:
            avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            logging.info("All tasks already completed. Average Accuracy (CNN): {}".format(avg_acc))
            return avg_acc

        logging.info("All tasks already completed, but no CNN curve was found.")
        return 0.0

    for task in range(start_task, data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        model.incremental_train(data_manager)
        cnn_accy = model.eval_task()
        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])
        model.cnn_curve = cnn_curve
        model.nme_curve = nme_curve

        checkpoint_path = os.path.join(
        checkpoint_dir,
            "task_{}.pkl".format(task),
        )

        model.save_checkpoint(checkpoint_path)


        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
    return sum(cnn_curve["top1"])/len(cnn_curve["top1"])
       
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
