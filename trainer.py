import sys
import logging
import copy
import torch
import glob
import re
import ast
from utils import model_factory
from data.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np


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

def get_task_id_from_range(class_start, init_cls, increment):
    if class_start == 0:
        return 0

    return 1 + ((class_start - init_cls) // increment)

def restore_curve_from_log(log_path, completed_task, init_cls, increment):
    expected_len = completed_task + 1
    top1_by_task = {}
    top5_by_task = {}
    current_task = None

    if not os.path.isfile(log_path):
        return {"top1": [], "top5": []}

    learning_pattern = re.compile(r"Learning on\s+(\d+)-(\d+)")
    cnn_pattern = re.compile(r"CNN:\s*(\{.*\})")
    top5_pattern = re.compile(r"CNN top5 curve:\s*(\[.*\])")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
        for line in log_file:
            learning_match = learning_pattern.search(line)
            if learning_match:
                class_start = int(learning_match.group(1))
                current_task = get_task_id_from_range(class_start, init_cls, increment)
                continue

            if current_task is None or current_task > completed_task:
                continue

            cnn_match = cnn_pattern.search(line)
            if cnn_match:
                cnn_accy = ast.literal_eval(cnn_match.group(1))
                top1_by_task[current_task] = cnn_accy["total"]
                continue

            top5_match = top5_pattern.search(line)
            if top5_match:
                curve = ast.literal_eval(top5_match.group(1))
                if curve:
                    top5_by_task[current_task] = curve[-1]

    if not all(task in top1_by_task for task in range(expected_len)):
        top1_curve = []
    else:
        top1_curve = [top1_by_task[task] for task in range(expected_len)]

    if not all(task in top5_by_task for task in range(expected_len)):
        top5_curve = []
    else:
        top5_curve = [top5_by_task[task] for task in range(expected_len)]

    return {"top1": top1_curve, "top5": top5_curve}

def complete_curve_from_log(cnn_curve, log_path, completed_task, init_cls, increment):
    expected_len = completed_task + 1
    if len(cnn_curve["top1"]) >= expected_len and len(cnn_curve["top5"]) >= expected_len:
        return cnn_curve

    restored_curve = restore_curve_from_log(
        log_path,
        completed_task,
        init_cls,
        increment,
    )

    if len(restored_curve["top1"]) == expected_len:
        cnn_curve["top1"] = restored_curve["top1"]

    if len(restored_curve["top5"]) == expected_len:
        cnn_curve["top5"] = restored_curve["top5"]

    if len(cnn_curve["top1"]) >= expected_len and len(cnn_curve["top5"]) >= expected_len:
        logging.info("Restored CNN curve from log: top1=%s", cnn_curve["top1"])
        logging.info("Restored CNN top5 curve from log: top5=%s", cnn_curve["top5"])

    return cnn_curve

def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    output_root = args.get("output_root", "")
    logs_root = os.path.join(output_root, "logs")
    logs_name = os.path.join(
        logs_root,
        str(args["model_name"]),
        str(args["dataset"]),
        str(init_cls),
        str(args["increment"]),
    )
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = os.path.join(
        logs_name,
        "{}_{}_{}".format(
            args["prefix"],
            args["seed"],
            args["convnet_type"],
        ),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
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
    model.class_order = list(data_manager._class_order)

    checkpoint_dir = os.path.join(
        output_root,
        "ckpt",
        str(args["prefix"]),
        str(args["dataset"]),
        "{}_{}".format(args["init_cls"], args["increment"]),
    )
    if args.get("isolate_runs", args.get("model_name") == "umt_adapter"):
        checkpoint_dir = os.path.join(checkpoint_dir, "seed_{}".format(args["seed"]))

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
            model.cnn_curve = complete_curve_from_log(
                getattr(model, "cnn_curve", {"top1": [], "top5": []}),
                logfilename + ".log",
                completed_task,
                args["init_cls"],
                args["increment"],
            )

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

    end_task = data_manager.nb_tasks
    max_tasks_per_run = args.get("max_tasks_per_run")
    if max_tasks_per_run is not None:
        max_tasks_per_run = max(1, int(max_tasks_per_run))
        end_task = min(end_task, start_task + max_tasks_per_run)
        logging.info(
            "This run will process tasks [%d, %d) out of %d tasks.",
            start_task,
            end_task,
            data_manager.nb_tasks,
        )

    for task in range(start_task, end_task):
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

        if args.get("keep_last_checkpoint", False) and task > 0:
            previous_checkpoint = os.path.join(
                checkpoint_dir,
                "task_{}.pkl".format(task - 1),
            )
            if os.path.isfile(previous_checkpoint):
                os.remove(previous_checkpoint)
                logging.info("Removed superseded checkpoint: %s", previous_checkpoint)


        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    if end_task < data_manager.nb_tasks:
        logging.info(
            "Stopped cleanly after %d task(s) because max_tasks_per_run=%d. "
            "Resume the same seed to continue at task %d.",
            end_task - start_task,
            max_tasks_per_run,
            end_task,
        )
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


def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
