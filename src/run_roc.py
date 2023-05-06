from collect import FairExecutor
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchattacks
from typing import *
from datetime import datetime
from tqdm import tqdm, trange
import sklearn
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

from utils import *
from models import MyFNNetClf
from load_data import MyDataset, AdultDataset, CompasDataset


class ROCExecutor(FairExecutor):
    def __init__(self, dataset: MyDataset, sensitive_name: str, save_path: str,
                 device_type: str = 'gpu', repeat_num: int = 10, width: int = 2, ratio: float = 0.2) -> None:
        super().__init__(dataset=dataset,
                         sensitive_name=sensitive_name,
                         save_path=save_path,
                         device_type=device_type,
                         repeat_num=repeat_num)
        self.name: str = "ROC"
        self.model_width_option = [width]
        self.ratio_list = [ratio]

    def run(self,) -> None:
        proc_args = {
            "privileged_groups": [{self.sensitive_name: 1}],
            "unprivileged_groups": [{self.sensitive_name: 0}],
            "low_class_thresh": 0.01,
            "high_class_thresh": 0.99,
            "num_class_thresh": 100,
            "num_ROC_margin": 50,
            "metric_name": "Average odds difference",
            "metric_ub": 0.05,
            "metric_lb": -0.05
        }
        self.post_proc_run(fair_proc_cls=RejectOptionClassification, fair_proc_args=proc_args)


@ time_statistic
def main(args):
    if args.dataset == "adult":
        obj_dataset = AdultDataset("./data/adult/", if_discretize=False, if_drop_lot=False)
    elif args.dataset == "compas":
        obj_dataset = CompasDataset("./data/compas/", if_discretize=False, if_drop_lot=False)
    elif args.dataset == "german":
        pass
    elif args.dataset == "crime":
        pass
    else:
        raise ValueError("Dataset not supported!")

    executer = ROCExecutor(
        dataset=obj_dataset, sensitive_name=args.sensitive_name, save_path=args.save_path,
        repeat_num=args.repeat_num, width=args.width, ratio=args.ratio)
    executer.run()
    # executer.save_to_csv()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str,
                        choices=["adult", "compas", "german", "crime"],
                        default="adult")
    parser.add_argument("--sensitive_name", "-sn", type=str,
                        choices=["sex", "race", "age"], default="sex")
    parser.add_argument("--save_path", "-sp", type=str,
                        default="")
    parser.add_argument("--repeat_num", "-rn", type=int,
                        default=10)
    parser.add_argument("--width", "-w", type=int,
                        default=2)
    parser.add_argument("--ratio", "-r", type=float,
                        default=1.0)

    # parser.add_argument("--if_exec", "-e", type=bool,~
    #                     default=False, action="store_true")
    # parser.add_argument("--ratio", "-r", type=float,
    #                     default=0.0)
    args = parser.parse_args()
    print(args)
    main(args)
    print_time_statistic()
