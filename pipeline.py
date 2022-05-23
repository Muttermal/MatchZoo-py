# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 10:20
# @Author  : zhangguangyi
# @File    : pipeline.py

import argparse
import logging
import os
from transformers import get_linear_schedule_with_warmup
import torch
from torch.nn.utils import clip_grad_norm_
from typing import Callable
import yaml

import matchzoo as mz
from matchzoo.utils.dynamic_import import import_class
logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)
cuda = "cuda" if torch.cuda.is_available() else "cpu"


def instantiate(func: Callable, params: dict):
    """函数实例化"""
    if params:
        try:
            return func(**params)
        except TypeError:
            return func()
    else:
        return func()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pipeline")
    parser.add_argument("--config_file", type=str, default="configs/test.yaml", help="config file")
    args = parser.parse_args()
    paras = yaml.load(open(args.config_file, "r", encoding="utf-8"))

    # *****************************确定任务及模型********************************
    task_name = paras["task"]["name"]
    task = getattr(mz.tasks, task_name)
    task = instantiate(task, paras["task"].get("params"))
    task.metrics, task.losses = [], []
    for metric in paras["metrics"]:
        used_metric = import_class(metric["name"], ".metrics", "matchzoo")
        task.metrics.append(instantiate(used_metric, metric.get("params")))
    for loss in paras["losses"]:
        used_loss = import_class(loss["name"], ".losses", "matchzoo")
        task.losses.append(instantiate(used_loss, loss.get("params")))
    logger.info("`{}` initialized with metrics {}".format(task_name, task.metrics))
    logger.info("`{}` initialized with losses {}".format(task_name, task.losses))
    model = import_class(paras["model"]["name"], ".models", "matchzoo")
    # 支持导入本地训练好的模型
    try:
        model_path = os.path.join(paras["trainer"]["train"].get("save_dir"), "pytorch_model.pth")
        model = torch.load(model_path, map_location='cpu')
        logger.info(f"load model from path:{model_path}")
    except FileNotFoundError:
        logger.info("load pre-training model")

    if paras["phase"] == "train":
        pass
    # *****************************准备数据********************************
    logger.info('data loading ...')
    train_pack_raw = mz.datasets.load_data.read_data(data_root=paras["dataset"]["train"]["data_root"], stage='train',
                                                     task=task)
    dev_pack_raw = mz.datasets.load_data.read_data(data_root=paras["dataset"]["dev"]["data_root"], stage='dev',
                                                   task=task)
    test_pack_raw = mz.datasets.load_data.read_data(data_root=paras["dataset"]["test"]["data_root"], stage='test',
                                                    task=task)
    logger.info('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')
    preprocessor = getattr(mz.preprocessors, paras["preprocessor"]["name"])
    preprocessor = instantiate(preprocessor, paras["preprocessor"].get("params"))
    train_pack_processed = preprocessor.transform(train_pack_raw)
    dev_pack_processed = preprocessor.transform(dev_pack_raw)
    test_pack_processed = preprocessor.transform(test_pack_raw)

    trainset = mz.dataloader.Dataset(
        data_pack=train_pack_processed,
        **paras["dataset"]["train"]
    )
    testset = mz.dataloader.Dataset(
        data_pack=test_pack_processed,
        **paras["dataset"]["test"]
    )
    used_callback = getattr(mz.dataloader.callbacks, paras["callback"]["name"])
    used_callback = instantiate(used_callback, paras["callback"].get("params"))
    trainloader = mz.dataloader.DataLoader(
        dataset=trainset,
        callback=used_callback,
        device=paras["service"]["gpu_id"],
        **paras["dataset"]["train"]
    )
    testloader = mz.dataloader.DataLoader(
        dataset=testset,
        callback=used_callback,
        device=paras["service"]["gpu_id"],
        **paras["dataset"]["test"]
    )

    # *****************************初始化模型********************************
    model = instantiate(model, paras["model"].get("params"))
    model.params["task"] = task
    for k, v in paras["model"].get("params").items():
        model.params[k] = v
    model.build()
    logger.info("model type {}".format(paras["model"]["params"]["mode"]))
    logger.info('Trainable params: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # *****************************开始训练********************************
    no_decay = paras["optimizer"].get("no_decay")
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 5e-5},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = import_class(paras["optimizer"]["name"], ".optim", "torch")
    optimizer = optimizer(
        params=optimizer_grouped_parameters,
        **paras["optimizer"]["params"]
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=6, num_training_steps=-1)

    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=trainloader,
        validloader=testloader,
        validate_interval=None,
        device=paras["service"]["gpu_id"],
        **paras["trainer"]["train"]
    )
    trainer.run()

    # *****************************预测结果********************************
