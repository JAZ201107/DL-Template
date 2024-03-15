import os
from tqdm import tqdm
import logging
from pickle import dump
import time

import torch
import numpy as np

from utils.misc import (
    AverageMeter,
    load_checkpoint,
    save_checkpoint,
    save_dict_to_json,
    save_learning_curve,
)
import config


def train_one_epoch(
    model, dataloader, criterion, optimizer, metrics, params, lr_scheduler=None
):
    model.train()

    summ = ()
    loss_avg = AverageMeter()

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, train_labels) in enumerate(dataloader):
            if params.cuda:
                train_batch, train_labels = train_batch.cuda(
                    non_blocking=True
                ), train_labels.cuda(non_blocking=True)

            output_batch = model(train_batch)
            loss = criterion(output_batch, train_labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if i % config.SAVE_SUMMARY_STEPS == 0:
                output_batch = output_batch.detach().cpu().numpy()
                labels_batch = train_labels.detach().cpu().numpy()

                summary_batch = {
                    metric: metrics[metric](output_batch, labels_batch)
                    for metric in metrics
                }

                summary_batch["loss"] = loss.item()
                summ.append(summary_batch)

            loss_avg.update(loss.item())
            t.set_postfix(loss="{:05.3f}".format(loss_avg()))
            t.update()

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " : ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean


def evaluate(model, dataloader, criterion, metrics, params):
    model.eval()
    summ = []

    loss_avg = AverageMeter()

    with tqdm(total=len(dataloader)) as t:
        for data_batch, labels_batch in dataloader:
            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(
                    non_blocking=True
                ), labels_batch.cuda(non_blocking=True)

            output_batch = model(data_batch)
            loss = criterion(output_batch, labels_batch)

            output_batch = output_batch.detach().cpu().numpy()
            labels_batch = labels_batch.detach().cpu().numpy()

            summary_batch = {
                metric: metrics[metric](output_batch, labels_batch)
                for metric in metrics
            }

            summary_batch["loss"] = loss.item()
            summ.append(summary_batch)

            loss_avg.update(loss.item())

            t.set_postfix(loss="{:05.3f}".format(loss_avg()))
            t.update()

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    # metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join(
        "{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items()
    )

    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def train_and_evaluate(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    metrics,
    params,
    model_dir=None,
    restore_file=None,
    lr_scheduler=None,
):
    if restore_file is not None:
        logging.info(f"Restoring parameters from {restore_file}")
        load_checkpoint(restore_file, model, optimizer)

    best_val_acc = None  # some metric to get the best model
    summ = {"train": {}, "valid": {}}
    start_time = time()

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch + 1} / {params.num_epochs}")
        train_summ = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_dataloader,
            metrics=metrics,
            params=params,
        )

        val_summ = evaluate(
            model=model,
            criterion=criterion,
            dataloader=val_dataloader,
            metrics=metrics,
            params=params,
        )

        val_acc = val_summ["some_metrics"]
        is_best = val_acc >= best_val_acc

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # save best val metrics in a json file
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")

            save_dict_to_json(val_summ, best_json_path)

        save_checkpoint(
            state={
                "epoch": epoch + 1,
                "model_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
            },
            is_best=is_best,
            checkpoint=model_dir,
        )
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_summ, last_json_path)

        summ["train"][epoch + 1] = train_summ
        summ["valid"][epoch + 1] = val_summ

    with open(os.path.join(model_dir, config.TRAIN_VAL_METRICS_SUMM)) as f:
        dump(summ, f)
    save_learning_curve(summ, model_dir)

    logging.info("- total time taken: %.2fs" % (time() - start_time))
