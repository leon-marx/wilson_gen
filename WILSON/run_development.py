import argparser
import os
from utils.logger import DummyLogger
from torch.utils.data.distributed import DistributedSampler
from sampler import InterleaveSampler

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from dataset import get_dataset
from metrics import StreamSegMetrics
from train import Trainer
import matplotlib.pyplot as plt


def save_ckpt(path, trainer, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "scheduler_state": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "best_score": best_score,
    }
    if trainer.pseudolabeler is not None:
        state["pseudolabeler"] = trainer.pseudolabeler.state_dict()

    # torch.save(state, path)
    print("The model would have been saved here, uncomment the line in save_ckpt() to enable")


def main(opts):
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = int(os.environ.get("LOCAL_RANK")), torch.device(int(os.environ.get("LOCAL_RANK")))
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)
    opts.device_id = device_id

    # Initialize logging
    task_name = f"{opts.dataset}-{opts.task}"
    if opts.overlap and opts.dataset == 'voc':
        task_name += "-ov"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    logger = DummyLogger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step,
                         name=f"{task_name}_{opts.name}")

    ckpt_path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step}.pth"

    if not os.path.exists(f"checkpoints/step/{task_name}"):
        os.makedirs(f"checkpoints/step/{task_name}")

    logger.print(f"Device: {device}")

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    opts.batch_size = opts.batch_size // world_size
    train_dst, val_dst, test_dst, labels, n_classes = get_dataset(opts)
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    if opts.replay:
        train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                    sampler=InterleaveSampler(train_dst, batch_size=opts.batch_size),
                                    num_workers=opts.num_workers, drop_last=True)
    else:
        train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                    sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                    num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1, shuffle=False,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {n_classes}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")
    opts.max_iters = opts.epochs * len(train_loader)
    if opts.lr_policy == "warmup":
        opts.start_decay = opts.pseudo_ep * len(train_loader)

    # xxx Set up Trainer
    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(logger, device=device, opts=opts)

    # xxx Load old model from old weights if step > 0!
    if opts.step > 0:
        # get model path
        if opts.step_ckpt is not None:
            path = opts.step_ckpt
        else:
            path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}.pth"
        trainer.load_step_ckpt(path)

    # if opts.step > 0 and opts.weakly:
    #     if opts.pl_ckpt is not None:
    #         pl_path = opts.pl_ckpt
    #     else:
    #         pl_path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step}w.pth"
    #     trainer.load_pseudolabeler(pl_path)

    # Load training checkpoint if any
    if opts.continue_ckpt:
        opts.ckpt = ckpt_path
    if opts.ckpt is not None:
        cur_epoch, best_score = trainer.load_ckpt(opts.ckpt)
    else:
        logger.info("[!] Start from epoch 0")
        cur_epoch = 0
        best_score = 0.

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_config(opts)

    TRAIN = not opts.test
    if opts.step == 0:
        val_metrics = StreamSegMetrics(n_classes)
    else:
        val_metrics = StreamSegMetrics(n_classes, n_old_classes=trainer.old_classes)
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    # train/val here
    while cur_epoch < opts.epochs and TRAIN:
        # =====  Train  =====
        if cur_epoch == opts.inpainting_epoch:
            trainer.inpaint_onehots()
            opts.mask_pre_inp = False
            trainer.opts.mask_pre_inp = False
            train_dst.update_pseudolabels()
            epoch_loss = trainer.train(cur_epoch=cur_epoch, train_loader=train_loader)

        else:
            epoch_loss = trainer.train(cur_epoch=cur_epoch, train_loader=train_loader)

        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0] + epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("Train/Tot", epoch_loss[0] + epoch_loss[1], cur_epoch)
        logger.add_scalar("Train/Reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("Train/Cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            val_score = trainer.validate(loader=val_loader, metrics=val_metrics)

            logger.print("Done validation Model")

            logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            # this is not the best, but the last model.
            # val set == test set, saving best model not legal
            if rank == 0:  # save best model at the last iteration
                score = val_score['Mean IoU']
                if opts.ckpt_interval > 0:
                    if (cur_epoch + 1) % opts.ckpt_interval == 0:
                        # best model to build incremental steps
                        save_ckpt(f"{ckpt_path[:-4]}_epoch_{cur_epoch + 1}.pth", trainer, cur_epoch, score)
                        logger.info("[!] Checkpoint saved.")

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("Val/Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val/MeanAcc", val_score['Agg'][1], cur_epoch)
            logger.add_scalar("Val/MeanPrec", val_score['Agg'][2], cur_epoch)
            logger.add_scalar("Val/MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_scalar("Val/MeanIoU_Bkg", val_score['Mean IoU Bkg'], cur_epoch)
            logger.add_scalar("Val/MeanIoU_Old", val_score['Mean IoU Old'], cur_epoch)
            logger.add_scalar("Val/MeanIoU_New", val_score['Mean IoU New'], cur_epoch)
            logger.add_table("Val/Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_table("Val/Acc_IoU", val_score['Class Acc'], cur_epoch)
            logger.add_figure("Val/Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)
            plt.close(val_score["Confusion Matrix"])

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

            if opts.weakly:
                val_score_cam = trainer.validate_CAM(loader=val_loader, metrics=val_metrics)
                logger.add_scalar("Val_CAM/MeanAcc", val_score_cam['Agg'][1], cur_epoch)
                logger.add_scalar("Val_CAM/MeanPrec", val_score_cam['Agg'][2], cur_epoch)
                logger.add_scalar("Val_CAM/MeanIoU", val_score_cam['Mean IoU'], cur_epoch)
                logger.add_scalar("Val_CAM/MeanIoU_Bkg", val_score_cam['Mean IoU Bkg'], cur_epoch)
                logger.add_scalar("Val_CAM/MeanIoU_Old", val_score_cam['Mean IoU Old'], cur_epoch)
                logger.add_scalar("Val_CAM/MeanIoU_New", val_score_cam['Mean IoU New'], cur_epoch)
                logger.info(val_metrics.to_str(val_score_cam))
                plt.close(val_score_cam["Confusion Matrix"])
                logger.print("Done validation CAM")

        logger.commit()
        logger.info(f"End of Validation {cur_epoch}/{opts.epochs}")

        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(ckpt_path, trainer, cur_epoch, best_score)
        logger.info("[!] Checkpoint saved.")

    torch.distributed.barrier()

    if opts.weakly and opts.test:
        val_loader = data.DataLoader(val_dst, batch_size=1, shuffle=False,
                                     sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                     num_workers=opts.num_workers)
        val_score_cam = trainer.validate_CAM(loader=val_loader, metrics=val_metrics, multi_scale=True)
        logger.add_scalar("Val_CAM/MeanAcc", val_score_cam['Agg'][1], cur_epoch)
        logger.add_scalar("Val_CAM/MeanPrec", val_score_cam['Agg'][2], cur_epoch)
        logger.add_scalar("Val_CAM/MeanIoU", val_score_cam['Mean IoU'], cur_epoch)
        logger.add_scalar("Val_CAM/MeanIoU_Bkg", val_score_cam['Mean IoU Bkg'], cur_epoch)
        logger.add_scalar("Val_CAM/MeanIoU_Old", val_score_cam['Mean IoU Old'], cur_epoch)
        logger.add_scalar("Val_CAM/MeanIoU_New", val_score_cam['Mean IoU New'], cur_epoch)
        logger.info(val_metrics.to_str(val_score_cam))
        plt.close(val_score_cam["Confusion Matrix"])
        logger.print("Done validation CAM")

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    val_score = trainer.validate(loader=test_loader, metrics=val_metrics)
    logger.info(f"*** End of Test")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test/Class_IoU", val_score['Class IoU'])
    logger.add_table("Test/Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test/Confusion_Matrix", val_score['Confusion Matrix'])
    plt.close(val_score["Confusion Matrix"])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    # logger.add_results(results)

    logger.add_scalar("Test/Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("Test/MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("Test/MeanIoU_Bkg", val_score['Mean IoU Bkg'], opts.step)
    logger.add_scalar("Test/MeanIoU_Old", val_score['Mean IoU Old'], opts.step)
    logger.add_scalar("Test/MeanIoU_New", val_score['Mean IoU New'], opts.step)
    logger.add_scalar("Test/MeanAcc", val_score['Mean Acc'], opts.step)
    logger.commit()

    logger.log_results(task=task_name, name=opts.name, results=val_score['Class IoU'].values())
    logger.log_aggregates(task=task_name, name=opts.name, results=val_score['Agg'])
    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()

    opts = parser.parse_args()

    os.makedirs("checkpoints/step", exist_ok=True)

    #region Debugging Parameters
    os.chdir("WILSON")
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    import get_free_port
    start_port = random.choice(list(range(1994, 2994)))
    port = get_free_port.next_free_port(port=start_port)
    os.environ["MASTER_PORT"] = str(port)
    opts.num_workers = 4
    opts.sample_num = 8
    opts.dataset = "voc"
    opts.lr = 0.001
    opts.batch_size = 24
    opts.val_interval = 2
    opts.random_seed = 7
    ########## overall settings:
    opts.name = "DEBUGGING"
    opts.overlap = False
    opts.task = "10-5"
    ########## base step:
    opts.step = 0
    opts.epochs = 30
    opts.lr_init=0.01
    opts.bce = True
    ########## incr step:
    # opts.step = 1
    # opts.epochs = 40
    # opts.step_ckpt = "checkpoints/step/voc-10-10/Base_0.pth"
    # opts.weakly = True
    # opts.alpha = 0.5
    # opts.loss_de = 1
    # opts.lr_policy = "warmup"
    # opts.affinity = True
    ########## replay settings:
    # opts.replay_root = "replay_data_mrte"
    # opts.replay = True
    # opts.replay_ratio = None
    # opts.replay_size = 1
    #endregion Debugging Parameters

    opts = argparser.modify_command_options(opts)

    main(opts)
