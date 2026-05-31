import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.CPGNet import CPGNet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, AvgMeter

EPS = 1e-8
LOG_INTERVAL = 20
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_matched_state_dict(model, state_dict, print_stats=True):
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f"Loaded state_dict: {num_matched}/{num_total} matched")


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def val(model, epoch, save_path, writer, opt):
    global best_mae, best_epoch

    model.eval()
    with torch.no_grad():
        mae_sum = 0.0

        test_loader = test_dataset(
            image_root=os.path.join(opt.test_path, "rgb/"),
            aop_root=os.path.join(opt.test_path, "test-aop/"),
            dop_root=os.path.join(opt.test_path, "test-dop/"),
            gt_root=os.path.join(opt.test_path, "gt/"),
            testsize=opt.trainsize,
            strict_check=True,
            show_pairs=1 if epoch == 1 else 0,
            pol_as_gray=opt.pol_as_gray,
        )

        for _ in range(test_loader.size):
            image, aop, dop, gt, _name = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + EPS)

            image = image.cuda(non_blocking=True)
            aop = aop.cuda(non_blocking=True)
            dop = dop.cuda(non_blocking=True)

            P1, P2 = model(image, aop, dop)

            res = F.interpolate(P2[-1], size=gt.shape, mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + EPS)
            mae_sum += np.sum(np.abs(res - gt)) / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar("MAE", mae, global_step=epoch)

        print(f"Epoch: {epoch}, MAE: {mae:.6f}, bestMAE: {best_mae:.6f}, bestEpoch: {best_epoch}.")
        if epoch == 1:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, "Net_epoch_best.pth"))
            print(f"Save initial best state_dict successfully! Best epoch: {epoch}.")
        elif mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, "Net_epoch_best.pth"))
            print(f"Save state_dict successfully! Best epoch: {epoch}.")

        logging.info(
            f"[Val Info]: Epoch:{epoch} MAE:{mae:.6f} bestEpoch:{best_epoch} bestMAE:{best_mae:.6f}"
        )

    return mae


def train(train_loader, model, optimizer, epoch, opt):
    model.train()

    size_rates = [1]
    loss_p1_meter = AvgMeter()
    total_step = len(train_loader)

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            images, aop, dop, gts = pack
            images = images.cuda(non_blocking=True)
            aop = aop.cuda(non_blocking=True)
            dop = dop.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                aop = F.interpolate(aop, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                dop = F.interpolate(dop, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode="bilinear", align_corners=True)

            P1, P2 = model(images, aop, dop)

            losses1 = [structure_loss(out, gts) for out in P1]
            loss_p1 = 0.0
            gamma = 0.2
            for it in range(len(P1)):
                loss_p1 += (gamma * it) * losses1[it]

            losses2 = [structure_loss(out, gts) for out in P2]
            loss_p2 = 0.0
            gamma2 = 0.2
            for it in range(len(P2)):
                loss_p2 += (gamma2 * it) * losses2[it]

            loss = loss_p1 + loss_p2
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_p1_meter.update(loss_p1.data, opt.batchsize)

        if i % LOG_INTERVAL == 0 or i == total_step:
            current_lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], "
                f"Step [{i:04d}/{total_step:04d}], "
                f"lateral-5: {loss_p1_meter.show():0.4f}, "
                f"lr: {current_lr:.8e}"
            )
            print(msg)
            logging.info(msg)

    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        if epoch % 3 == 1:
            torch.save(model.state_dict(), os.path.join(save_path, "CPGNet1.pth"))
        elif epoch % 3 == 2:
            torch.save(model.state_dict(), os.path.join(save_path, "CPGNet2.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, "CPGNet3.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training setup
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=6.666e-5)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--trainsize", type=int, default=704)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--load", type=str, default=None)

    # path
    parser.add_argument("--train_path", type=str, default=os.path.join(PROJECT_ROOT, "datasets/PCOD/train"))
    parser.add_argument("--test_path", type=str, default=os.path.join(PROJECT_ROOT, "datasets/PCOD/test"))
    parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, "ckpt"))
    parser.add_argument("--epoch_save", type=int, default=1)

    # data format
    parser.add_argument("--pol_as_gray", action="store_true", help="aop/dop load as grayscale (L)")

    # lr scheduler for 100 epochs
    parser.add_argument("--lr_decay_factor", type=float, default=0.5, help="new_lr = old_lr * factor")
    parser.add_argument(
        "--lr_decay_patience",
        type=int,
        default=4,
        help="epochs with no val improvement before reducing lr",
    )
    parser.add_argument("--min_lr", type=float, default=1e-6, help="minimum learning rate")

    opt = parser.parse_args()
    os.makedirs(opt.save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(opt.save_path, "log.log"),
        format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
        level=logging.INFO,
        filemode="a",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    )
    logging.info("Network-Train")
    logging.info(str(opt))

    model = CPGNet().cuda()

    if opt.load is not None:
        pretrained_dict = torch.load(opt.load, map_location="cpu")
        print("!!!!!! Successfully load model from !!!!!!", opt.load)
        load_matched_state_dict(model, pretrained_dict)

    print("model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    params = model.parameters()
    if opt.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, lr=opt.lr, weight_decay=1e-4, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=opt.lr_decay_factor,
        patience=opt.lr_decay_patience,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=opt.min_lr,
    )

    image_root = os.path.join(opt.train_path, "rgb/")
    aop_root = os.path.join(opt.train_path, "train-aop/")
    dop_root = os.path.join(opt.train_path, "train-dop/")
    gt_root = os.path.join(opt.train_path, "gt/")

    train_loader = get_loader(
        image_root=image_root,
        aop_root=aop_root,
        dop_root=dop_root,
        gt_root=gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        augmentation=opt.augmentation,
        strict_check=True,
        show_pairs=5,
        pol_as_gray=opt.pol_as_gray,
    )

    writer = SummaryWriter(os.path.join(opt.save_path, "summary"))

    print("#" * 20, "Start Training", "#" * 20)
    best_mae = 1.0
    best_epoch = 0
    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, opt)

        if epoch % opt.epoch_save == 0:
            mae = val(model, epoch, opt.save_path, writer, opt)
            scheduler.step(mae)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, global_step=epoch)

        print(f"Current LR: {current_lr:.8e}")
        logging.info(f"Current LR: {current_lr:.8e}")

        if current_lr < prev_lr:
            print(f"LR reduced from {prev_lr:.8e} to {current_lr:.8e}")
            logging.info(f"LR reduced from {prev_lr:.8e} to {current_lr:.8e}")

        prev_lr = current_lr

    writer.close()
