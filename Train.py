import torch
import os
import argparse
import logging
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from backbone.PLCamo import MyNet
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

def load_partial_weights(model, state_dict, verbose=True):
    own_state = model.state_dict()
    matched = total = 0

    for key in own_state:
        total += 1
        if key in state_dict and own_state[key].shape == state_dict[key].shape:
            own_state[key].copy_(state_dict[key])
            matched += 1

    model.load_state_dict(own_state)
    if verbose:
        print(f"Loaded weights: {matched}/{total} layers matched")

def weighted_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = (pred * mask * weight).sum(dim=(2, 3))
    union = (pred + mask * weight).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)

    return (bce + iou).mean()

def compute_layered_loss(outputs, target, gamma=0.2):
    losses = [weighted_loss(out, target) for out in outputs]
    total = 0
    for idx, loss in enumerate(losses):
        total += gamma * idx * loss
    return total

def validate(model, epoch, save_path, writer, best_mae, best_epoch, cfg):
    model.eval()
    mae_sum = 0.0

    test_loader = test_dataset(
        image_root=os.path.join(cfg.test_path, 'rgb'),
        polarization_root=os.path.join(cfg.test_path, 'polarization'),
        gt_root=os.path.join(cfg.test_path, 'gt'),
        testsize=cfg.trainsize
    )

    with torch.no_grad():
        for _ in range(test_loader.size):
            img, polar_img, gt, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= gt.max() + 1e-8

            img = img.cuda()
            polar_img = polar_img.cuda()

            _, out = model(img, polar_img)
            pred = F.interpolate(out[-1], size=gt.shape, mode='bilinear', align_corners=False)
            pred = pred.sigmoid().cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

            mae_sum += np.sum(np.abs(pred - gt)) / (gt.shape[0] * gt.shape[1])

    mae = mae_sum / test_loader.size
    writer.add_scalar("Val/MAE", mae, epoch)

    print(f"Epoch {epoch:3d} | MAE {mae:.4f} | Best MAE {best_mae:.4f} | Best Epoch {best_epoch}")
    logging.info(f"[Val] Epoch {epoch} | MAE {mae} | Best {best_mae}")

    if mae < best_mae:
        best_mae = mae
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
        print(f"-> Best model saved at epoch {epoch}")

    return best_mae, best_epoch

def train_epoch(model, loader, optimizer, epoch, total_steps, cfg, loss_recorder):
    model.train()
    loss_recorder.reset()

    for i, (imgs, polars, gts) in enumerate(loader, 1):
        optimizer.zero_grad()

        imgs = imgs.cuda()
        polars = polars.cuda()
        gts = gts.cuda()

        out1, out2 = model(imgs, polars)
        loss1 = compute_layered_loss(out1, gts, 0.2)
        loss2 = compute_layered_loss(out2, gts, 0.2)
        total_loss = loss1 + loss2

        total_loss.backward()
        clip_gradient(optimizer, cfg.clip)
        optimizer.step()

        loss_recorder.update(loss1.data, cfg.batchsize)

        if i % 20 == 0 or i == total_steps:
            msg = f"{datetime.now()} | Epoch {epoch:3d} | Step {i:4d}/{total_steps} | Loss {loss_recorder.show():.4f}"
            print(msg)
            logging.info(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--augmentation', default=False)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--trainsize', type=int, default=704)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=50)
    parser.add_argument('--train_path', type=str, default='./datasets/PlantCAMO1250/train')
    parser.add_argument('--test_path', type=str, default='./datasets/PlantCAMO1250/test')
    parser.add_argument('--save_path', type=str, default='./ckpt/')
    parser.add_argument('--epoch_save', type=int, default=1)
    cfg = parser.parse_args()

    os.makedirs(cfg.save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(cfg.save_path, 'train.log'),
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    model = MyNet().cuda()
    if cfg.load:
        print(f"Loading checkpoint: {cfg.load}")
        weights = torch.load(cfg.load)
        load_partial_weights(model, weights)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-4)

    train_loader = get_loader(
        image_root=os.path.join(cfg.train_path, 'rgb'),
        polarization_root=os.path.join(cfg.train_path, 'polarization'),
        gt_root=os.path.join(cfg.train_path, 'gt'),
        batchsize=cfg.batchsize,
        trainsize=cfg.trainsize,
        augmentation=cfg.augmentation
    )
    total_steps = len(train_loader)
    writer = SummaryWriter(os.path.join(cfg.save_path, 'summary'))
    loss_recorder = AvgMeter()

    best_mae = 1.0
    best_epoch = 0

    print("=" * 50)
    print("Training Started")
    print("=" * 50)

    for epoch in range(1, cfg.epoch + 1):
        adjust_lr(optimizer, cfg.lr, epoch, cfg.decay_rate, cfg.decay_epoch)
        train_epoch(model, train_loader, optimizer, epoch, total_steps, cfg, loss_recorder)

        if epoch % cfg.epoch_save == 0:
            torch.save(model.state_dict(), os.path.join(cfg.save_path, f"model_epoch_{epoch}.pth"))
            best_mae, best_epoch = validate(model, epoch, cfg.save_path, writer, best_mae, best_epoch, cfg)

if __name__ == "__main__":
    main()
