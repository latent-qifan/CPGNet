import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.CPGNet import CPGNet
from utils.dataloader import test_dataset

EPS = 1e-8
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsize", type=int, default=704, help="testing size")
    parser.add_argument("--pth_path", type=str, default=os.path.join(WORKSPACE_ROOT, "Net_epoch_best.pth"))
    parser.add_argument("--data_path", type=str, default=os.path.join(PROJECT_ROOT, "datasets/PCOD/test"))
    parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, "results/PCOD"))
    parser.add_argument("--strict_check", action="store_true", help="raise error if any mismatch/missing")
    return parser.parse_args()


def main():
    opt = parse_args()
    os.makedirs(opt.save_path, exist_ok=True)

    # ---- model ----
    model = CPGNet()
    state = torch.load(opt.pth_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.cuda()
    model.eval()

    # ---- roots ----
    image_root = os.path.join(opt.data_path, "rgb/")
    aop_root = os.path.join(opt.data_path, "test-aop/")
    dop_root = os.path.join(opt.data_path, "test-dop/")
    gt_root = os.path.join(opt.data_path, "gt/")

    print("[TEST ROOT]")
    print("  image_root:", image_root)
    print("  aop_root  :", aop_root)
    print("  dop_root  :", dop_root)
    print("  gt_root   :", gt_root)

    # ---- loader ----
    test_loader = test_dataset(
        image_root=image_root,
        aop_root=aop_root,
        dop_root=dop_root,
        gt_root=gt_root,
        testsize=opt.testsize,
        strict_check=opt.strict_check,  # 默认 False；你想严格就加 --strict_check
    )

    print("Test samples:", test_loader.size)

    t1 = time.perf_counter()

    with torch.no_grad():
        for _ in range(test_loader.size):
            image, aop, dop, gt, name = test_loader.load_data()

            # gt 用来确定原始尺寸（你的代码就是这么干的）
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + EPS)

            image = image.cuda(non_blocking=True)
            aop = aop.cuda(non_blocking=True)
            dop = dop.cuda(non_blocking=True)

            # forward: CPGNet(image, aop, dop)
            _, p2 = model(image, aop, dop)

            res = F.interpolate(p2[-1], size=gt.shape, mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + EPS)

            out_path = os.path.join(opt.save_path, name)
            cv2.imwrite(out_path, (res * 255).astype(np.uint8))

    t2 = time.perf_counter()
    print(f"Finish! Average Time Is {((t2 - t1) * 1000) / max(test_loader.size, 1):.2f} ms")


if __name__ == "__main__":
    main()
