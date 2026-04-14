# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os

import cv2
import sod_metrics as M
import torch
import torch.nn.functional as F
from tqdm import tqdm

FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()

MASK_ROOT = "./datasets//test/gt"
PRED_ROOT = "./results/"

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
ALIGN_CORNERS = False


def upsample_like(src, tar_shape):
    """Resize src(H,W) to tar_shape(H,W) with bilinear. Output: np.ndarray float."""
    src_t = torch.from_numpy(src).float()
    out = F.interpolate(
        src_t.unsqueeze(0).unsqueeze(0),
        size=tar_shape,
        mode="bilinear",
        align_corners=ALIGN_CORNERS,
    )
    return out.squeeze(0).squeeze(0).numpy()


def main():
    if not os.path.isdir(MASK_ROOT):
        raise FileNotFoundError(f"MASK_ROOT not found: {MASK_ROOT}")
    if not os.path.isdir(PRED_ROOT):
        raise FileNotFoundError(f"PRED_ROOT not found: {PRED_ROOT}")

    mask_names = sorted([n for n in os.listdir(MASK_ROOT) if n.lower().endswith(VALID_EXTS)])
    pred_names = set([n for n in os.listdir(PRED_ROOT) if n.lower().endswith(VALID_EXTS)])


    missing_preds = [n for n in mask_names if n not in pred_names]
    if missing_preds:
 
        raise FileNotFoundError(f"Missing pred files (show 20): {missing_preds[:20]} (total={len(missing_preds)})")

    for mask_name in tqdm(mask_names, total=len(mask_names)):
        mask_path = os.path.join(MASK_ROOT, mask_name)
        pred_path = os.path.join(PRED_ROOT, mask_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise FileNotFoundError(f"Failed to read pred: {pred_path}")

     
        if pred.ndim != 2:
            pred = pred[:, :, 0]
        if mask.ndim != 2:
            mask = mask[:, :, 0]

        pred = upsample_like(pred, tar_shape=mask.shape)
        assert pred.shape == mask.shape, f"Shape mismatch: pred={pred.shape}, mask={mask.shape}, name={mask_name}"

        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]

    print(
        "Smeasure:", sm.round(3), "; ",
        "wFmeasure:", wfm.round(3), "; ",
        "MAE:", mae.round(3), "; ",
        "adpEm:", em["adp"].round(3), "; ",
        "meanEm:", "-" if em["curve"] is None else em["curve"].mean().round(3), "; ",
        "maxEm:", "-" if em["curve"] is None else em["curve"].max().round(3), "; ",
        "adpFm:", fm["adp"].round(3), "; ",
        "meanFm:", fm["curve"].mean().round(3), "; ",
        "maxFm:", fm["curve"].max().round(3),
        sep="",
    )

 
    out_path = "result.txt"
    with open(out_path, "a+", encoding="utf-8") as f:
        print(
            "Smeasure:", sm.round(3), "; ",
            "meanEm:", "-" if em["curve"] is None else em["curve"].mean().round(3), "; ",
            "wFmeasure:", wfm.round(3), "; ",
            "MAE:", mae.round(3), "; ",
            file=f,
        )


if __name__ == "__main__":
    main()
