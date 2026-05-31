import argparse
import os

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from evaluation import sod_metrics as M
except ImportError:
    import sod_metrics as M

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
ALIGN_CORNERS = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", type=str, default=os.path.join(PROJECT_ROOT, "datasets/PCOD/test/gt"))
    parser.add_argument("--pred_root", type=str, default=os.path.join(PROJECT_ROOT, "results/PCOD"))
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "results/PCOD/result.txt"))
    return parser.parse_args()


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
    opt = parse_args()
    fm_metric = M.Fmeasure()
    wfm_metric = M.WeightedFmeasure()
    sm_metric = M.Smeasure()
    em_metric = M.Emeasure()
    mae_metric = M.MAE()

    if not os.path.isdir(opt.mask_root):
        raise FileNotFoundError(f"mask_root not found: {opt.mask_root}")
    if not os.path.isdir(opt.pred_root):
        raise FileNotFoundError(f"pred_root not found: {opt.pred_root}")

    mask_names = sorted([n for n in os.listdir(opt.mask_root) if n.lower().endswith(VALID_EXTS)])
    pred_names = set([n for n in os.listdir(opt.pred_root) if n.lower().endswith(VALID_EXTS)])

    # 只做检查，不改变评测逻辑
    missing_preds = [n for n in mask_names if n not in pred_names]
    if missing_preds:
        # 只展示前20个，避免刷屏
        raise FileNotFoundError(f"Missing pred files (show 20): {missing_preds[:20]} (total={len(missing_preds)})")

    for mask_name in tqdm(mask_names, total=len(mask_names)):
        mask_path = os.path.join(opt.mask_root, mask_name)
        pred_path = os.path.join(opt.pred_root, mask_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise FileNotFoundError(f"Failed to read pred: {pred_path}")

        # 兼容异常 shape（保留你原本行为）
        if pred.ndim != 2:
            pred = pred[:, :, 0]
        if mask.ndim != 2:
            mask = mask[:, :, 0]

        pred = upsample_like(pred, tar_shape=mask.shape)
        assert pred.shape == mask.shape, f"Shape mismatch: pred={pred.shape}, mask={mask.shape}, name={mask_name}"

        fm_metric.step(pred=pred, gt=mask)
        wfm_metric.step(pred=pred, gt=mask)
        sm_metric.step(pred=pred, gt=mask)
        em_metric.step(pred=pred, gt=mask)
        mae_metric.step(pred=pred, gt=mask)

    fm = fm_metric.get_results()["fm"]
    wfm = wfm_metric.get_results()["wfm"]
    sm = sm_metric.get_results()["sm"]
    em = em_metric.get_results()["em"]
    mae = mae_metric.get_results()["mae"]

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

    os.makedirs(os.path.dirname(opt.output), exist_ok=True)
    with open(opt.output, "a+", encoding="utf-8") as f:
        print(
            "Smeasure:", sm.round(3), "; ",
            "meanEm:", "-" if em["curve"] is None else em["curve"].mean().round(3), "; ",
            "wFmeasure:", wfm.round(3), "; ",
            "MAE:", mae.round(3), "; ",
            file=f,
        )


if __name__ == "__main__":
    main()
