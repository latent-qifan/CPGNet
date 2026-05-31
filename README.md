# CPGNet

Official anonymous implementation of **Conditional Polarization Guidance for Camouflaged Object Detection**.

## Overview

CPGNet is an asymmetric RGB-polarization framework for camouflaged object detection. Instead of treating polarization as a parallel modality for heavy fusion, CPGNet uses polarization as conditional structural guidance for hierarchical RGB representation learning.

The main components are:

- **PIM**: integrates AoLP and DoLP to produce reliable polarization guidance.
- **PGE**: injects polarization guidance into multi-scale RGB features.
- **EFM**: performs DoLP edge-guided frequency refinement.
- **IFD**: iteratively refines predictions through feedback decoding.

## Environment

```bash
conda create -n cpgnet python=3.8 -y
conda activate cpgnet
pip install -r requirements.txt
```

## Dataset Structure

Place the PCOD dataset under `datasets/PCOD`:

```text
datasets/PCOD/
  train/
    rgb/
    train-aop/
    train-dop/
    gt/
  test/
    rgb/
    test-aop/
    test-dop/
    gt/
```

The repository does not include datasets, pretrained weights, checkpoints, or prediction maps.

## Pretrained Backbone

Download the PVTv2-B2 pretrained weights and place them at:

```text
pretrained_pvt/pvt_v2_b2.pth
```

## Training

```bash
python train.py \
  --train_path datasets/PCOD/train \
  --test_path datasets/PCOD/test \
  --save_path ckpt \
  --batchsize 4 \
  --trainsize 704 \
  --epoch 100
```

## Testing

```bash
python test.py \
  --pth_path ckpt/Net_epoch_best.pth \
  --data_path datasets/PCOD/test \
  --save_path results/PCOD \
  --testsize 704
```

## Evaluation

```bash
python evaluation/evaluate.py
```

## Reproducibility

The default training configuration follows the manuscript: input size 704, batch size 4, AdamW optimizer, and 100 training epochs.

