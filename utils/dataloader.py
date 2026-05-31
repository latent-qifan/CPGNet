import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageOps

IMG_EXTS = (".jpg", ".png", ".jpeg")
GT_TRAIN_EXTS = (".png",)
GT_TEST_EXTS = (".png", ".tif", ".tiff")


def _is_valid(name: str, exts) -> bool:
    name = name.lower()
    return any(name.endswith(ext) for ext in exts)


def _list_paths(root: str, exts) -> list:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"[DATA] folder not found: {root}")
    return sorted([os.path.join(root, f) for f in os.listdir(root) if _is_valid(f, exts)])


def _key(path: str) -> str:
    """
    生成对齐 key：把文件名去扩展名，并去掉 AoP/DoP 的常见后缀
    例如:
      rgb:  ..._image0.jpg            -> ..._image0
      aop:  ..._image0_aolp.png       -> ..._image0
      dop:  ..._image0_dolp.png       -> ..._image0
    """
    base = os.path.splitext(os.path.basename(path))[0]

    # 你现在最关键的：AoLP 的 _aolp
    if base.endswith("_aolp"):
        base = base[:-5]  # len("_aolp")=5

    # 兼容常见后缀（如果你数据没有也没关系）
    suffixes = [
        "_dolp", "_dop", "_DoLP", "_DOLP",
        "_aop", "_AoLP", "_AOLP",
    ]
    for suf in suffixes:
        if base.endswith(suf):
            base = base[:-len(suf)]
            break

    return base


def _sync_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _rgb_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f).convert("RGB")
        return ImageOps.exif_transpose(img)


def _gray_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f).convert("L")
        return ImageOps.exif_transpose(img)


def _align_and_check_4(images, aops, dops, gts, strict=True, check_size=True, name="train", show_pairs=0):
    """
    按 key 对齐四路数据，保证 (rgb, aop, dop, gt) 一一对应。
    strict=True: 缺失/重复/尺寸不一致直接 raise，避免错配训练
    """
    def build_map(paths, tag):
        m = {}
        dup = []
        for p in paths:
            k = _key(p)
            if k in m:
                dup.append(k)
            m[k] = p
        if dup and strict:
            raise ValueError(f"[DATA CHECK][{name}] duplicate keys in {tag}: {dup[:20]} ... total={len(dup)}")
        return m

    im = build_map(images, "images")
    ao = build_map(aops, "aop")
    do = build_map(dops, "dop")
    gt = build_map(gts, "gts")

    all_keys = sorted(set(im) | set(ao) | set(do) | set(gt))
    missing_img = [k for k in all_keys if k not in im]
    missing_aop = [k for k in all_keys if k not in ao]
    missing_dop = [k for k in all_keys if k not in do]
    missing_gt = [k for k in all_keys if k not in gt]

    if strict and (missing_img or missing_aop or missing_dop or missing_gt):
        msg = [f"[DATA CHECK][{name}] missing files detected:"]
        if missing_img:
            msg.append(f"  - missing images: {missing_img[:20]} ... total={len(missing_img)}")
        if missing_aop:
            msg.append(f"  - missing aop: {missing_aop[:20]} ... total={len(missing_aop)}")
        if missing_dop:
            msg.append(f"  - missing dop: {missing_dop[:20]} ... total={len(missing_dop)}")
        if missing_gt:
            msg.append(f"  - missing gts: {missing_gt[:20]} ... total={len(missing_gt)}")
        raise ValueError("\n".join(msg))

    common = sorted(set(im) & set(ao) & set(do) & set(gt))
    if strict and not common:
        raise ValueError(f"[DATA CHECK][{name}] no matched keys among images/aop/dop/gts.")

    aligned_images = [im[k] for k in common]
    aligned_aops = [ao[k] for k in common]
    aligned_dops = [do[k] for k in common]
    aligned_gts = [gt[k] for k in common]

    if check_size:
        bad = []
        for k in common[: min(len(common), 200)]:  # 前200个快速检查，足够发现系统性问题
            with Image.open(im[k]) as a, Image.open(ao[k]) as b, Image.open(do[k]) as c, Image.open(gt[k]) as d:
                a = ImageOps.exif_transpose(a)
                b = ImageOps.exif_transpose(b)
                c = ImageOps.exif_transpose(c)
                d = ImageOps.exif_transpose(d)
                if not (a.size == b.size == c.size == d.size):
                    bad.append((k, a.size, b.size, c.size, d.size))
        if bad and strict:
            k, s1, s2, s3, s4 = bad[0]
            raise ValueError(
                f"[DATA CHECK][{name}] size mismatch example: key={k} "
                f"image={s1}, aop={s2}, dop={s3}, gt={s4} (bad={len(bad)})"
            )

    if show_pairs > 0:
        print(f"[PAIR CHECK][{name}] show {show_pairs} pairs:")
        for k in common[:show_pairs]:
            print("key:", k)
            print("  rgb:", im[k])
            print("  aop:", ao[k])
            print("  dop:", do[k])
            print("   gt:", gt[k])

    return aligned_images, aligned_aops, aligned_dops, aligned_gts


class CODataset(data.Dataset):
    """train dataloader (rgb + aop + dop + gt)"""

    def __init__(
        self,
        image_root,
        aop_root,
        dop_root,
        gt_root,
        trainsize,
        augmentations,
        strict_check: bool = True,
        show_pairs: int = 0,
        pol_as_gray: bool = False,   # 如果你的 aop/dop 是单通道灰度图，设 True
    ):
        self.trainsize = trainsize
        self.augmentations = (augmentations == "True") if isinstance(augmentations, str) else bool(augmentations)
        self.pol_as_gray = pol_as_gray

        self.images = _list_paths(image_root, IMG_EXTS)
        self.aop_images = _list_paths(aop_root, IMG_EXTS)
        self.dop_images = _list_paths(dop_root, IMG_EXTS)
        self.gts = _list_paths(gt_root, GT_TRAIN_EXTS)

        self.images, self.aop_images, self.dop_images, self.gts = _align_and_check_4(
            self.images, self.aop_images, self.dop_images, self.gts,
            strict=strict_check,
            check_size=True,
            name="train",
            show_pairs=show_pairs
        )

        self.size = len(self.images)

        # transforms
        if self.augmentations:
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.pol_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
            ])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.pol_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = _rgb_loader(self.images[index])

        if self.pol_as_gray:
            aop = _gray_loader(self.aop_images[index])
            dop = _gray_loader(self.dop_images[index])
        else:
            aop = _rgb_loader(self.aop_images[index])
            dop = _rgb_loader(self.dop_images[index])

        gt = _gray_loader(self.gts[index])

        # 保证四路同步增强：同一个 seed
        seed = int(np.random.randint(2_147_483_647))
        _sync_seed(seed); image = self.img_transform(image)
        _sync_seed(seed); aop = self.pol_transform(aop)
        _sync_seed(seed); dop = self.pol_transform(dop)
        _sync_seed(seed); gt = self.gt_transform(gt)

        return image, aop, dop, gt


def get_loader(
    image_root,
    aop_root,
    dop_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    augmentation=False,
    strict_check: bool = True,
    show_pairs: int = 0,
    pol_as_gray: bool = False,
):
    dataset = CODataset(
        image_root=image_root,
        aop_root=aop_root,
        dop_root=dop_root,
        gt_root=gt_root,
        trainsize=trainsize,
        augmentations=augmentation,
        strict_check=strict_check,
        show_pairs=show_pairs,
        pol_as_gray=pol_as_gray,
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


class test_dataset:
    """test dataloader (rgb + aop + dop + gt)"""

    def __init__(self, image_root, aop_root, dop_root, gt_root, testsize, strict_check: bool = True, show_pairs: int = 0, pol_as_gray: bool = False):
        self.testsize = testsize
        self.pol_as_gray = pol_as_gray

        self.images = _list_paths(image_root, IMG_EXTS)
        self.aop_images = _list_paths(aop_root, IMG_EXTS)
        self.dop_images = _list_paths(dop_root, IMG_EXTS)
        self.gts = _list_paths(gt_root, GT_TEST_EXTS)

        self.images, self.aop_images, self.dop_images, self.gts = _align_and_check_4(
            self.images, self.aop_images, self.dop_images, self.gts,
            strict=strict_check,
            check_size=False,
            name="test",
            show_pairs=show_pairs
        )

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.pol_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
        ])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = _rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        if self.pol_as_gray:
            aop = _gray_loader(self.aop_images[self.index])
            dop = _gray_loader(self.dop_images[self.index])
        else:
            aop = _rgb_loader(self.aop_images[self.index])
            dop = _rgb_loader(self.dop_images[self.index])

        aop = self.pol_transform(aop).unsqueeze(0)
        dop = self.pol_transform(dop).unsqueeze(0)

        gt = _gray_loader(self.gts[self.index])

        name = os.path.basename(self.images[self.index])
        if name.lower().endswith(".jpg") or name.lower().endswith(".jpeg"):
            name = os.path.splitext(name)[0] + ".png"

        self.index += 1
        return image, aop, dop, gt, name
