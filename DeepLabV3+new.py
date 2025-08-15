import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import os, random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.amp import autocast, GradScaler
import logging

# ---------------------------
# Backbone / ASPP / Decoder / DeepLabv3Plus
# ---------------------------
class ResNetBackbone(nn.Module):
    def __init__(self, output_stride=16):
        super().__init__()
        replace_stride_with_dilation = [False, True, True] if output_stride == 16 else [False, True, True]
        self.base_model = resnet50(pretrained=True, replace_stride_with_dilation=replace_stride_with_dilation)
        for m in [self.base_model.conv1, self.base_model.bn1, self.base_model.layer1]:
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        c1 = self.base_model.layer1(x)
        c2 = self.base_model.layer2(c1)
        c3 = self.base_model.layer3(c2)
        c4 = self.base_model.layer4(c3)
        return c2, c4

class ASPP(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=(6, 12, 18)):
        super().__init__()
        rates = [1, *atrous_rates]
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1 if r == 1 else 3,
                          padding=0 if r == 1 else r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        outs = [branch(x) for branch in self.branches]
        gp = self.global_pool(x)
        outs.append(F.interpolate(gp, size=size, mode='bilinear', align_corners=False))
        return self.project(torch.cat(outs, dim=1))

class Decoder(nn.Module):
    def __init__(self, low_level_channels=512, out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels + 48, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, low_level_features):
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=False)
        low_level_features = self.relu(self.bn1(self.conv1(low_level_features)))
        x = torch.cat((x, low_level_features), dim=1)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class DeepLabv3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, output_stride=16):
        super().__init__()
        self.backbone = ResNetBackbone(output_stride)
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.decoder = Decoder(low_level_channels=512, out_channels=256)
        self.final_conv = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        low_level, high_level = self.backbone(x)
        x = self.aspp(high_level)
        x = self.decoder(x, low_level)
        x = self.final_conv(x)
        return F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

# ---------------------------
# Dataset
# ---------------------------
class SnowIceDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=None, size=(512, 512), cache=False):
        self.transform = transform
        self.size = size
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.lbl_paths = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.cache = cache
        if cache:
            self.cached = [(self._load_img(i), self._load_lbl(i)) for i in range(len(self.img_paths))]

    def _load_img(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB').resize(self.size, Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _load_lbl(self, idx):
        lbl = Image.open(self.lbl_paths[idx]).convert('L').resize(self.size, Image.NEAREST)
        return np.array(lbl, dtype=np.int64)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if self.cache:
            img, lbl = self.cached[idx]
        else:
            img, lbl = self._load_img(idx), self._load_lbl(idx)
        img = self.transform(img) if self.transform else torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return img, torch.from_numpy(lbl).long()

# ---------------------------
# 训练 + 日志
# ---------------------------
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        "batch_size": 16,
        "num_workers": 4,
        "epochs": 2000,
        "lr": 5e-4,
        "weight_decay": 0.01,
        "mixed_precision": True,
        "cache_data": True
    }

    tf = transforms.Compose([transforms.ToTensor()])

    full_train = SnowIceDataset(
        "data/snow_ice_data/img5",
        "data/snow_ice_data/lab5",
        transform=tf, size=(512, 512), cache=config["cache_data"]
    )

    idxs = random.sample(range(len(full_train)), max(1, len(full_train) // 5))
    train_set = Subset(full_train, idxs)

    val_set = SnowIceDataset(
        "data/snow_ice_data/img4",
        "data/snow_ice_data/lab4",
        transform=tf, size=(512, 512), cache=False
    )

    pw = config["num_workers"] > 0
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True,
                              persistent_workers=pw, prefetch_factor=2)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=True,
                            persistent_workers=pw, prefetch_factor=2)

    model = DeepLabv3Plus(3, 3).to(device)
    opt = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=config["lr"] * 2, steps_per_epoch=len(train_loader),
                                        epochs=config["epochs"], pct_start=0.1)
    crit = nn.CrossEntropyLoss()

    scaler = GradScaler(device=device_type) if config["mixed_precision"] else None
    best_miou = -1

    for ep in range(1, config["epochs"] + 1):
        model.train()
        for bi, (imgs, lbls) in enumerate(train_loader, 1):
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, enabled=config["mixed_precision"]):
                out = model(imgs)
                loss = crit(out, lbls)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            sch.step()

        # 验证过程
        if ep % 5 == 0:
            model.eval()
            pred_sum, targ_sum = None, None
            val_loss = 0

            with torch.no_grad(), autocast(device_type=device_type, enabled=config["mixed_precision"]):
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    outs = model(imgs)
                    val_loss += crit(outs, lbls).item()
                    preds = torch.argmax(outs, 1)
                    pred_sum = preds if pred_sum is None else torch.cat((pred_sum, preds), 0)
                    targ_sum = lbls if targ_sum is None else torch.cat((targ_sum, lbls), 0)

            metrics = calculate_metrics(pred_sum, targ_sum)

            if metrics['mIoU'] > best_miou:
                best_miou = metrics['mIoU']
                torch.save({
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "best_miou": best_miou
                }, "best_deeplabv3plus_snow_ice_model.pth")
                logging.info(f"✅ Saved best model (mIoU={best_miou:.4f})")

if __name__ == "__main__":
    train_model()
