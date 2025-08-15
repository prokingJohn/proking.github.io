import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler


class LightConvBlock(nn.Module):
    """使用深度可分离卷积的轻量级卷积块"""
    def __init__(self, in_ch, out_ch):
        super(LightConvBlock, self).__init__()
        self.conv = nn.Sequential(
            # 深度可分离卷积（减少参数数量）
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),  # 深度卷积
            nn.Conv2d(in_ch, out_ch, 1),  # 逐点卷积
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # 第二层卷积
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPP(nn.Module):
    """通过通道压缩和轻量级卷积降低显存占用"""
    def __init__(self, in_ch=3, out_ch=3):
        super(UNetPP, self).__init__()

        # 编码器（通道数减半，减少50%参数）
        self.enc0_0 = LightConvBlock(in_ch, 32)   # 原64→32
        self.pool0 = nn.MaxPool2d(2)
        self.enc1_0 = LightConvBlock(32, 64)    # 原128→64
        self.pool1 = nn.MaxPool2d(2)
        self.enc2_0 = LightConvBlock(64, 128)   # 原256→128
        self.pool2 = nn.MaxPool2d(2)
        self.enc3_0 = LightConvBlock(128, 256)  # 原512→256
        self.pool3 = nn.MaxPool2d(2)
        self.enc4_0 = LightConvBlock(256, 512)  # 原1024→512（底层通道压缩）

        # 解码器（匹配压缩后的通道数）
        self.up3_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3_1 = LightConvBlock(256+512, 256)  # 编码器输出+上采样特征

        self.up2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2_1 = LightConvBlock(128+256, 128)

        self.up1_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1_1 = LightConvBlock(64+128, 64)

        self.up0_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec0_1 = LightConvBlock(32+64, 32)

        # 输出层
        self.output = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        # 编码器（输入尺寸512→256→128→64→32→16）
        enc0_0 = self.enc0_0(x)          # (B,32,512,512)
        enc1_0 = self.enc1_0(self.pool0(enc0_0))  # (B,64,256,256)
        enc2_0 = self.enc2_0(self.pool1(enc1_0))  # (B,128,128,128)
        enc3_0 = self.enc3_0(self.pool2(enc2_0))  # (B,256,64,64)
        enc4_0 = self.enc4_0(self.pool3(enc3_0))  # (B,512,32,32)

        # 解码器（上采样恢复尺寸）
        dec3_1 = self.dec3_1(torch.cat([enc3_0, self.up3_1(enc4_0)], dim=1))  # (B,256,64,64)
        dec2_1 = self.dec2_1(torch.cat([enc2_0, self.up2_1(dec3_1)], dim=1))  # (B,128,128,128)
        dec1_1 = self.dec1_1(torch.cat([enc1_0, self.up1_1(dec2_1)], dim=1))  # (B,64,256,256)
        dec0_1 = self.dec0_1(torch.cat([enc0_0, self.up0_1(dec1_1)], dim=1))  # (B,32,512,512)

        return self.output(dec0_1)

UNet = UNetPP


# ---------------------------
# 改进的数据加载类
# ---------------------------
class SnowIceDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, size=(512, 512), cache=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.size = size
        self.cache = cache

        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.labels = sorted([f for f in os.listdir(label_dir) if f.endswith(('png', 'jpg', 'jpeg'))])

        # 确保图像和标签文件数量匹配
        assert len(self.images) == len(self.labels), "图像和标签文件数量不匹配"

        if self.cache:
            self._preload_data()

    def _preload_data(self):
        self.cached_images = []
        self.cached_labels = []
        for img_name, lbl_name in zip(self.images, self.labels):
            try:
                img = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
                img = img.resize(self.size, resample=Image.BILINEAR)
                img = np.array(img, dtype=np.float32) / 255.0

                lbl = Image.open(os.path.join(self.label_dir, lbl_name)).convert('L')
                lbl = lbl.resize(self.size, resample=Image.NEAREST)
                lbl = np.array(lbl, dtype=np.int64)

                # 确保标签值在有效范围内
                lbl = np.clip(lbl, 0, 2)

                self.cached_images.append(img)
                self.cached_labels.append(lbl)
            except Exception as e:
                print(f"Error loading {img_name} or {lbl_name}: {e}")
                continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.cache:
            img = self.cached_images[idx]
            lbl = self.cached_labels[idx]
        else:
            try:
                img_name = self.images[idx]
                lbl_name = self.labels[idx]

                img = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
                img = img.resize(self.size, resample=Image.BILINEAR)
                img = np.array(img, dtype=np.float32) / 255.0

                lbl = Image.open(os.path.join(self.label_dir, lbl_name)).convert('L')
                lbl = lbl.resize(self.size, resample=Image.NEAREST)
                lbl = np.array(lbl, dtype=np.int64)

                # 确保标签值在有效范围内
                lbl = np.clip(lbl, 0, 2)
            except Exception as e:
                print(f"Error loading image {idx}: {e}")
                # 返回零图像和标签
                img = np.zeros((self.size[0], self.size[1], 3), dtype=np.float32)
                lbl = np.zeros(self.size, dtype=np.int64)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        lbl = torch.from_numpy(lbl).long()
        return img, lbl

from datetime import datetime
# ---------------------------
# 训练主函数（保持不变）
# ---------------------------
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, CUDA available: {torch.cuda.is_available()}")

    # 新增：日志文件初始化
    log_file_path = 'unetpp.txt'
    log_file = open(log_file_path, 'a', encoding='utf-8')  # 追加模式打开日志文件
    log_file.write("=" * 80 + "\n")
    log_file.write(f"训练开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n")

    config = {
        "batch_size": 4,  # 减小批量大小
        "num_workers": 4,
        "epochs": 2000,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "mixed_precision": True,
        "cache_data": False,
        "gradient_clip": 1.0,
        "accumulation_steps": 4  # 添加梯度累积
    }

    # 数据增强
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = SnowIceDataset(
            image_dir=r"data/snow_ice_data/img4",
            label_dir=r"data/snow_ice_data/lab4",
            transform=train_transform,
            size=(512, 512),
            cache=config["cache_data"]
        )
        val_dataset = SnowIceDataset(
            image_dir=r"data/snow_ice_data/img4",
            label_dir=r"data/snow_ice_data/lab4",
            transform=val_transform,
            size=(512, 512),
            cache=False
        )
    except Exception as e:
        print(f"数据加载错误: {e}")
        log_file.close()
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True  # 丢弃最后不完整的批次
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )

    model = UNet(in_ch=3, out_ch=3).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        eps=1e-8  # 增加数值稳定性
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if config["mixed_precision"] else None

    best_miou = -float('inf')
    log_file = open('Unet++.txt', 'w', encoding='utf-8')

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config["mixed_precision"]):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if config["mixed_precision"]:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                log_msg = f"Epoch {epoch}/{config['epochs']}, Step {batch_idx + 1}/{len(train_loader)}, " \
                          f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
                print(log_msg)
                log_file.write(log_msg + '\n')

        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad(), autocast(enabled=config["mixed_precision"]):
                for images, labels in val_loader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()

                    preds = torch.argmax(outputs, dim=1).cpu()
                    all_preds.append(preds)
                    all_targets.append(labels.cpu())

            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            metrics = calculate_metrics(all_preds, all_targets)

            log_msg = f"\n\n========== Epoch {epoch} Validation Results ==========\n"
            log_msg += f"1. 损失指标:\n   - 验证损失: {val_loss / len(val_loader):.4f}\n"
            log_msg += f"\n2. IoU指标:\n   - 背景 IoU: {metrics['IoU_per_class'][0]:.4f}\n   - 冰 IoU:    {metrics['IoU_per_class'][1]:.4f}\n   - 雪 IoU:    {metrics['IoU_per_class'][2]:.4f}\n   - 平均 IoU (mIoU): {metrics['mIoU']:.4f}\n"
            log_msg += f"\n3. Dice指标:\n   - 背景 Dice: {metrics['Dice_per_class'][0]:.4f}\n   - 冰 Dice:    {metrics['Dice_per_class'][1]:.4f}\n   - 雪 Dice:    {metrics['Dice_per_class'][2]:.4f}\n   - 平均 Dice (mDice): {metrics['mDice']:.4f}\n"
            log_msg += f"\n4. 分类综合指标:\n   - 准确率 (Accuracy):   {metrics['Accuracy']:.4f}\n   - 精确率 (Precision):   {metrics['Precision']:.4f}\n   - 召回率 (Recall):     {metrics['Recall']:.4f}\n   - F1分数 (F-score):    {metrics['Fscore']:.4f}\n"
            log_msg += f"\n5. 误检率 (FAR):\n   - 背景 FAR: {metrics['FAR_per_class'][0]:.4f}\n   - 冰 FAR:    {metrics['FAR_per_class'][1]:.4f}\n   - 雪 FAR:    {metrics['FAR_per_class'][2]:.4f}\n   - 平均 FAR:   {metrics['mFAR']:.4f}\n"
            log_msg += f"\n6. 混淆矩阵 (行:真实类别, 列:预测类别):\n          预测: 背景   冰     雪\n    ───────────────────────────────\n"
            for i, class_name in enumerate(["背景", "冰", "雪"]):
                log_msg += f"真实 {class_name}:  {metrics['ConfusionMatrix'][i][0]:6d}  {metrics['ConfusionMatrix'][i][1]:6d}  {metrics['ConfusionMatrix'][i][2]:6d}\n"
            log_msg += "======================================================\n"

            print(log_msg)
            log_file.write(log_msg)

            if metrics['mIoU'] > best_miou:
                best_miou = metrics['mIoU']
                torch.save(model.state_dict(), 'best_unetpp_snow_ice_model.pth')
                save_msg = f"✅  当前最佳模型已保存（mIoU: {best_miou:.4f}）"
                print(save_msg)
                log_file.write(save_msg + '\n')

            model.train()

    log_file.write("=" * 80 + "\n")
    log_file.write(f"训练结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n")
    log_file.close()
def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> dict:
    # 确保数据在同一设备且为长整型
    pred = pred.to(torch.long)
    target = target.to(torch.long)
    assert pred.device == target.device, "pred和target必须在同一设备"

    # 展开为一维张量（方便后续统计）
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # ---------------------------
    # 基础统计量（向量化计算）
    # ---------------------------
    # 计算每个类别的预测数量（P_c）和真实数量（T_c）
    pred_count = torch.bincount(pred_flat, minlength=num_classes).float()  # shape [C]
    target_count = torch.bincount(target_flat, minlength=num_classes).float()  # shape [C]

    # 计算交集（Intersection_c）: 预测和真实同时为c的像素数
    mask = (pred_flat == target_flat)  # 预测与真实相同的位置
    intersection = torch.bincount(target_flat[mask], minlength=num_classes).float()  # shape [C]

    # 计算并集（Union_c）= P_c + T_c - Intersection_c
    union = pred_count + target_count - intersection  # shape [C]

    # 计算TP、FP、FN（用于FAR和混淆矩阵）
    tp = intersection
    fp = pred_count - tp
    fn = target_count - tp

    # 总像素数（转换为PyTorch张量，避免int类型调用clamp）
    total_pixels = torch.tensor(pred_flat.numel(), device=pred.device, dtype=torch.float)  # shape []

    # TN相关计算（真实非c的总像素数 - FP_c）
    tn_total = total_pixels - target_count  # shape [C]
    tn = tn_total - fp  # shape [C]

    # IoU（交并比）: Intersection / Union（避免除零）
    iou_per_class = intersection / union.clamp(min=1e-8)  # shape [C]
    miou = iou_per_class.mean()  # 平均IoU

    # Dice系数: 2*Intersection / (P_c + T_c)
    dice_per_class = 2 * intersection / (pred_count + target_count).clamp(min=1e-8)  # shape [C]
    mdice = dice_per_class.mean()  # 平均Dice

    # 准确率: (TP_total) / 总像素数
    accuracy = tp.sum() / total_pixels.clamp(min=1e-8)  # 修复：total_pixels现在是张量

    # 精确率（宏平均）: (TP_c / (TP_c + FP_c)) 的平均
    precision_per_class = tp / (tp + fp).clamp(min=1e-8)  # shape [C]
    precision = precision_per_class.mean()

    # 召回率（宏平均）: (TP_c / (TP_c + FN_c)) 的平均
    recall_per_class = tp / (tp + fn).clamp(min=1e-8)  # shape [C]
    recall = recall_per_class.mean()

    # F1分数: 2*(P*R)/(P+R) 的平均
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class).clamp(
        min=1e-8)  # shape [C]
    f1 = f1_per_class.mean()

    # FAR（误检率）: FP_c / (FP_c + TN_c) 的平均
    far_per_class = fp / (fp + tn).clamp(min=1e-8)  # shape [C]
    mfar = far_per_class.mean()

    indices = target_flat * num_classes + pred_flat  # 转换为一维索引（target*C + pred）
    cm = torch.bincount(indices, minlength=num_classes * num_classes).view(num_classes,
                                                                           num_classes).long()  # shape [C, C]

    return {
        'IoU_per_class': iou_per_class.tolist(),
        'mIoU': miou.item(),
        'Dice_per_class': dice_per_class.tolist(),
        'mDice': mdice.item(),
        'Accuracy': accuracy.item(),
        'Precision': precision.item(),
        'Recall': recall.item(),
        'Fscore': f1.item(),
        'FAR_per_class': far_per_class.tolist(),
        'mFAR': mfar.item(),
        'ConfusionMatrix': cm.tolist()
    }

if __name__ == "__main__":
    train_model()
