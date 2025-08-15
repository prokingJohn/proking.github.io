# xsimplechat

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from datetime import datetime
import warnings
from torch.utils.data import Subset
import random

warnings.filterwarnings('ignore')


class PSPNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(PSPNet, self).__init__()

        # 骨干网络 - 三个块，下采样3次（缩小8倍）
        self.backbone = nn.Sequential(
            # Block 1: 输入 -> 1/2 -> 1/4
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/4

            # Block 2: 1/4 -> 1/8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/8

            # Block 3: 保持1/8，增加通道数
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 金字塔池化模块
        self.pyramid_pooling = PyramidPoolingModule(in_channels=256, out_channels=64)

        # 最终分类层：输入通道为256（原始特征）+ 4*64（四个池化分支）= 512
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]  # 记录输入大小

        # 骨干网络特征提取
        features = self.backbone(x)  # [B, 256, H/8, W/8]

        # 金字塔池化
        pyramid_features = self.pyramid_pooling(features)  # [B, 512, H/8, W/8]

        # 分类
        output = self.classifier(pyramid_features)  # [B, n_class, H/8, W/8]

        # 上采样到原图大小
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

        return output


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        # 自适应池化层
        self.pools = [nn.AdaptiveAvgPool2d(ps) for ps in pool_sizes]
        # 每个分支的1x1降维卷积
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in pool_sizes
        ])

    def forward(self, x):
        size = x.size()[2:]  # H, W
        out = [x]  # 第一个是原特征
        for pool, conv in zip(self.pools, self.convs):
            pooled = pool(x)                        # 不同尺度池化
            reduced = conv(pooled)                  # 1×1卷积降维
            upsampled = F.interpolate(reduced, size=size, mode='bilinear', align_corners=False)
            out.append(upsampled)                   # 放入结果列表
        return torch.cat(out, dim=1)                # 拼接通道



# 为了兼容原代码，创建一个别名
UNet = PSPNet


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


# ---------------------------
# 改进的训练函数
# ---------------------------
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, CUDA available: {torch.cuda.is_available()}")

    log_file_path = 'PsPNet0814.txt'
    log_file = open(log_file_path, 'a', encoding='utf-8')
    log_file.write("=" * 80 + "\n")
    log_file.write(f"训练开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n")

    config = {
        "batch_size": 8,
        "num_workers": 4,
        "epochs": 2000,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "mixed_precision": True,
        "cache_data": False,
        "gradient_clip": 1.0
    }

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        full_train_dataset = SnowIceDataset(
            image_dir=r"data/snow_ice_data/img5",
            label_dir=r"data/snow_ice_data/lab5",
            transform=train_transform,
            size=(512, 512),
            cache=config["cache_data"]
        )
        # ✅ 随机选取原来的 1/5 样本
        total_indices = list(range(len(full_train_dataset)))
        keep_n = max(1, len(total_indices) // 5)  # 至少保留 1 个样本
        selected_indices = random.sample(total_indices, keep_n)
        train_dataset = Subset(full_train_dataset, selected_indices)

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

    # DataLoader：persistent_workers 仅在 num_workers>0 时启用
    pw = config["num_workers"] > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=pw,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=pw
    )

    model = UNet(in_channels=3, out_channels=3).to(device)
    print("使用PSPNet模型，已加载预定义结构")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler() if config["mixed_precision"] else None
    best_miou = -float('inf')

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            try:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=config["mixed_precision"]):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                if config["mixed_precision"]:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()
                batch_count += 1

                if (batch_idx + 1) % 100 == 0:
                    avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch}/{config['epochs']}, Step {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
                    log_file.write(f"Epoch {epoch}, Step {batch_idx+1}, Loss: {loss.item():.4f}, "
                                   f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}\n")
                    log_file.flush()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU内存不足，跳过批次 {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"运行时错误: {e}")
                    continue

        # ==== 验证 ====
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

            # ---------------------------
            # 日志输出函数（同时输出到控制台和文件）
            # ---------------------------
            def log_message(msg):
                print(msg)
                log_file.write(msg + "\n")  # 写入日志文件，手动添加换行符

            # 格式化输出并记录所有指标
            log_message(f"\n\n========== Epoch {epoch} Validation Results ==========")
            log_message(f"1. 损失指标:")
            log_message(f"   - 验证损失: {val_loss / len(val_loader):.4f}")

            log_message(f"\n2. IoU指标:")
            log_message(f"   - 背景 IoU: {metrics['IoU_per_class'][0]:.4f}")
            log_message(f"   - 冰 IoU:    {metrics['IoU_per_class'][1]:.4f}")
            log_message(f"   - 雪 IoU:    {metrics['IoU_per_class'][2]:.4f}")
            log_message(f"   - 平均 IoU (mIoU): {metrics['mIoU']:.4f}")

            log_message(f"\n3. Dice指标:")
            log_message(f"   - 背景 Dice: {metrics['Dice_per_class'][0]:.4f}")
            log_message(f"   - 冰 Dice:    {metrics['Dice_per_class'][1]:.4f}")
            log_message(f"   - 雪 Dice:    {metrics['Dice_per_class'][2]:.4f}")
            log_message(f"   - 平均 Dice (mDice): {metrics['mDice']:.4f}")

            log_message(f"\n4. 分类综合指标:")
            log_message(f"   - 准确率 (Accuracy):   {metrics['Accuracy']:.4f}")
            log_message(f"   - 精确率 (Precision):   {metrics['Precision']:.4f}")
            log_message(f"   - 召回率 (Recall):     {metrics['Recall']:.4f}")
            log_message(f"   - F1分数 (F-score):    {metrics['Fscore']:.4f}")

            log_message(f"\n5. 误检率 (FAR):")
            log_message(f"   - 背景 FAR: {metrics['FAR_per_class'][0]:.4f}")
            log_message(f"   - 冰 FAR:    {metrics['FAR_per_class'][1]:.4f}")
            log_message(f"   - 雪 FAR:    {metrics['FAR_per_class'][2]:.4f}")
            log_message(f"   - 平均 FAR:   {metrics['mFAR']:.4f}")

            log_message(f"\n6. 混淆矩阵 (行:真实类别, 列:预测类别):")
            log_message("          预测: 背景   冰     雪")
            log_message("    ───────────────────────────────")
            for i, class_name in enumerate(["背景", "冰", "雪"]):
                log_message(
                    f"真实 {class_name}:  {metrics['ConfusionMatrix'][i][0]:6d}  {metrics['ConfusionMatrix'][i][1]:6d}  {metrics['ConfusionMatrix'][i][2]:6d}")

            log_message(f"======================================================\n")

            if metrics['mIoU'] > best_miou:
                best_miou = metrics['mIoU']
                torch.save(model.state_dict(), 'best_PsPnet0814_model.pth')
                log_message(f"✅  当前最佳模型已保存（mIoU: {best_miou:.4f}）")

            model.train()
            # 训练结束后关闭日志文件
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