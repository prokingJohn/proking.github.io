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


# ---------------------------
# 深度可分离卷积模块
# ---------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)


# ---------------------------
# FCN模型
# ---------------------------
class FCN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, efficient=True):
        super(FCN, self).__init__()
        self.efficient = efficient
        conv_block = DepthwiseSeparableConv if efficient else self._standard_conv_block

        # --------------------------- 编码器（严格尺寸对齐的下采样路径） ---------------------------
        # 第1层：输入→64通道，尺寸1/2（保持尺寸整除）
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)  # 输出尺寸: H/2, W/2

        # 第2层：64→128通道，尺寸1/4
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)  # 输出尺寸: H/4, W/4

        # 第3层：128→256通道，尺寸1/8（保存pool3输出）
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, ceil_mode=True)  # 输出尺寸: H/8, W/8

        # 第4层：256→512通道，尺寸1/16（保存pool4输出）
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, ceil_mode=True)  # 输出尺寸: H/16, W/16

        # 第5层：512→1024通道，尺寸1/32（保存pool5输出）
        self.enc5 = conv_block(512, 1024)
        self.pool5 = nn.MaxPool2d(2, ceil_mode=True)  # 输出尺寸: H/32, W/32

        # --------------------------- 解码器（动态尺寸适配的8s跳跃连接） ---------------------------
        # 1. pool5上采样模块（2倍上采样，动态匹配pool4尺寸）
        self.up_pool5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0),  # 2倍上采样
            nn.ReLU(inplace=True)
        )

        # 2. pool4融合后上采样模块（2倍上采样，动态匹配pool3尺寸）
        self.up_fused4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),  # 2倍上采样
            nn.ReLU(inplace=True)
        )

        # 3. 最终8倍上采样（动态恢复原始尺寸）
        self.classifier = nn.Conv2d(256, out_channels, 1)
        self.up_final = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=8, stride=8, padding=0, bias=False
        )

    def _standard_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # padding=1保持尺寸
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # padding=1保持尺寸
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --------------------------- 编码器正向传播（保存三层特征图） ---------------------------
        # 第1-3层：下采样至1/8尺寸（pool3）
        x = self.enc1(x)  # 尺寸不变（3x3卷积+padding=1）
        x = self.pool1(x)  # (B, 64, H/2, W/2)

        x = self.enc2(x)  # 尺寸不变
        x = self.pool2(x)  # (B, 128, H/4, W/4)

        x = self.enc3(x)  # 尺寸不变
        x_pool3 = self.pool3(x)  # (B, 256, H/8, W/8)  ← 关键保存层

        # 第4层：下采样至1/16尺寸（pool4）
        x = self.enc4(x_pool3)  # 输入H/8→经conv_block后尺寸不变
        x_pool4 = self.pool4(x)  # (B, 512, H/16, W/16)  ← 关键保存层

        # 第5层：下采样至1/32尺寸（pool5）
        x = self.enc5(x_pool4)  # 输入H/16→经conv_block后尺寸不变
        x_pool5 = self.pool5(x)  # (B, 1024, H/32, W/32)  ← 关键保存层

        # --------------------------- 解码器正向传播（尺寸对齐校验） ---------------------------
        # 1. pool5上采样2倍 → 尺寸H/16, W/16（与pool4一致）
        x_pool5_up = self.up_pool5(x_pool5)
        # assert x_pool5_up.shape[2:] == x_pool4.shape[2:], \
        #     f"pool5上采样后尺寸不匹配: {x_pool5_up.shape[2:]} vs pool4 {x_pool4.shape[2:]}"
        x_fused4 = x_pool5_up + x_pool4  # 融合后尺寸: (B, 512, H/16, W/16)

        # 2. 融合结果上采样2倍 → 尺寸H/8, W/8（与pool3一致）
        x_fused4_up = self.up_fused4(x_fused4)
        # assert x_fused4_up.shape[2:] == x_pool3.shape[2:], \
        #     f"pool4融合后上采样尺寸不匹配: {x_fused4_up.shape[2:]} vs pool3 {x_pool3.shape[2:]}"
        x_fused8 = x_fused4_up + x_pool3  # 融合后尺寸: (B, 256, H/8, W/8)

        # 3. 最终8倍上采样恢复原始尺寸
        x_class = self.classifier(x_fused8)  # (B, out_channels, H/8, W/8)
        x_output = self.up_final(x_class)     # (B, out_channels, H, W)

        return x_output

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
# 表格化输出混淆矩阵
# ---------------------------
def print_confusion_matrix(cm, class_names=('Background', 'Ice', 'Snow')):
    """
    打印格式化的混淆矩阵表格
    cm: 混淆矩阵 (numpy array)
    class_names: 类别名称列表
    """
    num_classes = len(class_names)
    print("\n混淆矩阵:")
    print("          | " + " | ".join([f"{cls:^10}" for cls in class_names]) + " |")
    print("-" * (len(class_names) * 13 + 3))
    for i in range(num_classes):
        row = f"{class_names[i]:<10} | "
        for j in range(num_classes):
            row += f"{cm[i, j]:^10} | "
        print(row)
        print("-" * (len(class_names) * 13 + 3))


# ---------------------------
# 训练函数（整合FAR输出）
# ---------------------------
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, CUDA available: {torch.cuda.is_available()}")
    # config = {
    #     "batch_size": 16, "num_workers": 4, "epochs": 400,
    #     "lr": 5e-4, "weight_decay": 0.01,
    #     "mixed_precision": True, "efficient_conv": True
    # }

    log_file_path = 'FCN-8s.txt'
    log_file = open(log_file_path, 'a', encoding='utf-8')  # 追加模式打开日志文件
    log_file.write("=" * 80 + "\n")
    log_file.write(f"训练开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("=" * 80 + "\n")

    config = {
        "batch_size": 8,  # 减小批次大小
        "num_workers": 4,
        "epochs": 2000,
        "lr": 1e-4,  # 降低学习率
        "weight_decay": 0.01,
        "mixed_precision": True,
        "cache_data": False,
        "gradient_clip": 1.0,  # 添加梯度裁剪
        "efficient_conv": True
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
            image_dir=r"data/snow_ice_data/image/train",
            label_dir=r"data/snow_ice_data/label/train",
            transform=train_transform,
            size=(512, 512),
            cache=config["cache_data"]
        )
        val_dataset = SnowIceDataset(
            image_dir=r"data/snow_ice_data/image/val",
            label_dir=r"data/snow_ice_data/label/val",
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

    model = FCN(efficient=config["efficient_conv"]).to(device)

    # 使用更稳定的优化器设置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        eps=1e-8  # 增加数值稳定性
    )

    # 使用余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if config["mixed_precision"] else None
    best_miou = -float('inf')

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(config["mixed_precision"]):
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
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}/{config['epochs']}, Step {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

                # 记录到日志文件
                log_file.write(f"Epoch {epoch}, Step {batch_idx + 1}, Loss: {loss.item():.4f}, "
                               f"LR: {current_lr:.6f}\n")
                log_file.flush()
        if (epoch % 5 == 0):
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
            metrics = calculate_metrics(all_preds, all_targets)  # 包含FAR的指标字典

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
                torch.save(model.state_dict(), 'best_unet8s_snow_ice_model.pth')
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