import os, warnings, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import random
warnings.filterwarnings('ignore')

# ==================== 初始化权重 ====================
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ================= Window Partition & Reverse =================
def window_partition(x, win_size):
    """ x: (B, H, W, C) -> windows: (B*nW, win, win, C) """
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows, win_size, H, W):
    """ windows: (B*nW, win, win, C) -> (B, H, W, C) """
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ================= Window Attention =================
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=2, win_size=8):
        super().__init__()
        self.win_size = win_size
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B,H,W,C)
        Hp = (H + self.win_size - 1) // self.win_size * self.win_size
        Wp = (W + self.win_size - 1) // self.win_size * self.win_size
        pad_h, pad_w = Hp - H, Wp - W
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        # Partition windows
        windows = window_partition(x, self.win_size)  # (B*nW, win, win, C)
        win_flat = windows.view(-1, self.win_size * self.win_size, C)

        # Window Attention
        attn_out, _ = self.attn(self.norm(win_flat), self.norm(win_flat), self.norm(win_flat))
        attn_out = attn_out.view(-1, self.win_size, self.win_size, C)

        # Merge windows
        x = window_reverse(attn_out, self.win_size, Hp, Wp)
        x = x[:, :H, :W, :].permute(0, 3, 1, 2)  # (B,C,H,W)
        return x


# ================= MP-Former Block with Window MHA =================
class MPFormerWinBlock(nn.Module):
    def __init__(self, dim, num_heads=2, mlp_ratio=4.0, win_size=8):
        super().__init__()
        self.dim = dim
        split_dim = dim // 2

        # Conv Path
        self.conv_path = nn.Sequential(
            nn.Conv2d(split_dim, split_dim, 3, padding=1, groups=split_dim, bias=False),
            nn.BatchNorm2d(split_dim),
            nn.ReLU(True)
        )

        # Transformer Path (Window Attention)
        self.global_attn = WindowAttention(split_dim, num_heads=num_heads, win_size=win_size)
        self.norm2 = nn.LayerNorm(split_dim)
        hidden_dim = int(split_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(split_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, split_dim)
        )

        # Fuse Path
        self.fuse = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        half = C // 2

        # Local conv branch
        local_feat = self.conv_path(x[:, :half])

        # Global transformer branch
        global_feat = self.global_attn(x[:, half:])
        B2, C2, H2, W2 = global_feat.shape
        gf_flat = global_feat.permute(0, 2, 3, 1).reshape(B2, H2 * W2, C2)
        gf_flat = self.ffn(self.norm2(gf_flat)).reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)

        # Merge
        out = torch.cat([local_feat, gf_flat], dim=1)
        out = self.fuse(out)
        return out + x


# ================= Encoder/Decoder =================
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, win_size=8):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            MPFormerWinBlock(out_ch, win_size=win_size)
        )

    def forward(self, x):
        return self.down(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, win_size=8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            MPFormerWinBlock(out_ch, win_size=win_size)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ================= MP-Former Window UNet =================
class MPFormerWindowUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, win_size=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            MPFormerWinBlock(64, win_size=win_size)
        )
        self.enc1 = EncoderBlock(64, 128, win_size)
        self.enc2 = EncoderBlock(128, 256, win_size)
        self.enc3 = EncoderBlock(256, 512, win_size)
        self.enc4 = EncoderBlock(512, 1024, win_size)

        self.dec1 = DecoderBlock(1024, 512, 512, win_size)
        self.dec2 = DecoderBlock(512, 256, 256, win_size)
        self.dec3 = DecoderBlock(256, 128, 128, win_size)
        self.dec4 = DecoderBlock(128, 64, 64, win_size)

        self.outc = nn.Conv2d(64, out_ch, 1)
        self.apply(init_weights)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.enc1(s1)
        s3 = self.enc2(s2)
        s4 = self.enc3(s3)
        b = self.enc4(s4)
        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)
        return self.outc(d4)


# ================= 绑定训练框架 =================
UNet = MPFormerWindowUNet

# ==================== 数据集 ====================
class SnowIceDataset(Dataset):
    def __init__(self, imgd, lbld, tf=None, size=(512,512), cache=False):
        self.imgs = sorted(f for f in os.listdir(imgd) if f.endswith(('png','jpg','jpeg')))
        self.lbls = sorted(f for f in os.listdir(lbld) if f.endswith(('png','jpg','jpeg')))
        self.imgd, self.lbld, self.tf, self.size, self.cache = imgd, lbld, tf, size, cache
        assert len(self.imgs) == len(self.lbls)
        if cache: self.cache_data()
    def cache_data(self):
        self.ci, self.cl = [], []
        for im, lm in zip(self.imgs, self.lbls):
            img = Image.open(os.path.join(self.imgd, im)).convert('RGB').resize(self.size, Image.BILINEAR)
            lbl = Image.open(os.path.join(self.lbld, lm)).convert('L').resize(self.size, Image.NEAREST)
            self.ci.append(np.array(img, dtype=np.float32)/255.)
            self.cl.append(np.clip(np.array(lbl, dtype=np.int64), 0, 2))
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        if self.cache:
            img, lbl = self.ci[idx], self.cl[idx]
        else:
            img = Image.open(os.path.join(self.imgd, self.imgs[idx])).convert('RGB').resize(self.size, Image.BILINEAR)
            lbl = Image.open(os.path.join(self.lbld, self.lbls[idx])).convert('L').resize(self.size, Image.NEAREST)
            img, lbl = np.array(img, dtype=np.float32)/255., np.clip(np.array(lbl, dtype=np.int64), 0, 2)
        img = torch.from_numpy(img).permute(2,0,1) if not self.tf else self.tf(img)
        return img, torch.from_numpy(lbl).long()

# ==================== metrics计算 ====================
def calculate_metrics(pred, target, C=3):
    p, t = pred.long().view(-1), target.long().view(-1)
    pc, tc = torch.bincount(p, minlength=C).float(), torch.bincount(t, minlength=C).float()
    inter = torch.bincount(t[p==t], minlength=C).float()
    union = pc + tc - inter
    tp, fp, fn = inter, pc-inter, tc-inter
    tot = torch.tensor(p.numel(), dtype=torch.float, device=pred.device)
    tn = tot - tc - fp
    iou = inter/union.clamp(1e-8); dice = 2*inter/(pc+tc).clamp(1e-8)
    prec = tp/(tp+fp).clamp(1e-8); rec = tp/(tp+fn).clamp(1e-8)
    f1 = 2*(prec*rec)/(prec+rec).clamp(1e-8); far = fp/(fp+tn).clamp(1e-8)
    cm = torch.bincount(t*C + p, minlength=C*C).view(C, C).long()
    return {'IoU_per_class':iou.tolist(),'mIoU':iou.mean().item(),
            'Dice_per_class':dice.tolist(),'mDice':dice.mean().item(),
            'Accuracy':tp.sum().item()/tot.item(),'Precision':prec.mean().item(),
            'Recall':rec.mean().item(),'Fscore':f1.mean().item(),
            'FAR_per_class':far.tolist(),'mFAR':far.mean().item(),
            'ConfusionMatrix':cm.tolist()}

# ==================== 训练函数 (无辅助损失) ====================
def train_model():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {"batch_size": 4, "num_workers": 2, "epochs": 2000, "lr": 1e-4, "weight_decay": 0.01,
           "mixed_precision": True, "cache_data": False, "gradient_clip": 1.0}
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_ds = SnowIceDataset("data/snow_ice_data/image/train", "data/snow_ice_data/label/train", tf, (512,512), cfg["cache_data"])
    total_indices = list(range(len(train_ds)))
    keep_n = max(1, len(total_indices))
    train_indices = random.sample(total_indices, keep_n)
    train_ds = Subset(train_ds, train_indices)
    val_ds = SnowIceDataset("data/snow_ice_data/image/val", "data/snow_ice_data/label/val", tf, (512,512))
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"], pin_memory=True,
                          persistent_workers=(cfg["num_workers"]>0), drop_last=True)
    val_dl = DataLoader(val_ds, cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True,
                        persistent_workers=(cfg["num_workers"]>0))
    model = UNet(3, 3).to(dev)
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-8)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 50, 2, 1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler() if cfg["mixed_precision"] else None
    best = -1
    log = open('mpformer.txt','a',encoding='utf-8')
    log.write(f"{'='*40} 开始 {datetime.now()} {'='*40}\n")
    for ep in range(1, cfg["epochs"]+1):
        model.train(); run_loss=0; count=0
        for i,(img,lbl) in enumerate(train_dl):
            img,lbl = img.to(dev), lbl.to(dev)
            opt.zero_grad(set_to_none=True)
            with autocast(cfg["mixed_precision"]):
                out = model(img)
                loss = crit(out,lbl)
            if cfg["mixed_precision"]:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                opt.step()
            sch.step()
            run_loss += loss.item(); count += 1
            if (i + 1) % 100 == 0:
                print(
                    f"Ep{ep} It{i + 1}/{len(train_dl)} Loss:{loss.item():.4f} Avg:{run_loss / count:.4f} LR:{sch.get_last_lr()[0]:.6f}")

        if ep % 5 == 0:
            model.eval(); vloss=0; pp,tt = [], []
            with torch.no_grad(), autocast(cfg["mixed_precision"]):
                for img,lbl in val_dl:
                    img,lbl = img.to(dev), lbl.to(dev)
                    o = model(img)
                    vloss += crit(o,lbl).item()
                    pp.append(o.argmax(1).cpu()); tt.append(lbl.cpu())
            m = calculate_metrics(torch.cat(pp), torch.cat(tt))

            def log_msg(s):
                print(s)
                log.write(s + "\n")

            log_msg(f"\n========== Epoch {ep} Validation Results ==========")
            log_msg(f"1. 损失指标:\n   - 验证损失: {vloss / len(val_dl):.4f}")
            log_msg("\n2. IoU指标:")
            for n, v in zip(["背景", "冰", "雪"], m["IoU_per_class"]):
                log_msg(f"   - {n} IoU: {v:.4f}")
            log_msg(f"   - 平均 IoU: {m['mIoU']:.4f}")
            log_msg("\n3. Dice指标:")
            for n, v in zip(["背景", "冰", "雪"], m["Dice_per_class"]):
                log_msg(f"   - {n} Dice: {v:.4f}")
            log_msg(f"   - 平均 Dice: {m['mDice']:.4f}")
            log_msg(
                f"\n4. 分类综合指标:\n   - 准确率: {m['Accuracy']:.4f}\n   - 精确率: {m['Precision']:.4f}\n   - 召回率: {m['Recall']:.4f}\n   - F1分数: {m['Fscore']:.4f}")
            log_msg("\n5. 误检率 (FAR):")
            for n, v in zip(["背景", "冰", "雪"], m["FAR_per_class"]):
                log_msg(f"   - {n} FAR: {v:.4f}")
            log_msg(f"   - 平均 FAR: {m['mFAR']:.4f}")
            log_msg("\n6. 混淆矩阵:")
            log_msg("          预测: 背景   冰     雪")
            log_msg("    ───────────────────────────────")
            for i, row in enumerate(m["ConfusionMatrix"]):
                log_msg(f"真实 {['背景', '冰', '雪'][i]}:  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")
            if m['mIoU'] > best:
                best = m['mIoU']
                torch.save(model.state_dict(), 'best_Swinunet_model.pth')
                log_msg(f"✅  当前最佳模型已保存（mIoU: {best:.4f}）")
    log.write(f"{'=' * 40} 结束 {datetime.now()} {'=' * 40}\n")
    log.close()


if __name__ == "__main__":
    train_model()
