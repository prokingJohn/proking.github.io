import os, warnings, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from torch.utils.data import Subset
import random
warnings.filterwarnings('ignore')

# ==================== 通用初始化 ====================
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# ==================== 模型部分 ====================
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, down=None, dil=1):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(inplanes, planes, 1, bias=False), nn.BatchNorm2d(planes)
        self.conv2, self.bn2 = nn.Conv2d(planes, planes, 3, stride, dil, dil, bias=False), nn.BatchNorm2d(planes)
        self.conv3, self.bn3 = nn.Conv2d(planes, planes*self.expansion, 1, bias=False), nn.BatchNorm2d(planes*self.expansion)
        self.relu, self.down = nn.ReLU(True), down
        self.apply(init_weights)
    def forward(self, x):
        identity = x
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
            x = self.relu(bn(conv(x)))
        if self.down: identity = self.down(identity)
        return self.relu(x+identity)

def make_layer(block, inplanes, planes, blocks, stride=1, dil=1):
    down = None
    if stride != 1 or inplanes != planes * block.expansion:
        down = nn.Sequential(nn.Conv2d(inplanes, planes*block.expansion, 1, stride, bias=False),
                             nn.BatchNorm2d(planes*block.expansion))
        down.apply(init_weights)
    layers = [block(inplanes, planes, stride, down, dil)]
    layers += [block(planes*block.expansion, planes, dil=dil) for _ in range(1, blocks)]
    return nn.Sequential(*layers)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, rates=(1,3,5)):
        super().__init__()
        self.atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1 if r==1 else 3, padding=0 if r==1 else r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(True)
            ) for r in rates
        ])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.project = nn.Sequential(nn.Conv2d(out_ch*(len(rates)+1), in_ch, 1, bias=False),
                                     nn.BatchNorm2d(in_ch), nn.ReLU(True), nn.Dropout(0.1))
        self.apply(init_weights)
    def forward(self, x):
        size = x.shape[2:]
        feats = [m(x) for m in self.atrous]
        gp = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False)
        return self.project(torch.cat(feats+[gp], 1))

class ResNetUNetASPPDilated(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxp = nn.MaxPool2d(3, 2, 1)
        self.l1 = make_layer(Bottleneck, 64, 64, 3)
        self.l2 = make_layer(Bottleneck, 256, 128, 4, 2)
        self.l3 = make_layer(Bottleneck, 512, 256, 6, 2, 2)
        self.l4 = make_layer(Bottleneck, 1024, 512, 3, 2, 4)
        self.aspp = ASPP(2048, 256)
        self.aux = self.make_aux(512, out_ch)
        # 修复后的解码器通道配置
        up_in_out = [(2048, 1024), (1024, 512), (512, 256), (256, 128), (128, 64)]
        skip_chs = [1024, 512, 256, 64, 0]
        dils = [(1, 2), (1, 2), (1, 1), (1, 1), (1, 1)]
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for (cin, cout), skip, dil in zip(up_in_out, skip_chs, dils):
            self.ups.append(nn.ConvTranspose2d(cin, cout, 2, 2))
            in_c = cout + skip if skip > 0 else cout
            self.decoders.append(self.make_dec(in_c, cout, dil))
        self.outc = nn.Conv2d(64, out_ch, 1)
        self.apply(init_weights)
    def make_aux(self, cin, out_ch):
        mods = []
        for c1, c2 in [(cin, 256), (256, 128), (128, 64)]:
            mods += [nn.Conv2d(c1, c2, 3, 1, 1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(True),
                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
        mods.append(nn.Conv2d(64, out_ch, 1)); return nn.Sequential(*mods)
    def make_dec(self, cin, cout, d):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=d[0], dilation=d[0], bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=d[1], dilation=d[1], bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(True)
        )
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x))); x1p = self.maxp(x1)
        x2 = self.l1(x1p); x3 = self.l2(x2); aux = self.aux(x3)
        x4 = self.l3(x3); x5 = self.l4(x4); x5 = self.aspp(x5)
        feats = [x4, x3, x2, x1]
        for i in range(5):
            up = self.ups[i](x5 if i == 0 else out)
            out = self.decoders[i](torch.cat([feats[i], up], 1) if i < 4 else up)
            if i < 3: out = F.dropout2d(out, 0.1)
        return self.outc(out), aux

UNet = ResNetUNetASPPDilated

# ==================== 数据集 ====================
class SnowIceDataset(Dataset):
    def __init__(self, imgd, lbld, tf=None, size=(512, 512), cache=False):
        self.imgs = sorted(f for f in os.listdir(imgd) if f.lower().endswith(('png', 'jpg', 'jpeg')))
        self.lbls = sorted(f for f in os.listdir(lbld) if f.lower().endswith(('png', 'jpg', 'jpeg')))

        self.imgd, self.lbld, self.tf, self.size, self.cache = imgd, lbld, tf, size, cache

        img_count = len(self.imgs)
        lbl_count = len(self.lbls)
        min_len = min(img_count, lbl_count)

        # 输出日志
        print(f"[SnowIceDataset] Images found: {img_count}")
        print(f"[SnowIceDataset] Labels found: {lbl_count}")
        print(f"[SnowIceDataset] Using min count: {min_len}")

        if img_count > min_len:
            print(f"[SnowIceDataset] Dropping {img_count - min_len} extra images.")
        if lbl_count > min_len:
            print(f"[SnowIceDataset] Dropping {lbl_count - min_len} extra labels.")

        # 截断
        self.imgs = self.imgs[:min_len]
        self.lbls = self.lbls[:min_len]

        if cache:
            self.cache_data()

    def cache_data(self):
        self.ci, self.cl = [], []
        for im, lm in zip(self.imgs, self.lbls):
            img = Image.open(os.path.join(self.imgd, im)).convert('RGB').resize(self.size, Image.BILINEAR)
            lbl = Image.open(os.path.join(self.lbld, lm)).convert('L').resize(self.size, Image.NEAREST)
            self.ci.append(np.array(img, dtype=np.float32) / 255.)
            self.cl.append(np.clip(np.array(lbl, dtype=np.int64), 0, 2))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.cache:
            img, lbl = self.ci[idx], self.cl[idx]
        else:
            img = Image.open(os.path.join(self.imgd, self.imgs[idx])).convert('RGB').resize(self.size, Image.BILINEAR)
            lbl = Image.open(os.path.join(self.lbld, self.lbls[idx])).convert('L').resize(self.size, Image.NEAREST)
            img = np.array(img, dtype=np.float32) / 255.
            lbl = np.clip(np.array(lbl, dtype=np.int64), 0, 2)

        img = torch.from_numpy(img).permute(2, 0, 1) if not self.tf else self.tf(img)
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

# ==================== 训练函数 ====================
def train_model():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {"batch_size": 4, "num_workers": 4, "epochs": 2000, "lr": 1e-4, "weight_decay": 0.01,
           "mixed_precision": True, "cache_data": False, "gradient_clip": 1.0}
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_ds = SnowIceDataset("data/snow_ice_data/image/train", "data/snow_ice_data/label/train", tf, (512,512), cfg["cache_data"])
    # 随机挑选 1/2 样本
    total_indices = list(range(len(train_ds)))
    keep_n = max(1, len(total_indices) // 2)  # 至少保留 1 个样本
    train_indices = random.sample(total_indices, keep_n)
    train_ds = Subset(train_ds, train_indices)

    val_ds = SnowIceDataset("data/snow_ice_data/image/val", "data/snow_ice_data/label/val", tf, (512,512))
    persistent = cfg["num_workers"] > 0
    if cfg["num_workers"] > 0:
        train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True,
                              persistent_workers=True, drop_last=True)
    else:
        train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=True,
                              drop_last=True)

    if cfg["num_workers"] > 0:
        val_dl = DataLoader(val_ds, cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True,
                            persistent_workers=True)
    else:
        val_dl = DataLoader(val_ds, cfg["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=True)

    model = UNet(3, 3).to(dev)
    opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-8)
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 50, 2, 1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler() if cfg["mixed_precision"] else None
    aux_w, best = 0.3, -1
    log = open('IceSnowNet.txt','a',encoding='utf-8')
    log.write(f"{'='*40} 开始 {datetime.now()} {'='*40}\n")
    for ep in range(1, cfg["epochs"]+1):
        model.train(); run_loss=main_loss_sum=aux_loss_sum=0; count=0
        for i,(img,lbl) in enumerate(train_dl):
            img,lbl = img.to(dev), lbl.to(dev)
            opt.zero_grad(set_to_none=True)
            with autocast(cfg["mixed_precision"]):
                out,aux = model(img)
                loss_main,loss_aux = crit(out,lbl), crit(aux,lbl)
                loss = loss_main + aux_w * loss_aux
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
            run_loss += loss.item(); main_loss_sum += loss_main.item(); aux_loss_sum += loss_aux.item(); count += 1
            if (i+1) % 50 == 0:
                print(f"Ep{ep} It{i+1}/{len(train_dl)} Loss:{loss.item():.4f} Avg:{run_loss/count:.4f} "
                      f"Main:{main_loss_sum/count:.4f} Aux:{aux_loss_sum/count:.4f} LR:{sch.get_last_lr()[0]:.6f}")
        if ep % 5 == 0:
            model.eval(); vloss=0; pp,tt = [], []
            with torch.no_grad(), autocast(cfg["mixed_precision"]):
                for img,lbl in val_dl:
                    img,lbl = img.to(dev), lbl.to(dev)
                    o,_ = model(img)
                    vloss += crit(o,lbl).item()
                    pp.append(o.argmax(1).cpu()); tt.append(lbl.cpu())
            m = calculate_metrics(torch.cat(pp), torch.cat(tt))
            def log_msg(s): print(s); log.write(s + "\n")
            log_msg(f"\n========== Epoch {ep} Validation Results ==========")
            log_msg(f"1. 损失指标:\n   - 验证损失: {vloss/len(val_dl):.4f}")
            log_msg("\n2. IoU指标:")
            for n,v in zip(["背景","冰","雪"], m["IoU_per_class"]): log_msg(f"   - {n} IoU: {v:.4f}")
            log_msg(f"   - 平均 IoU: {m['mIoU']:.4f}")
            log_msg("\n3. Dice指标:")
            for n,v in zip(["背景","冰","雪"], m["Dice_per_class"]): log_msg(f"   - {n} Dice: {v:.4f}")
            log_msg(f"   - 平均 Dice: {m['mDice']:.4f}")
            log_msg(f"\n4. 分类综合指标:\n   - 准确率: {m['Accuracy']:.4f}\n   - 精确率: {m['Precision']:.4f}\n   - 召回率: {m['Recall']:.4f}\n   - F1分数: {m['Fscore']:.4f}")
            log_msg("\n5. 误检率 (FAR):")
            for n,v in zip(["背景","冰","雪"], m["FAR_per_class"]): log_msg(f"   - {n} FAR: {v:.4f}")
            log_msg(f"   - 平均 FAR: {m['mFAR']:.4f}")
            log_msg("\n6. 混淆矩阵:")
            log_msg("          预测: 背景   冰     雪")
            log_msg("    ───────────────────────────────")
            for i,row in enumerate(m["ConfusionMatrix"]):
                log_msg(f"真实 {['背景','冰','雪'][i]}:  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")
            if m['mIoU'] > best:
                best = m['mIoU']
                torch.save(model.state_dict(), 'best_icesnownet_model.pth')
                log_msg(f"✅  当前最佳模型已保存（mIoU: {best:.4f}）")
    log.write(f"{'='*40} 结束 {datetime.now()} {'='*40}\n"); log.close()

if __name__ == "__main__":
    train_model()
