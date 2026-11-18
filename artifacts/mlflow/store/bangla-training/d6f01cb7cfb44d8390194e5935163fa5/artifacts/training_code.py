class CharDataset(Dataset):
    def __init__(self, imgs, labels, is_train=True):
        self.imgs = imgs
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].copy()
        if self.is_train:
            if random.random() < 0.7:
                angle = random.uniform(-10.0, 10.0)
                scale = random.uniform(0.90, 1.10)
                M = cv2.getRotationMatrix2D((32, 32), angle, scale)
                M[0, 2] += random.uniform(-3, 3)
                M[1, 2] += random.uniform(-3, 3)
                img = cv2.warpAffine(img, M, (64, 64), borderValue=0)
            if random.random() < 0.4:
                k = np.ones((2, 2), np.uint8)
                img = cv2.dilate(img, k) if random.random() < 0.5 else cv2.erode(img, k)
            if random.random() < 0.3:
                eh, ew = random.randint(4, 10), random.randint(4, 10)
                ey, ex = random.randint(0, 52), random.randint(0, 52)
                img[ey:ey+eh, ex:ex+ew] = 0
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return tensor, int(self.labels[idx]), idx

class ModelEMA:
    """Step-Aware Exponential Moving Average (ramps decay from 0.0 -> 0.999 so early epochs & short runs converge immediately)."""
    def __init__(self, model, decay=0.999):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=84):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // reduction), channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)
        self.short = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.short = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + self.short(x), inplace=True)

class LekhaResNet(nn.Module):
    def __init__(self, num_classes=85):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResBlock(32, 32, 1), ResBlock(32, 32, 1))
        self.layer2 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64, 1))
        self.layer3 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128, 1))
        self.layer4 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.pool(x).flatten(1))

class LekhaMobile(nn.Module):
    def __init__(self, num_classes=85):
        super().__init__()
        base = tvm.mobilenet_v3_small(weights=None)
        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(1, old_conv.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        in_f = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_f, num_classes)
        self.net = base

    def forward(self, x):
        return self.net(x)

def apply_cutmix_mixup(x, y, num_classes=85, alpha=0.8):
    bs = x.size(0)
    indices = torch.randperm(bs, device=x.device)
    y_a = F.one_hot(y, num_classes).float()
    y_b = F.one_hot(y[indices], num_classes).float()
    lam = np.random.beta(alpha, alpha)
    if random.random() < 0.5:
        x_mix = lam * x + (1.0 - lam) * x[indices]
    else:
        _, _, H, W = x.shape
        cut_rat = math.sqrt(1.0 - lam)
        cw, ch = int(W * cut_rat), int(H * cut_rat)
        cx, cy = random.randint(0, W), random.randint(0, H)
        x1, y1 = np.clip(cx - cw // 2, 0, W), np.clip(cy - ch // 2, 0, H)
        x2, y2 = np.clip(cx + cw // 2, 0, W), np.clip(cy + ch // 2, 0, H)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[indices, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * H))
    return x_mix, lam * y_a + (1.0 - lam) * y_b

@torch.no_grad()
def evaluate_char_model(model, loader, num_classes=85):
    model.eval()
    preds, targets = [], []
    for x, y, _ in loader:
        logits = model(x.to(DEVICE))[:, :num_classes]
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(y.numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)
    return accuracy_score(targets, preds), f1_score(targets, preds, average="macro", zero_division=0), preds, targets

def train_character_stage(stage_name, model, train_ds, val_ds, epochs, num_classes=85, use_mix=False, hard_mining_epoch=None, teacher=None):
    final_path = WORK_PATH / f"{stage_name}.pt"
    resume_ckpt = CKPT_DIR / f"{stage_name}_last.pt"

    if final_path.exists():
        print(f"[{stage_name}] Loaded existing checkpoint: {final_path}")
        model.load_state_dict(torch.load(final_path, map_location=DEVICE, weights_only=False))
        return model.to(DEVICE)

    ext_final = Path(CFG["RESUME_DIR"]) / f"{stage_name}.pt"
    if ext_final.exists():
        print(f"[{stage_name}] Restored from mounted input: {ext_final}")
        model.load_state_dict(torch.load(ext_final, map_location=DEVICE, weights_only=False))
        torch.save(model.state_dict(), final_path)
        return model.to(DEVICE)

    model = model.to(DEVICE)
    ema = ModelEMA(model, decay=CFG["EMA_DECAY"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    start_epoch, best_f1 = 0, -1.0
    sample_losses = np.ones(len(train_ds), dtype=np.float32)

    if resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch, best_f1 = ckpt["epoch"] + 1, ckpt["best_f1"]
        print(f"[{stage_name}] Resuming from epoch {start_epoch} (Best Val F1={best_f1:.4f})")

    val_loader = DataLoader(val_ds, batch_size=CFG["BATCH_SIZE"] * 2, shuffle=False, num_workers=CFG["NUM_WORKERS"])
    ce_smooth = nn.CrossEntropyLoss(label_smoothing=CFG["LABEL_SMOOTH"], reduction="none")

    for ep in range(start_epoch, epochs):
        if hard_mining_epoch is not None and ep >= hard_mining_epoch:
            thresh = np.percentile(sample_losses, 90)
            weights = np.where(sample_losses >= thresh, 2.0, 1.0)
            sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=CFG["BATCH_SIZE"], sampler=sampler, num_workers=CFG["NUM_WORKERS"])
        else:
            train_loader = DataLoader(train_ds, batch_size=CFG["BATCH_SIZE"], shuffle=True, num_workers=CFG["NUM_WORKERS"])

        warmup_ep = max(1, int(epochs * 0.05))
        lr = CFG["LR"] * float(ep + 1) / warmup_ep if ep < warmup_ep else CFG["LR"] * 0.5 * (1.0 + math.cos(math.pi * float(ep - warmup_ep) / max(1, epochs - warmup_ep)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        ep_loss = 0.0
        do_mix = use_mix and (ep < int(epochs * 0.90))

        for x, y, idxs in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                if do_mix and random.random() < 0.5:
                    x_in, y_soft = apply_cutmix_mixup(x, y, num_classes=num_classes)
                    logits = model(x_in)
                    per_sample = -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1)
                else:
                    logits = model(x)
                    per_sample = ce_smooth(logits, y)

                loss = per_sample.mean()
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    T = 4.0
                    kd_loss = F.kl_div(F.log_softmax(logits / T, dim=1), F.softmax(t_logits / T, dim=1), reduction="batchmean") * (T * T)
                    loss = 0.5 * loss + 0.5 * kd_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            ep_loss += loss.item() * x.size(0)
            sample_losses[idxs.numpy()] = per_sample.detach().cpu().numpy()

        val_acc, val_f1, _, _ = evaluate_char_model(ema.ema, val_loader, num_classes=num_classes)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(ema.ema.state_dict(), final_path)

        torch.save({"epoch": ep, "model": model.state_dict(), "ema": ema.ema.state_dict(), "opt": optimizer.state_dict(), "best_f1": best_f1}, resume_ckpt)
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"[{stage_name}] Ep {ep+1:02d}/{epochs} | LR={lr:.5f} | Loss={ep_loss/len(train_ds):.4f} | Val Acc={val_acc*100:.2f}% | Val F1={val_f1*100:.2f}%")

    model.load_state_dict(torch.load(final_path, map_location=DEVICE, weights_only=False))
    return model

# Section 3: Baseline CNN on Raw V0
v0_tr_mask, v0_va_mask = v0_data["splits"] == "train", v0_data["splits"] == "val"
ds_base_tr = CharDataset(v0_data["imgs"][v0_tr_mask], v0_data["labels"][v0_tr_mask], is_train=False)
ds_base_va = CharDataset(v0_data["imgs"][v0_va_mask], v0_data["labels"][v0_va_mask], is_train=False)
baseline_model = train_character_stage("baseline", BaselineCNN(84), ds_base_tr, ds_base_va, epochs=CFG["EPOCHS"]["baseline"], num_classes=84)

# Section 4: Model 1 — Lekha-ResNet on Engineered V1–V7
ds_eng_tr = CharDataset(eng_data["train_X"], eng_data["train_y"], is_train=True)
ds_eng_va = CharDataset(eng_data["val_X"], eng_data["val_y"], is_train=False)
model1_resnet = train_character_stage("model1", LekhaResNet(85), ds_eng_tr, ds_eng_va, epochs=CFG["EPOCHS"]["model1"], num_classes=85, use_mix=True, hard_mining_epoch=int(CFG["EPOCHS"]["model1"] * 0.66))
model1_resnet.eval()

# Section 5: Model 2 — Lekha-Mobile (Distilled from Model 1)
model2_mobile = train_character_stage("model2", LekhaMobile(85), ds_eng_tr, ds_eng_va, epochs=CFG["EPOCHS"]["model2"], num_classes=85, use_mix=False, teacher=model1_resnet)
model2_mobile.eval()