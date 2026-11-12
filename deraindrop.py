import os, random, time, argparse

from PIL import Image
import numpy as np

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import GradScaler, autocast
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torchvision import transforms
import lpips
from tqdm import tqdm

from model.MHFNet import MHFNet

pid = os.getpid()
print(f"The process is now running, PID: {pid}")

def seed_torch(seed=1021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure(d):
    os.makedirs(d, exist_ok=True)
    return d

class RainDSTrainDataset(Dataset):
    def __init__(self, root_dir, crop_size):
        self.root_dir = root_dir
        self.rain_dir = os.path.join(self.root_dir, 'train_set/raindrop')
        self.clear_dir = os.path.join(self.root_dir, 'train_set/gt')
        self.clear_filenames = sorted(
            [img for img in os.listdir(self.clear_dir) if
             img.endswith('.png') or img.endswith('.jpg') or img.endswith('.PNG') or img.endswith('.JPG')])
        self.rain_filenames = sorted(
            [img for img in os.listdir(self.rain_dir) if
             img.endswith('.png') or img.endswith('.jpg') or img.endswith('.PNG') or img.endswith('.JPG')])
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.clear_filenames)

    def __getitem__(self, idx):
        rain_img_name = os.path.join(self.rain_dir, self.rain_filenames[idx])
        clear_img_name = os.path.join(self.clear_dir, self.clear_filenames[idx])
        rain_img = Image.open(rain_img_name).convert("RGB")
        clear_img = Image.open(clear_img_name).convert("RGB")
        if rain_img.size != clear_img.size:
            print(self.rain_filenames[idx] + ':' + self.clear_filenames[idx])
            raise ValueError("The size of the input image pairs is inconsistent")
        if rain_img.size[0] < self.crop_size or rain_img.size[1] < self.crop_size:
            rain_img = rain_img.resize((self.crop_size, self.crop_size))
            clear_img = clear_img.resize((self.crop_size, self.crop_size))
        else:
            start_x = random.randint(0, rain_img.size[0] - self.crop_size)
            start_y = random.randint(0, rain_img.size[1] - self.crop_size)
            clear_img = clear_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))
            rain_img = rain_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))
        random_rotate = random.choice([0, 90, 180, 270])
        clear_img = clear_img.rotate(random_rotate)
        rain_img = rain_img.rotate(random_rotate)
        clear_img = self.transform(clear_img)
        rain_img = self.transform(rain_img)
        sample = {'raindrop_img': rain_img, 'clear_img': clear_img}

        return sample


class RainDSTestDataset(Dataset):
    def __init__(self, root_dir, crop_size):
        self.root_dir = root_dir
        self.rain_dir = os.path.join(self.root_dir, 'test_set/raindrop')
        self.clear_dir = os.path.join(self.root_dir, 'test_set/gt')
        self.clear_filenames = sorted(
            [img for img in os.listdir(self.clear_dir) if
             img.endswith('.png') or img.endswith('.jpg') or img.endswith('.PNG') or img.endswith('.JPG')])
        self.rain_filenames = sorted(
            [img for img in os.listdir(self.rain_dir) if
             img.endswith('.png') or img.endswith('.jpg') or img.endswith('.PNG') or img.endswith('.JPG')])
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.clear_filenames)

    def __getitem__(self, idx):
        rain_img_name = os.path.join(self.rain_dir, self.rain_filenames[idx])
        clear_img_name = os.path.join(self.clear_dir, self.clear_filenames[idx])
        rain_img = Image.open(rain_img_name).convert("RGB")
        clear_img = Image.open(clear_img_name).convert("RGB")
        if rain_img.size != clear_img.size:
            print(self.rain_filenames[idx] + ':' + self.clear_filenames[idx])
            raise ValueError("The size of the input image pairs is inconsistent")
        if rain_img.size[0] < self.crop_size or rain_img.size[1] < self.crop_size:
            rain_img = rain_img.resize((self.crop_size, self.crop_size))
            clear_img = clear_img.resize((self.crop_size, self.crop_size))
        else:
            start_x = random.randint(0, rain_img.size[0] - self.crop_size)
            start_y = random.randint(0, rain_img.size[1] - self.crop_size)
            clear_img = clear_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))
            rain_img = rain_img.crop((start_x, start_y, start_x + self.crop_size, start_y + self.crop_size))
        random_rotate = random.choice([0, 90, 180, 270])
        clear_img = clear_img.rotate(random_rotate)
        rain_img = rain_img.rotate(random_rotate)
        clear_img = self.transform(clear_img)
        rain_img = self.transform(rain_img)
        sample = {'raindrop_img': rain_img, 'clear_img': clear_img}

        return sample


def freq_loss(res, clear, criterion):
    clear_fft = torch.fft.fft2(clear, dim=(-2, -1))
    clear_fft = torch.stack((clear_fft.real, clear_fft.imag), -1)
    pred_fft = torch.fft.fft2(res, dim=(-2, -1))
    pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
    loss = criterion(pred_fft, clear_fft)
    return loss

def cal_loss(res, clear_img, criterion):
    loss1 = criterion(res, clear_img)
    loss2 = freq_loss(res, clear_img, criterion)

    return loss1 + loss2

def train(args):
    ensure(args.save_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_dataset = RainDSTrainDataset(args.data_root, crop_size=args.crop_size)
    val_dataset = RainDSTestDataset(args.data_root, crop_size=args.crop_size)

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = MHFNet(type=args.type, target_size=args.crop_size).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    loss_fn = nn.L1Loss()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    best_val_psnr = 0.0
    best_val_ssim = 0.0
    start_epoch = 1

    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_psnr = ckpt.get('val_psnr', 0.0)
        best_val_ssim = ckpt.get('val_ssim', 0.0)
        print(f"Resuming from epoch {start_epoch - 1}, best PSNR: {best_val_psnr:.2f}, best SSIM: {best_val_ssim:.4f}")

    total_steps = args.epoch * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    if args.resume:
        steps_done = (start_epoch - 1) * len(train_loader)
        for _ in range(steps_done):
            sched.step()

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(start_epoch, args.epoch + 1):
        model.train()
        train_running_loss = 0.0
        train_pbar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}/{args.epoch} [Train]')
        for batch in train_pbar:
            rain = batch['raindrop_img'].to(device)
            clear = batch['clear_img'].to(device)

            opt.zero_grad()
            if args.amp and scaler is not None:
                with autocast(device_type=args.device):
                    pred = model(rain)
                    loss = cal_loss(pred, clear, loss_fn)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = (scale != scaler.get_scale())
                if not skip_lr_sched:
                    sched.step()
            else:
                pred = model(rain)
                loss = cal_loss(pred, clear, loss_fn)
                loss.backward()
                opt.step()
                sched.step()
            train_running_loss += loss.item()
            # train_pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})

        avg_train_loss = train_running_loss / len(train_loader)

        model.eval()
        psnr.reset()
        ssim.reset()
        val_running_loss = 0.0
        val_pbar = tqdm(val_loader, total=len(val_loader), desc=f'Epoch {epoch}/{args.epoch} [Val]',
                        leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                rain = batch['raindrop_img'].to(device)
                clear = batch['clear_img'].to(device)
                pred = model(rain)
                loss = cal_loss(pred, clear, loss_fn)

                pred_denorm = torch.clamp(pred, 0, 1)
                clear_denorm = torch.clamp(clear, 0, 1)

                val_running_loss += loss.item()

                psnr.update(pred_denorm, clear_denorm)
                ssim.update(pred_denorm, clear_denorm)
                # val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_psnr = psnr.compute().item()
        avg_val_ssim = ssim.compute().item()

        print(
            f'Epoch {epoch} | Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val PSNR={avg_val_psnr:.2f}dB | Val SSIM={avg_val_ssim:.4f} | lr={sched.get_last_lr()[0]:.2e}')

        if avg_val_psnr >= best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
            }, os.path.join(args.save_dir, 'psnr_best.pth'))
            print(f'  -> New best validation PSNR achieved, model saved.')

        if avg_val_ssim >= best_val_ssim:
            best_val_ssim = avg_val_ssim
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
            }, os.path.join(args.save_dir, 'ssim_best.pth'))
            print(f'  -> New best validation SSIM achieved, model saved.')

    print(f'Rain removal training completed. Best validation PSNR: {best_val_psnr:.2f}dB, Best validation SSIM: {best_val_ssim:.4f} Optimal PSNR weight: ',
          os.path.join(args.save_dir, 'psnr_best.pth'), 'Optimal SSIM weight: ', os.path.join(args.save_dir, 'ssim_best.pth'))

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    val_dataset = RainDSTestDataset(args.data_root, crop_size=args.crop_size)

    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = MHFNet(type=args.type, target_size=args.crop_size).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    lpips_loss = lpips.LPIPS(net='alex').to(device).eval()
    lpips_vals = []

    total_time = 0.0
    total_images = 0

    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            rain = batch['raindrop_img'].to(device)
            clear = batch['clear_img'].to(device)
            start_time = time.time()
            pred = model(rain)
            pred = pred[-1]
            end_time = time.time()
            total_time += (end_time - start_time)
            total_images += rain.size(0)

            pred_denorm = torch.clamp(pred, 0, 1)
            clear_denorm = torch.clamp(clear, 0, 1)

            psnr.update(pred_denorm, clear_denorm)
            ssim.update(pred_denorm, clear_denorm)
            ms_ssim.update(pred_denorm, clear_denorm)

            pred_lpips = pred_denorm * 2.0 - 1.0
            clear_lpips = clear_denorm * 2.0 - 1.0
            lpips_val = lpips_loss(pred_lpips, clear_lpips)
            lpips_vals.append(lpips_val.mean().item())

    avg_val_psnr = psnr.compute().item()
    avg_val_ssim = ssim.compute().item()
    avg_val_ms_ssim = ms_ssim.compute().item()
    avg_val_lpips = np.mean(lpips_vals) if lpips_vals else float('nan')

    avg_time_per_image = total_time / total_images
    fps = total_images / total_time if total_time > 0 else float('inf')

    print(f'Val PSNR={avg_val_psnr:.2f}dB | Val SSIM={avg_val_ssim:.4f} | Val MS-SSIM={avg_val_ms_ssim:.4f} | Val LPIPS={avg_val_lpips:.4f}')
    print(f'Average inference time per image: {avg_time_per_image:.2f} s')
    print(f'FPS: {fps:.2f}')


# ---------- main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False, help='Train')
    parser.add_argument('--evaluate', type=bool, default=True, help='Evaluate')
    parser.add_argument('--amp', type=bool, default=False, help='amp')
    parser.add_argument('--type', type=str, default='S')
    parser.add_argument('--data_root', default='', help='dataset root')
    parser.add_argument('--weights', default='', help='weights')
    parser.add_argument('--crop_size', type=int, default=1024, help='crop size')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='lr')
    parser.add_argument('--seed', type=int, default=1021, help='seed')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint patch')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    seed_torch(args.seed)

    if args.train:
        train(args)
    elif args.evaluate:
        evaluate(args)
    else:
        parser.print_help()
