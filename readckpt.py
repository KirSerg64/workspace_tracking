import torch

ckpt = torch.load("output/checkpoints/train/ostrack/vitb_learnrect/OSTrack_ep0040.pth.tar", map_location="cpu")
print(ckpt['net']['preprocess.transparency'])
print(ckpt['net']['preprocess.color'])
ckpt = torch.load("output/checkpoints/train/ostrack/vitb_learnrect/OSTrack_ep0080.pth.tar", map_location="cpu")
print(ckpt['net']['preprocess.transparency'])
print(ckpt['net']['preprocess.color'])
ckpt = torch.load("output/checkpoints/train/ostrack/vitb_learnrect/OSTrack_ep0120.pth.tar", map_location="cpu")
print(ckpt['net']['preprocess.transparency'])
print(ckpt['net']['preprocess.color'])
ckpt = torch.load("output/checkpoints/train/ostrack/vitb_learnrect/OSTrack_ep0160.pth.tar", map_location="cpu")
print(ckpt['net']['preprocess.transparency'])
print(ckpt['net']['preprocess.color'])
ckpt = torch.load("output/checkpoints/train/ostrack/vitb_learnrect/OSTrack_ep0200.pth.tar", map_location="cpu")
print(ckpt['net']['preprocess.transparency'])
print(ckpt['net']['preprocess.color'])
ckpt = torch.load("output/checkpoints/train/ostrack/vitb_learnrect/OSTrack_ep0300.pth.tar", map_location="cpu")
print(ckpt['net']['preprocess.transparency'])
print(ckpt['net']['preprocess.color'])