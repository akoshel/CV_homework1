"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import NUM_PTS, CROP_SIZE, CropFrame, FlipHorizontal, Rotator, CropRectangle, ChangeBrightnessContrast
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch, create_submission
from model import RESNEXT_steroid

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def weighted_mse_loss(preds, ground_true, weights):
    return torch.mean(weights * torch.mean((preds - ground_true) ** 2, axis=1))

def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    val_mse_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean") #, reduction="mean"
        val_loss.append(loss.item())
        weights_mse = (1 / batch['scale_coef']) ** 2
        mse_loss = weighted_mse_loss(pred_landmarks,
                                     landmarks,
                                     weights_mse)
        val_mse_loss.append(mse_loss.item())

    return (np.mean(val_loss), np.mean(val_mse_loss))


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    os.makedirs("runs", exist_ok=True)

    # 1. prepare data & models
    # train_transforms = transforms.Compose([
    #     ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
    #     CropCenter(CROP_SIZE),
    #     TransformByKeys(transforms.ToPILImage(), ("image",)),
    #     TransformByKeys(transforms.ToTensor(), ("image",)),
    #     TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)), # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # ])

    crop_size = (224, 224)
    train_transforms = transforms.Compose([
        CropFrame(9),
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        FlipHorizontal(),
        Rotator(30),
        # CropRectangle(crop_size),
        ChangeBrightnessContrast(alpha_std=0.05, beta_std=10),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                        ("image",)
                        ),
    ])

    valid_transforms = transforms.Compose([
        CropFrame(9),
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        # CropRectangle(crop_size),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                        ("image",)
                        ),
    ])
    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                  shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), valid_transforms, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                shuffle=False, drop_last=False)

    device = torch.device("cuda:0") if args.gpu and torch.cuda.is_available() else torch.device("cpu")

    print("Creating model...")
    # model = models.resnext50_32x4d(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
    # checkpoint = torch.load("./runs/baseline_full3_best.pth", map_location='cpu')
    # model.load_state_dict(checkpoint, strict=True)
    model = RESNEXT_steroid()
    model.to(device)
    for p in model.base_net.parameters():
        p.requires_grad = False
    # model.base_net[8].requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    for p in model.linear7.parameters():
        p.requires_grad = True
    for p in model.attention.parameters():
        p.requires_grad = True
    for p in model.linear1.parameters():
        p.requires_grad = True
    # model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    # criterion = AdaptiveWingLoss()
    # criterion = torch.nn.MSELoss(size_average=True)
    # loss_fn = fnn.mse_loss
    criterion = fnn.l1_loss
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=1/np.sqrt(10),
        patience=4,
        verbose=True, threshold=0.01,
        threshold_mode='abs', cooldown=0,
        min_lr=1e-6, eps=1e-08
    )

    # 2. train & validate
    print("Ready for training...")
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device=device)
        val_loss, mse_loss = validate(model, val_dataloader, criterion, device=device)
        lr_scheduler.step(val_loss)
        print("Epoch #{:2}:\ttrain loss: {:5.2}\tval loss: {:5.2}\tmse loss: {:5.2}".format(
            epoch, train_loss, val_loss, mse_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)

    # 3. predict
    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), train_transforms, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                 shuffle=False, drop_last=False)

    with open(os.path.join("runs", f"{args.name}_best.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join("runs", f"{args.name}_test_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, os.path.join("runs", f"{args.name}_submit.csv"))


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))