import torch
import os
import cv2
import pdb

def make_train_step(model, loss_fn, optimizer):
    def train_step(images_batch, masks_batch):
        model.train()
        yhat = model(images_batch)
        loss = loss_fn(masks_batch, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

def evaluate(model, image):
    model.eval()
    with torch.no_grad():
        return model(image)

def save_predictions(preds, path, rounded_save, k):
    for pred in preds:
        if rounded_save:
            pred = torch.round(pred)
        pred *= 255
        img_path = os.path.join(path, f"image_{k}.jpg")
        k += 1
        cv2.imwrite(img_path, pred.cpu().numpy())
    return k