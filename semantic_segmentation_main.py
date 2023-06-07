# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:27:14 2023

@author: ms024
"""

import os
import cv2
import json
import torch
import argparse
import numpy as np
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from semantic_segmentation_dataset import SemanticSegmentationDataset
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

def segformer_parser():
    parser = argparse.ArgumentParser(description='SegFormer test')
    parser.add_argument('--api', default='get_info', type=str,
                        help='choose web API type: train | eval | eval_others | infer')
    return parser.parse_args()

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def ade_palette():
    """LIDL new method palette that maps each class to RGB values."""
    return [[180, 0, 50], [0, 0, 255]]

def show_img(image, mask):
    color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[mask == label, :] = color
    color_seg = color_seg[..., ::-1]
    
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imshow("img", img)
    cv2.imwrite("./results/result.png", img)
    cv2.waitKey(0)

def infer(checkpoint="nvidia/mit-b4"):
    
    feature_extractor = SegformerFeatureExtractor(reduce_labels=False)
    image = Image.open("./data/image_1.png")
    encoding = feature_extractor(image, return_tensors="pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    id2label = json.load(open('id2label.json', "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint,
                                                              num_labels=num_labels,
                                                              id2label=id2label,
                                                              label2id=label2id)
    model.load_state_dict(torch.load("./models/LIDL_20230425_b1.pth"))
    model = model.cuda()
    model.eval()
    
    iou_thresh = 0.5
    with torch.no_grad():
        pixel_values = encoding.pixel_values.to(device)
        outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    
    upsampled_logits = nn.functional.interpolate(logits,
                    size=image.size[::-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)
    
    pred = upsampled_logits.squeeze(1)[0].detach().cpu().numpy()
    pred = normalization(pred)
    pred[pred>=iou_thresh] = 1
    pred[pred<iou_thresh] = 0
    pred[pred==0] = 255
    pred[pred==1] = 1
    
    show_img(image, pred)
    
def evaluate(checkpoint="nvidia/mit-b4"):
    img_path = "path to evaluate dataset"
    img_files = os.listdir(img_path)
    
    feature_extractor = SegformerFeatureExtractor(reduce_labels=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    id2label = json.load(open('id2label.json', "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, num_labels=num_labels, id2label=id2label, label2id=label2id)
    model.load_state_dict(torch.load("./models/LIDL_20230331_b0.pth"))
    model = model.cuda()
    model.eval()
    
    iou_avg = []
    iou_thresh = 0.5
    for img_file in img_files:
    
        image = Image.open(os.path.join(img_path, img_file))
        encoding = feature_extractor(image, return_tensors="pt")
            
        with torch.no_grad():
            pixel_values = encoding.pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()
        
        upsampled_logits = nn.functional.interpolate(logits,
                        size=image.size[::-1], # (height, width)
                        mode='bilinear',
                        align_corners=False)
        
        pred = upsampled_logits.squeeze(1)[0].detach().cpu().numpy()
        pred = normalization(pred)
        pred[pred>=iou_thresh] = 1
        pred[pred<iou_thresh] = 0
        
        # pred[pred==0] = 255
        # pred[pred==1] = 1
        # show_img(image, pred)

        target = Image.open(os.path.join(img_path.replace("\img", "\mask"), img_file)).convert('L')
        target = np.array(target)
        iou = iou_mean(pred, target, 1)
        iou_avg.append(iou)

    print(f"average MIoU(threshold={iou_thresh}) = {np.mean(iou_avg)}")

def evaluate_others():
    img_path = "path to evaluate dataset 1"
    img_files = os.listdir(img_path)
    
    iou_avg = []
    for img_file in img_files:
        pred = Image.open(os.path.join("path to evaluate dataset 2", img_file)).convert('L')
        pred = np.array(pred)

        target = Image.open(os.path.join(img_path.replace("\img", "\mask"), img_file)).convert('L')
        target = np.array(target)
        iou = iou_mean(pred, target, 1)
        iou_avg.append(iou)
        
        # image = Image.open(os.path.join(img_path, img_file))
        # show_img(image, pred)

    print(f"average MIoU = {np.mean(iou_avg)}")

def iou_mean(pred, target, class_num = 1):
    iou_sum = 0
    
    pred = torch.from_numpy(pred).view(-1)
    target = np.array(target)
    target = torch.from_numpy(target).view(-1)
    
    for cls_ in range(0, class_num+1):
        # print(f"cls_ = {cls_}")
        pred_idx = pred == cls_
        target_idx = target == cls_
        
        intersection = (pred_idx[target_idx]).long().sum().data.cpu().item()
        union = pred_idx.long().sum().data.cpu().item() + target_idx.long().sum().data.cpu().item() - intersection
        
        if union != 0:
            iou_sum += float(intersection) / float(max(union, 1))
            # print(f"iou = {float(intersection) / float(max(union, 1))}")
            
    return iou_sum / (class_num+1)

def train(checkpoint="nvidia/mit-b4"):
    root_dir = r"path to train dataset"
    feature_extractor = SegformerFeatureExtractor(reduce_labels=False)
    
    train_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor)
    # valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, train=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=4)
    
    id2label = json.load(open('id2label.json', "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint,
                                                              num_labels=num_labels,
                                                              id2label=id2label,
                                                              label2id=label2id,
                                                              semantic_loss_ignore_index = 255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000005)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    for epoch in range(200):
        total_loss_train = []
        total_miou = []
        baseline_acc = 0.0
        
        for idx, batch in enumerate(train_dataloader):
              pixel_values = batch["pixel_values"].to(device)
              labels = batch["labels"].to(device)
              
              optimizer.zero_grad()
        
              outputs = model(pixel_values=pixel_values, labels=labels)
              loss, logits = outputs.loss, outputs.logits
             
              loss.backward()
              optimizer.step()
              
              total_loss_train.append(loss.item())
              
              with torch.no_grad():
                  predicted  = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                  predicted = normalization(predicted.squeeze(1).detach().cpu().numpy())
                  predicted[predicted<0.5] = 0
                  predicted[predicted>=0.5] = 1
                  
                  iou = iou_mean(predicted, labels.detach().cpu().numpy(), class_num = 1)
                  total_miou.append(iou)
        
        if (epoch + 1) % 1 == 0:
            if np.mean(total_miou) > baseline_acc:
                torch.save(model.state_dict(), os.path.join("models", "LIDL_20230331.pth"))
                baseline_acc = np.mean(total_miou)
        
        print(f'Epochs: {epoch + 1}, Total Train Loss: {np.mean(total_loss_train)}, Total MIoU: {np.mean(total_miou)}')


if __name__ == "__main__":
    args = segformer_parser()
    
    if args.api == "train":
        train("nvidia/mit-b0")
    
    elif args.api == "eval":
        evaluate("nvidia/mit-b0")
    
    elif args.api == "eval_others":
        evaluate_others()
    
    elif args.api == "infer":
        infer("nvidia/mit-b1")
    