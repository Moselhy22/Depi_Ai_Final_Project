#!/usr/bin/env python3
"""
VD-MIL Violence Detection Wrapper - CORRECT VERSION
"""
import os, sys
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

VDMIL_PATH = "/home/moselhy/Desktop/Depi_Ai_Final_Project/violence-detection-mil"
sys.path.insert(0, VDMIL_PATH)

from movinets import MoViNet
from movinets.config import _C

class VDMILViolenceDetector:
    def __init__(self, classifier_path, backbone_weights_dir=None, device='cuda',
                 clip_length=8, frame_size=(172, 172), stride=4):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.stride = stride
        self.frame_buffer = []
        print(f"[VD-MIL] Loading model on {self.device}...")
        
        self.model_movinet = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=True,
                                      model_dir=backbone_weights_dir or './movinet_weights')
        self.model_movinet.classifier[3] = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        self.model_movinet.to(self.device)
        self.model_movinet.eval()
        
        self.classifiers = []
        checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
        
        if hasattr(checkpoint, 'forward'):
            checkpoint.eval()
            self.classifiers.append(checkpoint.to(self.device))
            print(f"[VD-MIL] Loaded Net classifier")
        else:
            from movinet_classifier import Net
            net = Net()
            net.load_state_dict(checkpoint)
            net.eval()
            self.classifiers.append(net.to(self.device))
            print(f"[VD-MIL] Loaded classifier from state_dict")
        
        print("[VD-MIL] Model loaded successfully!")
    
    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (172, 172), cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb
    
    def update_buffer(self, frame):
        self.frame_buffer.append(self.preprocess_frame(frame))
        if len(self.frame_buffer) > self.clip_length:
            self.frame_buffer = self.frame_buffer[-self.clip_length:]
    
    def predict(self, frames=None):
        if frames is not None:
            self.frame_buffer = [self.preprocess_frame(f) for f in frames]
        if len(self.frame_buffer) < self.clip_length:
            return 0.0
        
        clip_frames = self.frame_buffer[-self.clip_length:]
        list_tensor = [torch.from_numpy(img)[None] for img in clip_frames]
        tensor_clip = torch.cat(list_tensor, axis=0)
        tensor_clip = tensor_clip.to(self.device)
        tensor_clip = tensor_clip.permute(3, 0, 1, 2)
        tensor_clip = tensor_clip[None]
        tensor_clip = tensor_clip.to(torch.float32)
        tensor_clip = tensor_clip / 255.0
        
        with torch.no_grad():
            self.model_movinet.clean_activation_buffers()
            xf = self.model_movinet(tensor_clip)
            norm_xf = F.normalize(xf, dim=1, p=2)
            scores = []
            for net in self.classifiers:
                y = net(norm_xf)
                scores.append(y.to('cpu').numpy())
            scores = np.concatenate(scores, axis=1)
            prob = float(scores[0][0])
        
        return prob
    
    def predict_clip(self, clip_frames):
        return self.predict(clip_frames)
    
    def reset_buffer(self):
        self.frame_buffer = []

class VDMILFrameProcessor:
    def __init__(self, detector, stride=4):
        self.detector = detector
        self.stride = stride
        self.frame_count = 0
        self.last_prediction = 0.0
    
    def process_frame(self, frame):
        self.detector.update_buffer(frame)
        self.frame_count += 1
        if self.frame_count % self.stride == 0 and len(self.detector.frame_buffer) >= self.detector.clip_length:
            self.last_prediction = self.detector.predict()
        return self.last_prediction
    
    def reset(self):
        self.detector.reset_buffer()
        self.frame_count = 0
        self.last_prediction = 0.0

def create_vdmil_detector(classifier_path=None, backbone_weights_dir=None, device='cuda'):
    if classifier_path is None:
        classifier_path = "/home/moselhy/Desktop/Depi_Ai_Final_Project/violence-detection-mil/models/checkpoints/violence/model_best.pt"
    if backbone_weights_dir is None:
        backbone_weights_dir = "/home/moselhy/Desktop/Depi_Ai_Final_Project/violence-detection-mil/movinet_weights"
    detector = VDMILViolenceDetector(classifier_path, backbone_weights_dir, device)
    return VDMILFrameProcessor(detector)
