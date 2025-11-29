import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Hyperparameters/Options
batch_size = 32
num_epochs = 10
learning_rate = .0001
dropout = 0.2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def EffNetB0(pretrained=True):
    # If pretrained is true, use ImageNet for weights, else use random weights
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, 2)  # binary classification: toxic vs. non-toxic
    )
    
    model = model.to(device)
    return model

# Channel attention module
class CAM(nn.Module):
  def __init__(self, channels, r):
    super(CAM, self).__init__()
    self.mlp = nn.Sequential(
        nn.Linear(channels, channels // r, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channels // r, channels, bias=False)
    )
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    b, c, h, w = x.size()
    avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
    max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
    attn = self.mlp(avg_pool) + self.mlp(max_pool)
    attn = self.sigmoid(attn).view(b, c, 1, 1)
    return attn * x

# Spatial attention module
class SAM(nn.Module):
  def __init__(self, kernel_size=7, bias=False):
    super(SAM, self).__init__()
    self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=bias)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    x_cat = torch.cat([avg_out, max_out], dim=1)
    x_out = self.conv(x_cat)
    attn = self.sigmoid(x_out)
    return attn * x

# CBAM
class CBAM(nn.Module):
  def __init__(self, channels, r=16):
    super(CBAM, self).__init__()
    self.channels = channels
    self.r = r
    self.cam = CAM(channels, r)
    self.sam = SAM(bias=False)

  def forward(self, x):
    output = self.cam(x)
    output = self.sam(output)
    return output
  
  # Implement CBAM before last classifier in EfficientNetB0
class EfficientNetB0_CBAM(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.2):
        super(EfficientNetB0_CBAM, self).__init__()
        if pretrained:
            base_model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
        else:
            base_model = models.efficientnet_b0(weights=None)
        
        self.features = base_model.features
        self.cbam = CBAM(channels=1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features=1280, out_features=num_classes)
        
    def forward(self, x):
      x = self.features(x)
      x = self.cbam(x)
      x = self.pool(x).flatten(1)
      x = self.dropout(x)
      x = self.classifier(x)
      return x
    
    # Implement CBAM before last classifier in EfficientNetB2
class EfficientNetB2_CBAM(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.2):
        super(EfficientNetB2_CBAM, self).__init__()
        if pretrained:
            base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            base_model = models.efficientnet_b2(weights=None)
        
        self.features = base_model.features
        self.cbam = CBAM(channels=1408)  # B2 has 1408 channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features=1408, out_features=num_classes)
        
    def forward(self, x):
      x = self.features(x)
      x = self.cbam(x)
      x = self.pool(x).flatten(1)
      x = self.dropout(x)
      x = self.classifier(x)
      return x
    
def EffNetB2_CBAM(pretrained=True):
    """Wrapper for compatibility with existing code"""
    model = EfficientNetB2_CBAM(num_classes=2, pretrained=pretrained)
    model = model.to(device)
    return model