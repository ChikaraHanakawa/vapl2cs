import torch
import torch.nn as nn
import os
import math
from pathlib import Path

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class L2CSEncoder(nn.Module):
    def __init__(self, load_pretrained=True, pretrained_path=None, freeze=False):
        """
        L2CS Encoder based on ResNet50 architecture for extracting visual features
        
        Args:
            load_pretrained: Whether to load pretrained weights
            pretrained_path: Path to pretrained weights file
            freeze: Whether to freeze the encoder parameters
        """
        super(L2CSEncoder, self).__init__()
        # Define the L2CS-based encoder (ResNet backbone)
        self.inplanes = 64
        block = Bottleneck
        layers = [3, 4, 6, 3]  # ResNet50
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # Output dimension
        self.dim = 512 * block.expansion  # 2048 for ResNet50
        
        # Load pretrained weights if specified
        if load_pretrained:
            if pretrained_path is None:
                # Use default path if not specified
                pretrained_path = str(Path.home() / "models" / "l2cs_resnet50.pth")
            
            if os.path.exists(pretrained_path):
                print(f"Loading L2CS pretrained weights from {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location='cpu')
                
                # Filter out classifier weights, keep only encoder weights
                encoder_state_dict = {}
                for key, value in state_dict.items():
                    if not key.startswith('fc_'):
                        encoder_state_dict[key] = value
                
                # Load weights
                self.load_state_dict(encoder_state_dict, strict=False)
                print("L2CS encoder weights loaded successfully")
            else:
                print(f"Warning: Pretrained weights file {pretrained_path} not found")
        
        # Freeze encoder if specified
        if freeze:
            print("Freezing L2CS encoder")
            for param in self.parameters():
                param.requires_grad = False
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Input shape: [B, C, H, W]
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # [B, dim]
        
        return x
