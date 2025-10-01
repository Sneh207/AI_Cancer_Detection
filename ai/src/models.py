import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CustomCNN(nn.Module):
    """
    Custom CNN architecture for chest X-ray cancer detection
    """
    def __init__(self, num_classes=1, dropout=0.5):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class ResNetModel(nn.Module):
    """
    ResNet-based model with transfer learning
    """
    def __init__(self, num_classes=1, pretrained=True, dropout=0.5, architecture='resnet50'):
        super(ResNetModel, self).__init__()
        
        # Load pretrained ResNet
        if architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif architecture == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = 2048
        elif architecture == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class DenseNetModel(nn.Module):
    """
    DenseNet-based model with transfer learning
    """
    def __init__(self, num_classes=1, pretrained=True, dropout=0.5, architecture='densenet121'):
        super(DenseNetModel, self).__init__()
        
        # Load pretrained DenseNet
        if architecture == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = 1024
        elif architecture == 'densenet161':
            self.backbone = models.densenet161(pretrained=pretrained)
            num_features = 2208
        elif architecture == 'densenet169':
            self.backbone = models.densenet169(pretrained=pretrained)
            num_features = 1664
        elif architecture == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
            num_features = 1920
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Remove the final classification layer; torchvision DenseNet forward
        # will still perform GAP + flatten before calling classifier. By setting
        # classifier to Identity, the backbone will return a 2D tensor of shape (N, num_features).
        self.backbone.classifier = nn.Identity()
        
        # Add custom classifier for 2D features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Backbone returns 2D features (N, num_features)
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class EfficientNetModel(nn.Module):
    """
    EfficientNet-based model with transfer learning
    """
    def __init__(self, num_classes=1, pretrained=True, dropout=0.5, architecture='efficientnet_b0'):
        super(EfficientNetModel, self).__init__()
        
        # Load pretrained EfficientNet
        if architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = 1280
        elif architecture == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            num_features = 1280
        elif architecture == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = 1536
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Remove the final classification layer; torchvision EfficientNet forward
        # returns 2D features (N, num_features) when classifier is Identity.
        self.backbone.classifier = nn.Identity()
        
        # Add custom classifier for 2D features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Backbone returns 2D features (N, num_features)
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def get_model(config):
    """
    Factory function to create models based on configuration
    """
    architecture = config['model']['architecture']
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    dropout = config['model']['dropout']
    
    if architecture == 'custom_cnn':
        model = CustomCNN(num_classes=num_classes, dropout=dropout)
    
    elif architecture.startswith('resnet'):
        model = ResNetModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            architecture=architecture
        )
    
    elif architecture.startswith('densenet'):
        model = DenseNetModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            architecture=architecture
        )
    
    elif architecture.startswith('efficientnet'):
        model = EfficientNetModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            architecture=architecture
        )
    
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model

def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model, input_size=(1, 3, 224, 224)):
    """
    Print model summary
    """
    total_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total trainable parameters: {total_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(input_size)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {list(dummy_input.shape)}")
        print(f"Output shape: {list(output.shape)}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
    
    return total_params