import torch.nn as nn

import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()
        
        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
