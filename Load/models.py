import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MyModel, self).__init__()

#        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        #self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)