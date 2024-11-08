# app/machineLearningModel.py
import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the PyTorch model


class MobileNetV3SmallClassifier(pl.LightningModule):
    def __init__(self, num_classes=27, learning_rate=1e-4):
        super(MobileNetV3SmallClassifier, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)


# Load the model and set it to evaluation mode
model_path = 'Mobile_Pytorch_Architecture_v1.pth'

# Initialize the model
model = MobileNetV3SmallClassifier(num_classes=27)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Class labels mapping
classLabels = {
    0: 'Bandstand', 1: 'Botany Centre', 2: 'Bukit Timah Gate Visitor Centre',
    3: 'CDL Green Gallery', 4: 'Centre For Education And Outreach',
    5: 'Centre For Ethnobotany', 6: 'Centre For Urban Greenery And Ecology',
    7: 'Clock Tower', 8: 'Curved Waterfall Nassim Gate', 9: 'Gardens Shop',
    10: 'Heritage Museum Singapore Botanic Gardens', 11: 'Hoya House',
    12: 'Nassim Gate Visitor Centre', 13: 'National Biodiversity Centre',
    14: 'National Orchid Garden', 15: 'National Parks Board HQ',
    16: 'Plant House', 17: 'Prive Botanic Gardens', 18: 'Raffles Building',
    19: 'Shaw Foundation Symphony Stage', 20: 'Sprouts Food Place',
    21: 'Sundial Garden', 22: 'Swiss Granite Fountain', 23: 'The Garage',
    24: 'Trees Of Stone', 25: 'Trellis Garden', 26: 'Water Sculpture - Gambir Gate'
}

# Prediction function


def predict_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item(), classLabels.get(predicted_class.item(), "Unknown")
