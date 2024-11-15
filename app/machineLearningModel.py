# app/machineLearningModel.py
import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the PyTorch model


class MobileNetV3SmallClassifier(pl.LightningModule):
    def __init__(self, num_classes=131, learning_rate=1e-4):
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
model_path = 'Mobile_Pytorch_ArchFlow_v3.pth'

# Initialize the model
model = MobileNetV3SmallClassifier(num_classes=131)
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
    24: 'Trees Of Stone', 25: 'Trellis Garden', 26: 'Water Sculpture - Gambir Gate',
    27: 'alpine sea holly', 28: 'anthurium', 29: 'artichoke', 30: 'azalea',
    31: 'balloon flower', 32: 'barberton daisy', 33: 'bee balm', 34: 'bird of paradise',
    35: 'bishop of llandaff', 36: 'black-eyed susan', 37: 'blackberry lily',
    38: 'blanket flower', 39: 'bolero deep blue', 40: 'bougainvillea', 41: 'bromelia',
    42: 'buttercup', 43: 'californian poppy', 44: 'camellia', 45: 'canna lily',
    46: 'canterbury bells', 47: 'cape flower', 48: 'carnation', 49: 'cautleya spicata',
    50: 'clematis', 51: "colt's foot", 52: 'columbine', 53: 'common dandelion',
    54: 'common tulip', 55: 'corn poppy', 56: 'cosmos', 57: 'cyclamen', 58: 'daffodil',
    59: 'daisy', 60: 'desert-rose', 61: 'fire lily', 62: 'foxglove', 63: 'frangipani',
    64: 'fritillary', 65: 'garden phlox', 66: 'gaura', 67: 'gazania', 68: 'geranium',
    69: 'giant white arum lily', 70: 'globe thistle', 71: 'globe-flower', 72: 'grape hyacinth',
    73: 'great masterwort', 74: 'hard-leaved pocket orchid', 75: 'hibiscus', 76: 'hippeastrum',
    77: 'iris', 78: 'japanese anemone', 79: 'king protea', 80: 'lenten rose', 81: 'lilac hibiscus',
    82: 'lotus', 83: 'love in the mist', 84: 'magnolia', 85: 'mallow', 86: 'marigold',
    87: 'mexican petunia', 88: 'monkshood', 89: 'moon orchid', 90: 'morning glory',
    91: 'orange dahlia', 92: 'osteospermum', 93: 'passion flower', 94: 'peruvian lily',
    95: 'petunia', 96: 'pincushion flower', 97: 'pink primrose', 98: 'pink quill',
    99: 'pink-yellow dahlia', 100: 'poinsettia', 101: 'primula', 102: 'prince of wales feathers',
    103: 'purple coneflower', 104: 'red ginger', 105: 'rose', 106: 'ruby-lipped cattleya',
    107: 'siam tulip', 108: 'silverbush', 109: 'snapdragon', 110: 'spear thistle',
    111: 'spring crocus', 112: 'stemless gentian', 113: 'sunflower', 114: 'sweet pea',
    115: 'sweet william', 116: 'sword lily', 117: 'thorn apple', 118: 'tiger lily',
    119: 'toad lily', 120: 'tree mallow', 121: 'tree poppy', 122: 'trumpet creeper',
    123: 'wallflower', 124: 'water lily', 125: 'watercress', 126: 'wild geranium',
    127: 'wild pansy', 128: 'wild rose', 129: 'windflower', 130: 'yellow iris'
}

# Prediction function with confidence threshold


def predict_image(image, threshold=0.2):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_prob, predicted_class = torch.max(probabilities, 1)
        if max_prob.item() < threshold:
            return None, "No valid object detected"
    return predicted_class.item(), classLabels.get(predicted_class.item(), "Unknown")
