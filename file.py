import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd

# Prepare file paths and labels
X = []
y = []
for root, _, filenames in os.walk('one-piece-cards/Cards'):
    for filename in filenames:
        if filename.endswith('.png'):
            file_path = os.path.join(root, filename)
            X.append(file_path)
            label = os.path.splitext(filename)[0]  # Remove the '.png' extension
            y.append(label)
X = pd.DataFrame(data=X, columns=['File Path'])
y = pd.DataFrame(data=y, columns=['Label'])

# Create a mapping from class index to card name
index_to_card_name = y['Label'].tolist()

# Define the model to match the architecture and class count of the saved model
def load_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Path to model weights
model_path = "best_model_notran.pth"

# Set num_classes to the original class count used during training
num_classes = 340  # Update this if the original number of classes differs
model = load_model(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the model state dict
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function for predicting the card name based on an image
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    # Map the predicted index to the card name using the index_to_card_name dictionary
    predicted_card_name = index_to_card_name[predicted.item()]
    return predicted_card_name

# Example usage
image_path = 'one-piece-cards/Cards/OP01/OP01-002.png'
prediction = predict(image_path)
print(f"Predicted card name: {prediction}")