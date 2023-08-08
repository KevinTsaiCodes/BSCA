import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse

# Import the ResNet model and the BasicBlock class from your training script
from train import ResNet, BasicBlock

# Define the class mapping (index to class name) used during training
class_mapping = {
    0: 'class_1',
    1: 'class_2',
    2: 'class_3',
    3: 'class_4',
}

# Function to load the image, preprocess it, and make predictions
def predict_image(image_path, model_path):
    # Load the image in full color
    image = Image.open(image_path).convert('RGB')

    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),     # Resize to the desired input size of the model
        transforms.ToTensor(),             # Convert the image to a PyTorch tensor
    ])

    # Preprocess the image and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)

    # Load the model instance and its weights
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=len(class_mapping))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()

    with torch.no_grad():
        # Make predictions
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class_idx = predicted.item()
        predicted_class = class_mapping[predicted_class_idx]
    return predicted_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This is a script to predict the class of an image"
                                                 " using a trained model. Type python predict_image.py"
                                                 " -h for more information")
    parser.add_argument("-i", "--image_path", type=str, required=True, help="path/to/the/input/image")
    parser.add_argument("-m", "--model_path", type=str, required=False, default="model/brain_slice_classifier_model.pt", help="path/to/the/trained/model")
    args = parser.parse_args()

    result = predict_image(args.image_path, args.model_path)
    print("Predicted class:", result)
