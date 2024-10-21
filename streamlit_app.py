pip install --upgrade pip
pip install seaborn, streamlit, pandas, seaborn ,matplotlib ,scikit-image ,torch ,torchvision ,time
from pathlib import Path

#streamlit run /home/Scott_Baba_/Documents/stremlit/resume.py
import streamlit as st, time, seaborn as sea, pandas as pd
import matplotlib.pyplot as plt, torch.nn.functional as F, torch.optim as op
import torch, torch.nn as nn, PIL.Image as Image, torchvision.transforms as trans
from skimage import io

transform = trans.Compose([
    trans.ToTensor()
    ,trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

st.title("Model Tuning Exerceise")
st.markdown("*made in python*")

st.subheader("Experiment Setup")
st.write("""
To evaluate the performance of different activation functions:
CELU, ELU, Mish, ReLU, SELU, and Tanh
a series of experiments were conducted on a classifier that
detects for humans labeling them human or not human. The 
models were trained using the Stochastic gradient descent
(SGD) as the optimizer with a learning rate of 0.001 and 
a momentum of 0.9 and evaluated using accuracy. The training 
set is 200 images not containing people 200 with people and the
test se is 80 images, including 40 with people.



""")



code = """
class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.convl = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 4)
        self.fc1 = nn.Linear(20184, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    
    def forward(self, x):
        x = self.pool(F.relu(self.convl(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x
"""

#st.code(code,)
#mis =85 80 88.75
#tnh = 82.25 83.75 86.25
#celu = 81.25 78.75 91.25
#elu = 81.25 87.5 85
#selu = 90 86.25 88.75
#mish = 80 87.5 85

y=[((96.25+98.75+100.0)/3),(97.5),((95+97.5+98.75)/3),((96.25+98.75+96.25)/3),((100+98.75+100)/3),((92.5+96.25+96.25)/3)]
sea.barplot(x=["celu","elu","mish","relu","selu","tanh"], y=y, hue=y, palette="crest", legend=False)
plt.ylim(90, 100)
plt.title('Cross Entropy Loss')
plt.ylabel('Accuracy(%)')
plt.xlabel('Activation Function')
st.pyplot(plt)
plt.clf()

y=[((81.25+78.75+91.25)/3),((81.25+87.5+85)/3),((80+87.5+85)/3),((81.25+81.25+86.25)/3),((90+86.25+88.75)/3),((82.25+83.75+86.25)/3)]
sea.barplot(x=["celu","elu","mish","relu","selu","tanh"], y=y, hue=y, palette="crest", legend=False)
plt.ylim(70, 100)
plt.title('Multi Margin Loss')
plt.ylabel('Accuracy(%)')
plt.xlabel('Activation Function')
st.pyplot(plt)
plt.clf()

st.subheader("Results and Analysis")
st.write("""
The results consistently demonstrated that SELU outperformed 
the other activation functions in terms of accuracy. This 
superiority was observed across both cross-entropy loss 
and MultiMarginLoss.



""")

st.subheader("Conclusion")
st.write(r"""
While SELU demonstrated superior performance in this experiment, 
it's important to note that our primary goal was not to create 
a world-class image classifier but to compare the effectiveness 
of different activation functions. The results suggest that 
SELU's self-normalizing property, consistent performance, and 
robustness to different scenarios make it a promising choice 
for a wide range of deep learning tasks but thurther analysis 
with a much larger set of data will be needed to draw any 
serious conclusions.

!99% accuracy is not a true reflection of the model the images
in the training and test data are from the same source mostly
portrait pictures of people in clear view and its an a small 
data set 580(images)!
""")

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.convl = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 4)
        self.fc1 = nn.Linear(29*29*24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    
    def forward(self, x):
        x = self.pool(F.relu(self.convl(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNet()
model.load_state_dict(torch.load("deploy.pth"))
model.eval()  # Set the model to evaluation mode

def test_img(x):
    #Display image
    img = io.imread(x)
    io.imshow(img)

    # Load the image to pass it through neural network
    image = Image.open(x)

    # Resixing the image to 128x128
    resize_transform = trans.Resize((128, 128))
    image = resize_transform(image)


    # Preprocess the image
    image_tensor = transform(image)

    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Set the network to evaluation mode
    model.eval()

    # Pass the image through the network
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class index
    _, predicted = torch.max(output, 1)

    # Retrieve the class label (if applicable)
    class_mapping = {0: "Human", 1: "Not a Human"}
    class_label = class_mapping[predicted.item()]

    print("Predicted class:", class_label)
    if class_label=="Human":
        st.text("Predicted class: Human")
    else:
        st.text("Predicted class: Not a Human")

image=st.file_uploader("would you like to test the model", type=["png", "jpeg", "jpg", "gif", "tiff", "bmp", "webp"])
if image is not None:
    st.image(image)
    test_img(image)
with st.expander("Best Performing Model Code"):
    st.code(code,)

