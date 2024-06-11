# Pneumonia-Detection
**Topic: Healthcare** <br>
**Problem Statement: Detection of anomalies in medical images (e.g., X-rays, MRIs)** <br>
**Selected Pre-trained Model: AlexNet** <br>
**Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data** <br>

## Steps :
1) Importing Libraries: 'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'keras'

2) Dataset Loading : <br>
Defined dataset folder path. <br>
Loaded training, validation, and test datasets into pandas DataFrames. <br>

3) Dataset Analysis : <br>
Computed sizes of training, validation, and test sets. <br>
Printed dataset sizes for understanding distribution. <br>

4) Image Data Preparation : <br>
Set image dimensions (224x224) and batch size (32). <br>
Created ImageDataGenerator instances for data augmentation and preprocessing. <br>

5) Model Architecture (AlexNet) : <br>
Used a pre-trained AlexNet model. <br>
Convolutional layers with 3x3 filters and same padding. <br>
Max pooling layers with 3x3 pool size and stride of 2. <br>
Flattening layer to convert 3D feature maps to 1D feature vectors. <br>
Fully connected layers with ReLU activation. <br>
Sigmoid activation for output layer. <br>

6) Model Training : <br>
Trainable params: 117,277,057 (447.38 MB) <br>
Trained the model on 10 epochs. <br>
**Epoch 4/10 had: Accuracy: 0.9676 - Loss: 0.1041 - Val_Accuracy: 1.0000 - Val_Loss: 0.0217 <br>
This is the best model so far.<br>
It achieved an accuracy of 97.76% and a loss of 0.1041. <br>
It's validation accuracy is 100% and validation loss is 0.0217.** <br>

7) Visualized some correctly and incorrectly predicted classes
   
8) Model Summary : <br>
Provided a summary diagram of the VGG16 model architecture. <br>
