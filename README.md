# Waste-Segmentation-Using-Deep-Learning

Waste Segmentation involves categorizing images of waste into four distinct classes: Biodegradable waste, Solid waste, E-waste, and Sanitary waste. This segmentation process is crucial for efficient waste management, allowing for better sorting and disposal strategies. However, manually sorting waste images is time-consuming and impractical. Hence, automating this process using deep learning techniques can significantly enhance waste management systems.

## Explanation:

#### Data Preparation:
The project starts with defining data transformations for augmentation and normalization. These transformations help in enhancing the model's robustness and performance.
The dataset is organized into three subsets: test, train, and validation, each containing images of the four waste classes.
PyTorch's ImageFolder class is used to load the dataset, and data loaders are created to efficiently iterate over the data during training.

#### Model Selection and Initialization:
The project utilizes a pre-trained ResNet-18 model, a convolutional neural network architecture known for its effectiveness in image classification tasks.
The final classification layer of the ResNet-18 model is modified to accommodate the specific output units required for waste segmentation (initially 1000, then reduced to 2).
We freeze all layers except the final classification layer to retain the pre-trained weights and avoid overfitting.

#### Training Loop:
The training loop consists of iterating over the specified number of epochs while alternating between training and evaluation phases.
During the training phase, the model learns from the training data and adjusts its parameters to minimize the defined loss function (CrossEntropyLoss).
The evaluation phase assesses the model's performance on both the training and validation sets to monitor training progress and prevent overfitting.
The training loop prints the loss and accuracy metrics for each phase in every epoch.

#### Model Evaluation and Saving:
Once training is complete, the trained model is saved to disk for future use.
Additionally, a saved model is loaded and evaluated using an unseen image to demonstrate its predictive capability.
The image undergoes preprocessing, including resizing, center cropping, and normalization, before being fed into the model for inference.
The model predicts the class label of the image, which is then mapped to the corresponding waste category.
Finally, the original image along with the predicted class label is displayed for visual inspection.
