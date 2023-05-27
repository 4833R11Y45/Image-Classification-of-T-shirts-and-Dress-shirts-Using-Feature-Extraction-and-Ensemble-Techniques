## Image Classification: T-shirts vs. Dress-shirts
This repository contains the code for a machine learning model that can classify between images of T-shirts and dress-shirts. The model is trained using a dataset derived from the Fashion-MNIST dataset.

# Files Provided
# training.py: 
Code to train and save the final model using extracted features.
# testing.py: 
Code to load the saved model and generate predictions for the test examples.

# Feature Extraction
The feature extraction process involves two techniques:
# HoG Features: 
We compute the Histogram of Oriented Gradients (HoG) features, which capture local gradients and edge directions in the images. This helps in capturing the shape and structure of the clothing items.
# Canny Edge Features: We perform Canny Edge Detection on the images to extract the edges. This provides information about the boundaries and contours of the clothing items.

By combining the HoG features and Canny edge features into a single feature vector, we aim to enhance the classification performance.

# Classification Techniques
We experiment with two classification techniques:

# K-Nearest Neighbors (KNN): 
A non-parametric algorithm that classifies new instances based on the majority vote of its k nearest neighbors in the feature space.

# Random Forest: 
An ensemble method that combines multiple decision trees to make predictions.

# Usage
Run the training.py script to train the models, optimize hyperparameters using cross-validation, and save the final model as myModel.pkl.
Run the testing.py script to load the saved model and generate predictions for the test examples. The predicted labels will be saved in myPredictions.csv.
Evaluation
The models' performance is evaluated using cross-validation, optimizing for classification accuracy. The best hyperparameters are chosen based on the cross-validation results.

The expected test accuracy for the final models is around 92% for the KNN model and 93% for the Random Forest model. However, the actual test accuracy may vary depending on the dataset's complexity and model generalization.

Feel free to explore the code and adapt it to your specific needs. Any feedback or suggestions are welcome!

Dependencies
The code requires the following dependencies:

NumPy
scikit-learn
matplotlib
OpenCV
scikit-image


You can install the dependencies using pip:
Image Classification: T-shirts vs. Dress-shirts
This repository contains the code for a machine learning model that can classify between images of T-shirts and dress-shirts. The model is trained using a dataset derived from the Fashion-MNIST dataset.

Files Provided
TrainData.csv: Contains 12,000 training examples. Each row represents a flattened 28x28 pixel grayscale image.
TrainLabels.csv: Contains the true labels for the training examples.
TestData.csv: Contains test examples.
training.py: Code to train and save the final model using extracted features.
testing.py: Code to load the saved model and generate predictions for the test examples.
Feature Extraction
The feature extraction process involves two techniques:

HoG Features: We compute the Histogram of Oriented Gradients (HoG) features, which capture local gradients and edge directions in the images. This helps in capturing the shape and structure of the clothing items.

Canny Edge Features: We perform Canny Edge Detection on the images to extract the edges. This provides information about the boundaries and contours of the clothing items.

By combining the HoG features and Canny edge features into a single feature vector, we aim to enhance the classification performance.

Classification Techniques
We experiment with two classification techniques:

K-Nearest Neighbors (KNN): A non-parametric algorithm that classifies new instances based on the majority vote of its k nearest neighbors in the feature space.

Random Forest: An ensemble method that combines multiple decision trees to make predictions.

Usage
Run the training.py script to train the models, optimize hyperparameters using cross-validation, and save the final model as myModel.pkl.
Run the testing.py script to load the saved model and generate predictions for the test examples. The predicted labels will be saved in myPredictions.csv.
Evaluation
The models' performance is evaluated using cross-validation, optimizing for classification accuracy. The best hyperparameters are chosen based on the cross-validation results.

The expected test accuracy for the final models is around 92% for the KNN model and 93% for the Random Forest model. However, the actual test accuracy may vary depending on the dataset's complexity and model generalization.

Feel free to explore the code and adapt it to your specific needs. Any feedback or suggestions are welcome!

Dependencies
The code requires the following dependencies:

NumPy
scikit-learn
matplotlib
OpenCV
scikit-image
You can install the dependencies using pip:
Image Classification: T-shirts vs. Dress-shirts
This repository contains the code for a machine learning model that can classify between images of T-shirts and dress-shirts. The model is trained using a dataset derived from the Fashion-MNIST dataset.

Files Provided
TrainData.csv: Contains 12,000 training examples. Each row represents a flattened 28x28 pixel grayscale image.
TrainLabels.csv: Contains the true labels for the training examples.
TestData.csv: Contains test examples.
training.py: Code to train and save the final model using extracted features.
testing.py: Code to load the saved model and generate predictions for the test examples.
Feature Extraction
The feature extraction process involves two techniques:

HoG Features: We compute the Histogram of Oriented Gradients (HoG) features, which capture local gradients and edge directions in the images. This helps in capturing the shape and structure of the clothing items.

Canny Edge Features: We perform Canny Edge Detection on the images to extract the edges. This provides information about the boundaries and contours of the clothing items.

By combining the HoG features and Canny edge features into a single feature vector, we aim to enhance the classification performance.

Classification Techniques
We experiment with two classification techniques:

K-Nearest Neighbors (KNN): A non-parametric algorithm that classifies new instances based on the majority vote of its k nearest neighbors in the feature space.

Random Forest: An ensemble method that combines multiple decision trees to make predictions.

Usage
Run the training.py script to train the models, optimize hyperparameters using cross-validation, and save the final model as myModel.pkl.
Run the testing.py script to load the saved model and generate predictions for the test examples. The predicted labels will be saved in myPredictions.csv.
Evaluation
The models' performance is evaluated using cross-validation, optimizing for classification accuracy. The best hyperparameters are chosen based on the cross-validation results.

The expected test accuracy for the final models is around 92% for the KNN model and 93% for the Random Forest model. However, the actual test accuracy may vary depending on the dataset's complexity and model generalization.

Feel free to explore the code and adapt it to your specific needs. Any feedback or suggestions are welcome!

Dependencies
The code requires the following dependencies:
NumPy
scikit-learn
matplotlib
OpenCV
scikit-image

You can install the dependencies using pip:
pip install numpy scikit-learn matplotlib opencv-python scikit-image
