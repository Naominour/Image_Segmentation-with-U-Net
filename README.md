# Image Segmentation with U-Net

This project focuses on image segmentation using a U-Net architecture. The aim is to accurately segment images to identify and delineate various objects within them. This project involves training a deep learning model to perform image segmentation tasks effectively.

![Image Segmentation](https://img.shields.io/badge/Skill-Image%20Segmentation-green)
![Deep Learning](https://img.shields.io/badge/Skill-Deep%20Learning-yellow)
![Neural Network U-Net](https://img.shields.io/badge/Skill-Neural%20Network%20U-Net-blueviolet)
![Model Training and Evaluation](https://img.shields.io/badge/Skill-Model%20Training%20and%20Evaluation-orange)
![Python Programming](https://img.shields.io/badge/Skill-Python%20Programming-brightgreen)


## File Structure
```bash
├── Image_segmentation_Unet_v2.ipynb # Jupyter notebook with the main code for image segmentation
├── outputs.py # Script for handling the outputs
├── test_utils.py # Utility functions for testing
└── data/ # Directory containing the dataset  
```

## Frameworks and Libraries
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.3.3-red.svg?style=flat&logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-blue.svg?style=flat&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.2.2-yellow.svg?style=flat&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6.2-green.svg?style=flat&logo=matplotlib)

## Project Architecture
1. **Data Preparation**:
   - The `test_utils.py` file handles the preparation and processing of the data, ensuring it is in the correct format for training and evaluation.

2. **Model Training**:
   - `The Image_segmentation_Unet_v2.ipynb` notebook contains the implementation of the U-Net model. This notebook covers the training process, including data loading, model building, and training the segmentation model.

3. **Output Management**:
   - The `outputs.py` file manages the output of the model, including saving and visualizing the segmented images.

## Key Features
- Efficient image segmentation using U-Net architecture
- Customizable model parameters
- Data preprocessing and augmentation
- Visualization of segmentation results
- Performance evaluation metrics

## Usage
**Clone the Repository:**
```bash
   git clone https://github.com/yourusername/your-repo-name.git
```
**Navigate to the project directory:**
```bash
cd your-repo-name
```
**Install Dependencies:**
```bash
pip install -r requirements.txt
```
**Run the Jupyter Notebook:**
```bash
jupyter notebook Image_segmentation_Unet_v2.ipynb
```
## Implementation
The project involves the following steps:

- **Data Preparation:** Preparing musical data for training using existing utilities.
- **Model Training:** Building and training the LSTM network with my modifications.
- **Music Generation:** Generating new jazz solos using the trained model.

## Results
**Audio Samples**
The trained U-Net model demonstrates effective image segmentation capabilities, with visualizations of the segmentation results. The model's performance is evaluated based on metrics such as accuracy, precision, recall, and IoU (Intersection over Union).
Feel free to explore the project files and experiment with different parameters to improve the segmentation results.

<center><img src="images\output-1.png" style="width:600px;"></center>
<center><img src="images\output-2.png" style="width:600px;"></center>
<center><img src="images\output-3.png" style="width:600px;"></center>
<center><img src="images\output-4.png" style="width:600px;"></center>
<center><img src="images\output-5.png" style="width:600px;"></center>
<center><img src="images\output-6.png" style="width:600px;"></center>
