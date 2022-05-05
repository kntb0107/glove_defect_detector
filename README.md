# Glove Defect Measurement System

Due to the COVID-19 rampage which has now become our daily lives, the usage of these medical gloves has increased in production. It is important to make sure that the gloves are sterile and are not riddled with an defects, and when there is one then it should be taken care of immediately.
This project was inspired with that in mind, and the problem was simplified to make this system to work.

The final program works with both OpenCV and AI, where the defect gets measured in mm, the hand oreintation is mentioned and the probability of whether the hole is small, medium or large is displayed. They are displayed and detected live on webcam.


### PYTHON FILES
1. opencv_mes.py --> detecting the hand oreintation + measuring the contours using OPENCV
2. detect.py --> using custom trained YOLOv5 weights, the probabilities of the sizes are detected. multiple holes can be detected.
3. combined.py --> both 1 and 2 python files are combined to produce both results at the same time.

### NOTEBOOK FILE
1. training.py --> training the custom dataset to create a custom yolov5 weight for detection purposes.

### Dataset used
- custom dataset used
- same colored gloves
- drawn on marker holes on white gloves
- male subject for photography
- annotated in Roboflow

## HOW TO RUN THE THE COLAB AND DETECT.PY FILE

Download the dataset and then make sure you upload the dataset inside the yolov5 folder.

## Results

opencv_mes.py detects both the hand orientation and measures the contours. 
![image](https://user-images.githubusercontent.com/72517618/166843060-0161b46d-3516-44fa-87e0-8ee9e6c9380d.png)

detect.py detects the sizes and its probabilites. for now, it only detects in three sizes.
![image](https://user-images.githubusercontent.com/72517618/166843317-b78d15b6-4c70-418d-b6cf-b794efcee011.png)


