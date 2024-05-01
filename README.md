
<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h1 align="center">Course Project: Jersey Number Recognition By Detecting and Recognizing Region of Interest</h3>


</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#environment-configuration">Environment Configuration</a></li>
        <li><a href="#model-training">Model Training</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


This project is dealing with the challenge from SoccerNet to identify players' jersey number in each tracklet. We propose a progressive method that first extracts keyframes from player tracklets with DBNet model, filters out noisy data through various strategies, and then employs ABINet to recognize jersey numbers.



<!-- GETTING STARTED -->
## Getting Started

The following contains several work including model training, intermediate data processing, etc., that are used to improve the final challenge accuracy. The order is not sequential. 



### Environment Configuration
* conda
  ```
  conda env create -f environment.yml
  ```
    It may require some other packages when running some code. Please download required packages when necessary.

### Model Training
Run in Greatlakes. For each model training, the data we are using is from <a href="https://universe.roboflow.com/volleyai-actions/jersey-number-detection-s01j4">Volleyball dataset</a>. We downloaded it in yolov8 data format. 
- DBNet detection
  ```
  cd DB
  python train.py experiments/seg_detector/jersey_extra_dataset.yaml --num_gpus 1 --validate
  ```
- ABINet recognition
  ```
  cd ABINet
  python main.py --config=configs/train_abinet.yaml
  ```
- YoloV8 detection
  ```
  cd yolo-bb
  python -m yolo_obb_trainer
  ```
- YoloV8 recognition
  ```
  cd yolo-cls
  python -m train_yolo_cls
  ```  
- self-made CNN arch for recognition
  ```
  cd mnist
  python -m main
  ```

### Data Processing
Extract Keyframes and Crop
  - we conducted several ways to extract the keyframes from the tracklet.  
    - In DB, run the script ```crop_challenge.sh```
    - Others, run the script ```job_script_yolo_clean_chal.sh```

### Prediction
- In ABINet, run the script ```inference.sh```
- Others, run the script ```job_script_yolo_predict_chal.sh```
 



