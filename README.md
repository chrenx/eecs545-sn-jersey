
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

The following contains several work including model training, intermediate data processing, etc., that are used to improve the accuracy of final challenge result. The order is not sequential. 

### Environment Configuration
* conda
  ```
  conda env create -f environment.yml
  ```
    It may require some other packages when running some code. Please download required packages when necessary.

### Model Training
#### DBNet
  ```
  cd DB
  python train.py /home/XXXX/DB/experiments/seg_detector/jersey_extra_dataset.yaml --num_gpus 1 --validate
  ```



