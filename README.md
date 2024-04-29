
# Face Verification ML Model Pipeline

This repository contains code for detecting and verifying faces in a video file. 

## Requirements

- Python 3.x
- OpenCV
- Tensorflow
- Keras-FaceNet
    - ```
      pip install keras-facenet
      ```
- MTCNN
    - ```
      pip install mtcnn
      ```

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/vinit55/Face-Verification-on-Video-File.git
    ```

2. Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
Navigate
```
cd Face-Verification-on-Video-File
```

To detect faces in a video, run one of the following command:
- MTCNN + FaceNet Model
```bash
python FaceVerify_MTCNN.py <video_filepath>
```

- HaarCascadeCkassifier + FaceNet Model
```
python FaceVerify.py <video_filepath>
```
