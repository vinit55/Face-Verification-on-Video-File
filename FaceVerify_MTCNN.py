import sys
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

def FaceVerification(vs):
    print('Searching in Frames...')
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    frame=0
    fail_count = 0
    while frame<total_frames:
        _, image = vs.read()
        frame+=1

        # Searching in every 5th frame
        if frame%5==0:
            print('Frame# ',frame)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            faces = detector.detect_faces(image_rgb)
            if len(faces)<2:
                continue
            elif len(faces)>2:
                # Return output as 1 - multiple faces found
                print('Multiple faces detected')
                return 1

            x, y, w, h = faces[0]['box']
            face1 = image[y:y+h, x:x+w]
            x, y, w, h = faces[1]['box']
            face2 = image[y:y+h, x:x+w]


            # Resize faces to match the input size of FaceNet model
            face1 = cv2.resize(face1, (160, 160))
            face2 = cv2.resize(face2, (160, 160))

            # Extract embeddings
            embedding1 = facenet_model.embeddings([face1])[0]
            embedding2 = facenet_model.embeddings([face2])[0]

            # Calculate Score
            distance = np.linalg.norm(embedding1 - embedding2)
            print(distance)

            if distance > threshold:
                if fail_count>2:
                    # Faces not matching
                    print('Faces not matching')
                    return 1
                else:
                    fail_count += 1
       
    vs.release()
    print('Not more than one person face present in the video')
    return 0
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Add Video file path")
        sys.exit(1)
        
    video_path = sys.argv[1]
    detector = MTCNN()
    facenet_model = FaceNet()
    threshold = 1.1
    vs = cv2.VideoCapture(video_path)

    result = FaceVerification(vs)
    print('Final Output: ',result)
   