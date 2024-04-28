import cv2
import numpy as np
from keras_facenet import FaceNet

def FaceVerification(vs):
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    frame=0
    while frame<total_frames:
        _, image = vs.read()
        frame+=1

        # Searching in every 5th frame
        if frame%5==0:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = facecascade.detectMultiScale(gray, 1.1, 15, minSize=(60, 60))

            if len(faces)<2:
                continue
            elif len(faces)>2:
                # Return output as 1 - multiple faces found
                print('Multiple faces detected')
                return 1

            face_list = []
            for (x,y,w,h) in faces:
                face_list.append([x,y,w,h])
            
            x,y,w,h = faces[0][0],faces[0][1],faces[0][2],faces[0][3]
            face1 = image[y:y+h, x:x+w]
            x,y,w,h = faces[1][0],faces[1][1],faces[1][2],faces[1][3]
            face2 = image[y:y+h, x:x+w]

            # Resize faces to match the input size of FaceNet model
            face1 = cv2.resize(face1, (160, 160))
            face2 = cv2.resize(face2, (160, 160))

            # Extract embeddings
            embedding1 = facenet_model.embeddings([face1])[0]
            embedding2 = facenet_model.embeddings([face2])[0]

            # Calculate Score
            distance = np.linalg.norm(embedding1 - embedding2)

            if distance > threshold:
                # Faces not matching
                print('Faces not matching')
                return 1
       
    vs.release()
    print('Not more than one person face present in the video')
    return 0
    
if __name__ == "__main__":
    facenet_model = FaceNet()
    threshold = 1.1
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    vs = cv2.VideoCapture('dataset/veriff19.mp4')

    result = FaceVerification(vs)
    print(result)
   