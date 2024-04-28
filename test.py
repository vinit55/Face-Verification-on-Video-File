import cv2
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vs = cv2.VideoCapture('C:/Users/VINIT/Desktop/Data_Scientist_-_Take-home_Assignment/dataset/veriff1.mp4')
_, image = vs.read()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = facecascade.detectMultiScale(gray, 1.1, 4)
    
for (x,y,w,h) in faces:
    cv2.imwrite('facebox1.jpg',image[y:y+h, x:x+w])
    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
    

cv2.imwrite('face1.png',image)
cv2.imshow('img', image)
cv2.waitKey(1)

vs.release()
