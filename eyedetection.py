import openCV as cv

imcolor = cv.LoadImage('detectionimg.jpg') # input image
# loading the classifiers
haarFace = cv.Load('haarcascade_frontalface_default.xml')
haarEyes = cv.Load('haarcascade_eye.xml')
# running the classifiers
storage = cv.CreateMemStorage()
detectedFace = cv.HaarDetectObjects(imcolor, haarFace, storage)
detectedEyes = cv.HaarDetectObjects(imcolor, haarEyes, storage)

# draw a green rectangle where the face is detected
if detectedFace:
 for face in detectedFace:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(155, 255, 25),2)

# draw a purple rectangle where the eye is detected
if detectedEyes:
 for face in detectedEyes:
  cv.Rectangle(imcolor,(face[0][0],face[0][1]),
               (face[0][0]+face[0][2],face[0][1]+face[0][3]),
               cv.RGB(155, 55, 200),2)

cv.NamedWindow('Face Detection', cv.CV_WINDOW_AUTOSIZE)
cv.ShowImage('Face Detection', imcolor) 
cv.WaitKey()