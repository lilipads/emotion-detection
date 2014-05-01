"""
resize edge image to standard size
"""
import cv

img = cv.LoadImage("datacropped/f4001-c.jpg")
new_img = cv.resize(img, [28,28])
cv.SaveImage("test.jpg", new_img)

