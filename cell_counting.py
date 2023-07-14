import cv2
import numpy as np
import math

image = cv2.imread('H:\Semester 7\Engineering Project\FYP\FYP final\images\img00003.jpeg')   #Input image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      #Grayscale conversion
cv2.imshow('Original image', image)
#cv2.imshow('Gray image', gray)

gray = 255 - gray;
cv2.imshow('Negative', gray)

ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)    #Segmented Image
#cv2.imshow('Image',thresh)

median = cv2.medianBlur(thresh,5)  # Filtered image using median filer
#cv2.imshow('Filtered Image',median)

#median = cv2.medianBlur(thresh,5)  # Filtered image using median filer
#cv2.imshow('Filtered Image',median)

#laplacian = cv2.Laplacian(median,cv2.CV_64F)   #Laplacian filter
#sobelx = cv2.Sobel(median,cv2.CV_64F,1,0,ksize=5) #x
#sobely = cv2.Sobel(sobelx,cv2.CV_64F,0,1,ksize=5) #y
#cv2.imshow('imag',sobely)

kernel = np.ones((5,5),np.uint8)   #edge detection
img_dilation = cv2.dilate(median, kernel,iterations = 1)
img_erosion = cv2.erode(img_dilation,kernel, iterations=1)
#cv2.imshow('Erosion',img_erosion)

#imgb = thresh - median   #Smoothing
#cv2.imshow('smooth',imgb)
boundary = gray - img_erosion;

#imgb = 255 - median
#cv2.imshow('coverted',imgb)

# Labeling

ret, labels = cv2.connectedComponents(img_erosion)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

cv2.imshow('labeled image',labeled_img)

#plt.subplot(222)
#plt.title('Objects counted:'+ str(ret-1))
#plt.imshow(labeled_img)
print('objects number is:', ret-1)
#plt.show()
