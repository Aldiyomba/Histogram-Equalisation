# Histogram-Equalisation
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = "Pout.jpg"
img = cv.imread(path)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

equ = cv.equalizeHist(img)
cv.imshow('equ.png',equ)
cv.waitKey(0)
cv.destroyAllWindows()
hist,bins = np.histogram(equ.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upperleft')
plt.show()
