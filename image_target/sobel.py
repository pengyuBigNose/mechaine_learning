import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def CannyF():
  # 高斯滤波
  imgCvG = cv.GaussianBlur(imgCv, (3, 3), 0)
  # sobel
  x_sobel = cv.Sobel(imgCv, cv.CV_16S, 1, 1, 3)
  abs = cv.convertScaleAbs(x_sobel) # 转为uint8格式
  print(abs)

  titles = ["Original Image", "GaussianBlur", "Sobel X"]
  images = [imgCv, imgCvG, abs]
  length = len(images)
  for i in range(length):
    plt.subplot(1, length, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
  plt.show()


if __name__ == '__main__':
  imgSrc = "image_target/img/monkey1.jpg"
  imgCv = cv.imread(imgSrc, cv.COLOR_RGBA2GRAY)
  CannyF()

  cv.waitKey(0)