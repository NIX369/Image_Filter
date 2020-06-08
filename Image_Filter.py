import numpy as np
import time
import cv2

path = input("Enter the path of the image: ")
img = cv2.imread(path,1)

dict = {1:"Gray Scale Image", 2:"Binary Image", 3:"Blue", 4:"Smoothing Image", 5:"Blurring the Image", 6:"Detecting the edge of an Image",
        7:"Saturation", 8:"Green", 9:"Red", 10:"Value ", 11:"Hue", 12:"Cropping the image", 13:"Doubling the Image", 14:"Transposing the image",
        15:"Face Detection"}

height = img.shape[0]
width = img.shape[1]

for i in range(1, len(dict) + 1):
    print(i,".", dict.get(i))

while True:
    num = input("Enter the number you would like to continue with: ")
    cv2.imshow("Image",img)
    if num == "1":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray Image", gray)
        img1 = gray

    elif num == "2":
         binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # through thresholding we will try to provide the value the value through which we can put below the particular value we assign the value 0 and above it will be white.
         cv2.imshow("Binary Image", binary_image)
         img1 = binary_image

    elif num == "3":
        B, G, R = cv2.split(img)
        zeros = np.zeros((height, width), dtype="uint8")
        img1 = cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

    elif num == "4":
        bilateral = cv2.bilateralFilter(img, 7, 20, 20)
        cv2.imshow("bilateralFilter Image ", bilateral)
        img1 = bilateral

    elif num == "5":
        blurImg = cv2.blur(img, (10, 10))
        cv2.imshow("blur image", blurImg)
        img1 = blurImg

    elif num == "6":
        canny = cv2.Canny(img, 100, 200)
        cv2.imshow("edges", canny)
        img1 = canny

    elif num == "7":
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("saturation", img_HSV[:, :, 1])
        img1 = img_HSV[:, :, 1]

    elif num == "8":
        B, G, R = cv2.split(img)
        zeros = np.zeros((height, width), dtype="uint8")
        img1 = cv2.imshow("Green", cv2.merge([zeros, G, zeros]))

    elif num == "9":
        B, G, R = cv2.split(img)
        zeros = np.zeros((height, width), dtype="uint8")
        img1 = cv2.imshow("Red", cv2.merge([zeros, zeros, R]))

    elif num == "10":
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("valuechannel", img_HSV[:, :, 2])
        img1 = img_HSV[:, :, 2]

    elif num=="11":
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("Hue", img_HSV[:, :, 0])
        img1 = img_HSV[:, :, 0]

    elif num == "12":
        height, width = img.shape[:2]
        start_row, start_col = int(height * 0.10), int(width * 0.10)  # starting pixel coordinates (topleft,of cropping rectangles)
        end_row, end_col = int(height * 0.86), int(width * 0.86)  # ending pixel coordinates (bottom right),this can be changed
        cropped = img[start_row:end_row, start_col:end_col]
        cv2.imshow('cropped', cropped)
        img1 = cropped

    elif num == "13":
        resized = cv2.resize(img, (int(img.shape[1]*1.5),int(img.shape[0]*1.5)))
        cv2.imshow('resized', resized)
        img1 = resized

    elif num == "14":
        rotation_image = cv2.transpose(img)  # this will covert the image of horizontal pixel elements into vertical pixel elements as in matrix
        cv2.imshow("legend", rotation_image)
        img1 = rotation_image

    elif num == "15":
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.06, minNeighbors=6)
        for x, y, w, h in faces:
            img1 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.imshow("Gray", img1)

    else:
        print('invalid input')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save = input('Do you want to save?y/n ')
    if save == 'y':
        file = input('Enter the image name to be saved ')
        cv2.imwrite(file + '.jpg', img1)
    elif save == 'n':
        print('its ok')
    else:
        print('invalid input')
    a = input('Do you break?y/n ')
    if a == 'y':
        break
    elif a == 'n':
        print('its ok')
    else:
        print('invalid input')
        pass

# Source: https://github.com/Raushan998/filtering-the-image/blob/master/filteringtheimage.py
