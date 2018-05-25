import cv2
import os
import imutils
import numpy as np

dir_path = "./ines/brute/"

def main():

  for img_file in os.listdir(dir_path):
    print img_file
    img = cv2.imread(dir_path + img_file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    dilate = cv2.dilate(thresh.copy(), np.ones((5,5), np.uint8), iterations=1)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for contour in cnts:
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg'
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
        # write original image with added contours to disk
        cv2.imshow('captcha_result', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

main()