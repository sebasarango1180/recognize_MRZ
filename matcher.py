#!/usr/bin/python2

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image
import datetime
import json

# All the 6 methods for comparison in a list

class Matcher:

    def __init__(self):

        '''self.methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                        'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']'''

        self.methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

        self.templates_dir = './templates/'
        self.ines_dir = './ines/'

        self.template_logo = cv2.imread('./templates/logoEUM.png', 0)
        self.template_IFE = cv2.imread('./templates/institutoFederalElectoral.png', 0)
        self.template_Cred = cv2.imread('./templates/credencialParaVotar.png', 0)

        self.is_ine = False
        self.has_logo = False
        self.has_IFE_banner = False
        self.has_Cred_banner = False

    def preprocess_image(self, img, size):

        #img = imutils.resize(img, height=size)

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #Convert to gray. Only if image is not uploaded with parameter 0.

        # initialize a rectangular and square structuring kernel
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        #sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

        # smooth the image using a 3x3 Gaussian, then apply the blackhat
        # morphological operator to find dark regions on a light background
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rectKernel)

        #thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #Filtering white noise.
        #dilation = cv2.dilate(thresh, np.ones((2,2),np.uint8),iterations = 2)
        #erosion = cv2.erode(thresh, np.ones((3,3),np.uint8),iterations = 3)

        '''cv2.imshow("tresh", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        #return erosion
        #return thresh
        return blackhat

    def check_methods(self, pic):

        img = cv2.imread(self.ines_dir + pic, 0)

        template = self.template_logo

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for meth in self.methods:

            method = eval(meth)

            print("Method: {}".format(method))
            w, h = template.shape[::-1]

            # Apply template Matching
            result = cv2.matchTemplate(gray_img, template, method)
            loc = np.where(result >= 0.8)

            for pt in zip(*loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

            cv2.imshow("Detected",img)
            '''plt.subplot(122), plt.imshow(self.img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)

            plt.show()'''
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def check_logo(self, logo):

        img = cv2.imread(self.ines_dir + logo, 0)
        template = self.template_logo

        img_black = self.preprocess_image(img, size=300)

        cv2.imshow("Original black", img_black)
        #cv2.imshow("Original black", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        template_black = self.preprocess_image(template, size=90)

        cv2.imshow("Template black", template_black)
        #cv2.imshow("Template black", template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        w, h = template.shape[::-1]

        # Apply template Matching

        res = cv2.matchTemplate(img_black, template_black, cv2.TM_CCOEFF_NORMED)
        #res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
        print("Matching (logo): {}".format(np.amax(res)))
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        max_loc = cv2.minMaxLoc(res)[3]

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, 255, 2)

        '''result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF)
        loc = np.where(result >= 0.8)'''
        '''for pt in zip(*loc[::-1]):   #When pattern is repeated in the picture.
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)'''

        img = imutils.resize(img, 700)
        cv2.imshow("Detected", img)

        '''plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Logo'), plt.xticks([]), plt.yticks([])

        plt.show()'''

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def check_IFE_banner(self, IFE_banner):

        img = cv2.imread(self.ines_dir + IFE_banner, 0)
        template = self.template_IFE

        img_black = self.preprocess_image(img, size=400)

        cv2.imshow("Original black", img_black)
        cv2.waitKey(0)

        template_black = self.preprocess_image(template, size=40)

        cv2.imshow("Template black", template_black)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        w, h = template.shape[::-1]

        # Apply template Matching

        res = cv2.matchTemplate(img_black, template_black, cv2.TM_CCOEFF)
        print("Matching (IFE): {}".format(np.amax(res)))
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        max_loc = cv2.minMaxLoc(res)[3]

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        
        '''result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF)
        loc = np.where(result >= 0.8)'''
        '''for pt in zip(*loc[::-1]):   #When pattern is repeated in the picture.
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)'''

        img = imutils.resize(img, 700)
        cv2.imshow("Detected", img)

        '''plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Logo'), plt.xticks([]), plt.yticks([])

        plt.show()'''

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def check_Cred_banner(self, Cred_banner):

        img = cv2.imread(self.ines_dir + Cred_banner, 0)
        template = self.template_Cred

        img_black = self.preprocess_image(img, size=400)

        '''cv2.imshow("Original black", img_black)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        template_black = self.preprocess_image(template, size=30)

        '''cv2.imshow("Template black", template_black)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        w, h = template.shape[::-1]

        # Apply template Matching

        res = cv2.matchTemplate(img_black, template_black, cv2.TM_CCOEFF)
        #res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
        print("Matching (Cred): {}".format(np.amax(res)))
        max_loc = cv2.minMaxLoc(res)[3]

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, 255, 2)
        
        '''result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF)
        loc = np.where(result >= 0.8)'''
        '''for pt in zip(*loc[::-1]):   #When pattern is repeated in the picture.
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)'''

        img = imutils.resize(img, 700)
        cv2.imshow("Detected", img)

        '''plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Logo'), plt.xticks([]), plt.yticks([])

        plt.show()'''

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def text_extraction_MRZ(self, img):

        data_dict = dict()
        img = cv2.imread(self.ines_dir + img, 0)
        #img2 = img.copy()
        #img = self.preprocess_image(img, 400)
        mrz = self.detect_MRZ_region(img)

        #find_text = cv2.bitwise_and(img, mask)
        #find_text = cv2.threshold(find_text, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #cv2.imshow("AND", find_text)
        '''cv2.imshow("AND",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        #cv2_to_PIL = Image.fromarray(find_text)
        #cv2_to_PIL = Image.fromarray(img)
        cv2_to_PIL = Image.fromarray(mrz)

        text = pytesseract.image_to_string(cv2_to_PIL)
        text = text.encode('utf-8')

        elector_key_number = (((text.split('\n')[1])).split('<<')[0]).split('<')[0][0:6]

        data_dict['id_number'] = (text.split('\n')[0]).split('<<')[0][5::]
        data_dict['registry_month'] = ((text.split('\n')[1]).split('<<')[0]).split('<')[1]
        data_dict['elector_key_number'] = (((text.split('\n')[1])).split('<<')[0]).split('<')[0][0:6]
        data_dict['birth_date'] = elector_key_number[4:6] + '/' + elector_key_number[2:4] + '/' + elector_key_number[0:2]
        data_dict['gender'] = (((text.split('\n')[1])).split('<<')[0]).split('<')[0][7]
        data_dict['last_name_1'] = ((text.split('\n')[2]).split('<<')[0]).split('<')[0]
        data_dict['last_name_2'] = ((text.split('\n')[2]).split('<<')[0]).split('<')[1]
        data_dict['first_name'] = ((text.split('\n')[2]).split('<<')[1]).split('<')[0]
        data_dict['second_name'] = ((text.split('\n')[2]).split('<<')[1]).split('<')[1]
        
        data = json.dumps(data_dict)

        return data

    def detect_MRZ_region(self, img):

        #img = imutils.resize(img, height=size)

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #Convert to gray. Only if image is not uploaded with parameter 0.

        # initialize a rectangular and square structuring kernel
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 20))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))

        # smooth the image using a 3x3 Gaussian, then apply the blackhat
        # morphological operator to find dark regions on a light background
        #blur = cv2.GaussianBlur(img, (3, 3), 0)
        #blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rectKernel)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKernel)

        #Filtering white noise.
        #kernel = np.ones((3,3),np.uint8)
        #dilation = cv2.dilate(blackhat,kernel,iterations = 2)
        #erosion = cv2.erode(dilation,kernel,iterations = 3)


        # compute the Scharr gradient of the blackhat image and scale the
        # result into the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        '''cv2.imshow("Gradiente",gradX)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        #Horizontal closing
        # apply a closing operation using the rectangular kernel to close
        # gaps in between letters -- then apply Otsu's thresholding method
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #Vertical closing
        # perform another closing operation, this time using the square
        # kernel to close gaps between lines of the MRZ, then perform a
        # series of erosions to break apart connected components
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        thresh = cv2.erode(thresh, None, iterations=4)

        # during thresholding, it's possible that border pixels were
        # included in the thresholding, so let's set 5% of the left and
        # right borders to zero
        '''p = int(image.shape[1] * 0.05)
        thresh[:, 0:p] = 0
        thresh[:, image.shape[1] - p:] = 0'''

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and use the contour to
            # compute the aspect ratio and coverage ratio of the bounding box
            # width to the width of the image
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            crWidth = w / float(img.shape[1])
    
            # check to see if the aspect ratio and coverage width are within
            # acceptable criteria
            if ar > 4 and crWidth > 0.75:
                # pad the bounding box since we applied erosions and now need
                # to re-grow it
                pX = int((x + w) * 0.03)
                pY = int((y + h) * 0.03)
                (x, y) = (x - pX, y - pY)
                (w, h) = (w + (pX * 2), h + (pY * 2))
    
                # extract the ROI from the image and draw a bounding box
                # surrounding the MRZ
                roi = img[y:y + h, x:x + w].copy()
                # print(type(roi))
                # print(str(roi))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break

        '''cv2.imshow("Vertical close",thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        #return thresh
        return roi

    def read_OCR_front(self, img):

        #img = self.preprocess_image(img, )
        img = cv2.imread(self.ines_dir + img, 0)
        regions = self.detect_OCR_regions(img)
        print(str(regions))
        for r in regions:
            print("region")
            cv2_to_PIL = Image.fromarray(r)

            text = pytesseract.image_to_string(cv2_to_PIL)
            text = text.encode('utf-8')
            print(text)

    def detect_OCR_regions(self, img):

        text_regions = []
        #img = imutils.resize(img, height=size)

        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #Convert to gray. Only if image is not uploaded with parameter 0.

        # initialize a rectangular and square structuring kernel
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 20))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))

        # smooth the image using a 3x3 Gaussian, then apply the blackhat
        # morphological operator to find dark regions on a light background
        #blur = cv2.GaussianBlur(img, (3, 3), 0)
        #blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rectKernel)
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKernel)

        #Filtering white noise.
        #kernel = np.ones((3,3),np.uint8)
        #dilation = cv2.dilate(blackhat,kernel,iterations = 2)
        #erosion = cv2.erode(dilation,kernel,iterations = 3)


        # compute the Scharr gradient of the blackhat image and scale the
        # result into the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        '''cv2.imshow("Gradiente",gradX)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        #Horizontal closing
        # apply a closing operation using the rectangular kernel to close
        # gaps in between letters -- then apply Otsu's thresholding method
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #Vertical closing
        # perform another closing operation, this time using the square
        # kernel to close gaps between lines of the MRZ, then perform a
        # series of erosions to break apart connected components
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        thresh = cv2.erode(thresh, None, iterations=4)

        # during thresholding, it's possible that border pixels were
        # included in the thresholding, so let's set 5% of the left and
        # right borders to zero
        '''p = int(image.shape[1] * 0.05)
        thresh[:, 0:p] = 0
        thresh[:, image.shape[1] - p:] = 0'''

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
        print("Contours: {}".format(cnts))
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and use the contour to
            # compute the aspect ratio and coverage ratio of the bounding box
            # width to the width of the image
            (x, y, w, h) = cv2.boundingRect(c)
            '''ar = w / float(h)
            crWidth = w / float(img.shape[1])
    
            # check to see if the aspect ratio and coverage width are within
            # acceptable criteria
            if ar < 4 and crWidth < 0.6:
                # pad the bounding box since we applied erosions and now need
                # to re-grow it
                pX = int((x + w) * 0.03)
                pY = int((y + h) * 0.03)
                (x, y) = (x - pX, y - pY)
                (w, h) = (w + (pX * 2), h + (pY * 2))'''
    
            # extract the ROI from the image and draw a bounding box
            # surrounding the MRZ
            roi = img[y:y + h, x:x + w].copy()
            # print(type(roi))
            # print(str(roi))
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text_regions.append(roi)

        '''cv2.imshow("Vertical close",thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        #return thresh
        return text_regions

if __name__== '__main__':

    rtc_inicio = datetime.datetime.now()

    #print(str(rtc_inicio))
    #ine = cv2.imread('./ines/7mod.jpeg', 0) #Uploads image in grayscale.
    #ine = cv2.imread('./ines/cedulaDiego-front.jpg', 0)
    #ine = cv2.imread('./ines/13e.png', 0)
    #ine = cv2.imread('./ines/12d.png', 0)
    #ine = cv2.imread('./ines/6mod.jpeg', 0)
    ine = '16back.png'

    #template = cv2.imread('./templates/logoEUM.png', 0)
    #template = cv2.imread('./templates/credencialParaVotar.png', 0)

    matcher = Matcher()

    #matcher.check_logo(ine)
    #matcher.check_IFE_banner(ine)
    #matcher.check_Cred_banner(ine)
    data = matcher.text_extraction_MRZ(ine)
    #matcher.read_OCR_front(ine)

    rtc_final = datetime.datetime.now()
    delta = rtc_final - rtc_inicio
    #print(str(rtc_final))
    print("Elapsed time: {} ms.".format(delta.total_seconds() * 1000))
