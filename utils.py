import numpy as np
import cv2
import matplotlib.pyplot as plt


def preprocess(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray,(9,9),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    inverted = cv2.bitwise_not(thresh,0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    morph = cv2.morphologyEx(inverted,cv2.MORPH_OPEN,kernel)
    result = cv2.dilate(morph,kernel)
    return result

def findcontours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    imgContours = img.copy()
    return contours

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def get_sudoku_grid(img , largest_contour):
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = img[y:y+h, x:x+w]
    return cropped_image

def fix_perspective(image, contour_points):
    dst_points = np.array([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]], dtype=np.float32)
    src_points = np.array(contour_points, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    output_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    # bordered_image = cv2.copyMakeBorder(output_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    kernel = np.ones((3,3),np.uint8)
    # dilated_image = cv2.dilate(output_image, kernel, iterations=1)
    return output_image

def remove_lines(image):
    
    image = cv2.resize(image,(900,900))
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=800, minLineLength=700, maxLineGap=40)
    line_image = np.zeros_like(image)
  
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
   
    line_image = cv2.bitwise_not(line_image)
    removed_lines_image = cv2.bitwise_and(image, image, mask=line_image)
    return removed_lines_image

def overlay(puzzle,solved_puzzle,image):
    cell_size = 900 // 9  
    new_image = image.copy()
    for x in range(0, 9):
        for y in range(0, 9):
            if puzzle[x][y] == 0:
                print('fpund')
                digit = solved_puzzle[x][y]
                position = (int((y + 0.35) * cell_size), int((x + 0.65) * cell_size))  
                
                new_image = cv2.putText(image, str(digit), position, cv2.FONT_HERSHEY_SIMPLEX, 2 , (200,23,0 ),3)
            else:
                print("Something wrong")
    return new_image