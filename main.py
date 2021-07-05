import time

import cv2
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import os

webCam = False
highQuality = False
cap = cv2.VideoCapture(0)
cap.set(10, 160)
i = 0


def stack_images(img_array, scale, labels_arr=[]):
    rows_available = isinstance(img_array[0], list)

    if rows_available:
        num_of_stack_rows = len(img_array)
        num_of_img_in_each_row = len(img_array[0])
        num_of_channels = img_array[0][0].shape[0]
        rows_in_channel = img_array[0][0].shape[1]

        for x in range(0, num_of_stack_rows):
            for y in range(0, num_of_img_in_each_row):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), fx=scale, fy=scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)

        image_blank = np.zeros((num_of_channels, rows_in_channel, 3), np.uint8)
        hor = [image_blank] * num_of_stack_rows      # Array of image
        for x in range(0, num_of_stack_rows):
            hor[x] = np.hstack(img_array[x])         # Adding Stacked image to the array one by one
        ver = np.vstack(hor)

    else:
        num_of_img = len(img_array)

        for x in range(0, num_of_img):
            img_array[x] = cv2.resize(img_array[x], (0, 0), fx=scale, fy=scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor

    if (len(labels_arr) != 0) and rows_available:
        num_of_img_in_each_row = len(img_array[0])
        num_of_rows = len(img_array)
        total_images = num_of_rows * num_of_img_in_each_row
        if total_images <= len(labels_arr):
            each_img_width = int(ver.shape[1] / num_of_img_in_each_row)
            each_img_height = int(ver.shape[0] / num_of_rows)
            k = 0
            for i in range(0, num_of_rows):
                for j in range(0, num_of_img_in_each_row):
                    cv2.rectangle(ver, (j * each_img_width, each_img_height * i),
                                  (j * each_img_width + len(labels_arr[k]) * 13 + 27, 30 + each_img_height * i),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(ver, labels_arr[k], (each_img_width * j + 10, each_img_height * i + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (198, 133, 61), 2)
                    k += 1
        elif total_images > len(labels_arr):
            print("Expecting more labels to be given")

    elif (len(labels_arr) != 0) and (not rows_available):
        k = 0
        num_of_img = len(img_array)
        if num_of_img <= len(labels_arr):
            each_img_width = int(ver.shape[1] / num_of_img)
            for i in range(0, num_of_img):
                cv2.rectangle(ver, (i * each_img_width, 0), (i * each_img_width + len(labels_arr[k]) * 13 + 27, 30),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, labels_arr[k], (each_img_width * i + 10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (198, 133, 61), 2)
                k += 1
        elif num_of_img > len(labels_arr):
            print("Expecting more labels to be given")
    return ver


def reorder(corner_points):
    # print(corner_points)
    corner_points = corner_points.reshape((4, 2))
    # print(corner_points)
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = corner_points.sum(axis=1)                  # Axis = 1 sum along rows, Axis = 0 sum along column
    # print(add)
    new_points[0] = corner_points[np.argmin(add)]    # argmin returns index of smallest element
    new_points[3] = corner_points[np.argmax(add)]

    diff = np.diff(corner_points, axis=1)            # Along rows (axis = 1); right - left; y - x
    # print(diff)
    new_points[1] = corner_points[np.argmin(diff)]
    new_points[2] = corner_points[np.argmax(diff)]
    return new_points


def find_biggest_contour(all_contours):
    biggest_contour = np.array([])
    crop_width, crop_height = widthImg, heightImg
    max_area = 0
    for j in all_contours:
        area = cv2.contourArea(j)
        if area > 5000:
            peri = cv2.arcLength(j, True)                       # True = Shape is closed
            approx = cv2.approxPolyDP(j, 0.1 * peri, True)      # True = Shape is closed
            if area > max_area and len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)           # x,y starting point and w,h width and height
                aspect_ratio = float(w)/float(h)
                crop_width = w
                crop_height = int(crop_width / aspect_ratio)
                biggest_contour = approx
                max_area = area
    return biggest_contour, crop_width, crop_height


def draw_rectangle(image, contour, thickness):
    cv2.line(image, (contour[0][0][0], contour[0][0][1]), (contour[1][0][0], contour[1][0][1]), (0, 255, 0), thickness)
    cv2.line(image, (contour[0][0][0], contour[0][0][1]), (contour[2][0][0], contour[2][0][1]), (0, 255, 0), thickness)
    cv2.line(image, (contour[3][0][0], contour[3][0][1]), (contour[2][0][0], contour[2][0][1]), (0, 255, 0), thickness)
    cv2.line(image, (contour[3][0][0], contour[3][0][1]), (contour[1][0][0], contour[1][0][1]), (0, 255, 0), thickness)
    return image


def fun(x):
    return x


def initialize_trackbars(initial_trackbar_val=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 90)
    cv2.createTrackbar("Threshold1", "Trackbars", 170, 255, fun)
    cv2.createTrackbar("Threshold2", "Trackbars", 170, 255, fun)


def val_trackbars():
    threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = (threshold1, threshold2)
    return src


initialize_trackbars()
file_path = r"C:\Users\Win 10 Pc\PycharmProjects\SDP\Images\2.jpg"
if not webCam:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    img = cv2.imread(file_path)

while True:
    if webCam:
        success, img = cap.read()

    manageWin = 0

    aspectRatio = float(img.shape[1] / img.shape[0])
    if aspectRatio > 1.68:
        manageWin = 1

    if not highQuality:
        heightImg = 640
    else:
        manageWin = 2
        heightImg = img.shape[0]

    widthImg = int(heightImg * aspectRatio)

    img = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    threshold = val_trackbars()
    imgThreshold = cv2.Canny(imgBlur, threshold[0], threshold[1])

    kernel = np.ones((5, 5))
    imgMorphClosing = cv2.dilate(imgThreshold, kernel, iterations=4)
    imgMorphClosing = cv2.erode(imgMorphClosing, kernel, iterations=3)

    imgContours = img.copy()    # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgNew = img.copy()

    contours, hierarchy = cv2.findContours(imgMorphClosing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    biggest, scanW, scanH = find_biggest_contour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = draw_rectangle(imgBigContour, biggest, 2)

        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [scanW, 0], [0, scanH], [scanW, scanH]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (scanW, scanH))

        # REMOVE 5 PIXELS FORM EACH SIDE
        imgWarpColored = imgWarpColored[5:imgWarpColored.shape[0] - 5, 5:imgWarpColored.shape[1] - 5]
        imgWarpColored = cv2.resize(imgWarpColored, (scanW, scanH))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThreshold = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThreshold = cv2.bitwise_not(imgAdaptiveThreshold)
        imgAdaptiveThreshold = cv2.medianBlur(imgAdaptiveThreshold, 3)

    else:
        imgWarpColored = imgBlank
        imgWarpGray = imgBlank
        imgAdaptiveThreshold = imgBlank

    # Image Array for Display
    if manageWin == 1:
        imageInputArray = ([img, imgThreshold], [imgMorphClosing, imgBigContour])
        imageOutputArray = ([imgWarpColored, imgAdaptiveThreshold])
        labelsInput = ["Original", "Threshold", "Morphological Closing", "Biggest Contour"]
        labelsOutput = ["Warp Perspective", "Adaptive Threshold"]

    elif manageWin == 2:
        imageInputArray = ([imgBigContour])
        imageOutputArray = ([imgWarpColored])
        labelsInput = ["Biggest Contour"]
        labelsOutput = ["Warp Perspective"]

    else:
        imageInputArray = ([img, imgBlur, imgThreshold], [imgMorphClosing, imgContours, imgBigContour])
        imageOutputArray = ([imgWarpColored, imgWarpGray, imgAdaptiveThreshold])
        labelsInput = ["Original", "Gray & Blurred", "Edges", "Morphological Closing", "Contours", "Biggest Contour"]
        labelsOutput = ["Warp Perspective", "Warp Gray", "Adaptive Threshold"]

    stackedInputImage = stack_images(imageInputArray, 0.60, labelsInput)
    cv2.imshow("Input", stackedInputImage)

    stackedOutputImage = stack_images(imageOutputArray, 0.60, labelsOutput)
    cv2.imshow("Output", stackedOutputImage)

    # Save Output When 's' key is pressed
    if cv2.waitKey(1000) == ord('s'):
        now = datetime.now()
        d = now.strftime("%d/%m/%Y %H:%M:%S")
        formatted = d[0:2] + "-" + d[3:5] + "-" + d[6:10] + "_" + d[11:13] + "-" + d[14:16] + "-" + d[17:19]
        # print(formatted)
        if highQuality:
            HQ = "HQ"
        else:
            HQ = ""

        parentDir = r"C:\Users\Win 10 Pc\PycharmProjects\SDP\Scanned"
        newDir = formatted
        path = os.path.join(parentDir, newDir)
        os.mkdir(path)

        if not cv2.imwrite(r"Scanned/" + formatted + "/ScanImgColored_" + HQ + formatted + ".jpg", imgWarpColored):
            print("Problem While Saving Colored Image")

        if not cv2.imwrite(r"Scanned/" + formatted + "/ScanImgGray_" + HQ + formatted + ".jpg", imgWarpGray):
            print("Problem While Saving Gray Image")

        if not cv2.imwrite(r"Scanned/" + formatted + "/ScanImgAdaptive_" + HQ + formatted + ".jpg", imgAdaptiveThreshold):
            print("Problem While Saving Adaptive Image")

        im1 = Image.open(r"Scanned/" + formatted + "/ScanImgColored_" + HQ + formatted + ".jpg")
        im2 = Image.open(r"Scanned/" + formatted + "/ScanImgGray_" + HQ + formatted + ".jpg")
        im3 = Image.open(r"Scanned/" + formatted + "/ScanImgAdaptive_" + HQ + formatted + ".jpg")

        im1 = im1.convert("RGB")
        im2 = im2.convert("RGB")
        im3 = im3.convert("RGB")

        imList = [im2, im3]
        im1.save(r"Scanned/" + formatted + "/Scanned_PDF_Combined_" + formatted + ".pdf",
                 save_all=True, append_images=imList)
        im1.save(r"Scanned/" + formatted + "/Scanned_PDF_Colored_" + formatted + ".pdf")

        successImg = cv2.imread("Success.jpeg")
        successImg = cv2.resize(successImg, (0, 0), fx=0.6, fy=0.6)
        saved = 1
        if saved == 1:
            cv2.namedWindow("Success")
            cv2.moveWindow("Success", 570, 280)
            cv2.imshow("Success", successImg)
            cv2.waitKey(800)
            saved = 0
            cv2.destroyWindow("Success")

    if cv2.waitKey(900) == ord('q'):
        time.sleep(1)
        break

    if cv2.waitKey(1000) == ord('r'):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.destroyAllWindows()
