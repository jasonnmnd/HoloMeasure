import numpy as np
import cv2
import skimage
import os
import matplotlib.pyplot as plt
import csv

rootdir = "/Users/jasondong/Desktop/ARIndepStudy/dataFolder/Trial 10"

def getPictureDetails():
    laplacianArr = []
    cornersArr = []
    entropyArr = []
    goEntropyArr = []
    userInput = []

    for dist in sorted(os.listdir(rootdir)):
        path = os.path.join(rootdir, dist)
        for directory in sorted(os.listdir(path)):
            dirPath = os.path.join(path, directory)
            filePath = os.path.join(dirPath, "position1.png")
            img_clr = cv2.imread(filePath)
            cv2.imshow('image', img_clr)

            cropped_image = img_clr[210:1200, 0:1200]
            cv2.imshow("cropped", cropped_image)
            # cv2.waitKey(0)
            # Calculate brightness
            brightness = (np.mean(cropped_image)) / 255

            # Calculate contrast
            contrast = cropped_image.std()

            # Calculate entropy
            entropy = skimage.measure.shannon_entropy(cropped_image)
            entropyArr.append(entropy)

            # Calculate GO Entropy
            sobel_h = cv2.Sobel(cropped_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_v = cv2.Sobel(cropped_image, cv2.CV_64F, 0, 1, ksize=3)
            orien = cv2.phase(np.array(sobel_h, np.float32), np.array(sobel_v, dtype=np.float32), angleInDegrees=True)
            orien_entropy = skimage.measure.shannon_entropy(orien)
            goEntropyArr.append(orien_entropy)

            # Calculate variance of the laplacian
            laplacian = cv2.Laplacian(cropped_image, cv2.CV_64F).var()
            laplacianArr.append(laplacian)

            # Calculate corners
            fast = cv2.FastFeatureDetector_create()
            kp = fast.detect(img_clr, None)
            corners = len(kp)
            cornersArr.append(corners)

            # print("Directory: " + dirPath + " Image - Brightness: " + str(brightness) + " Contrast: " + str(contrast) + " Entropy: " + str(entropy) + " Laplacian: " + str(laplacian) + " Corners: " + str(corners))
            print("Directory: " + dirPath + " Entropy: " + str(entropy) + " GoEntropy: " + str(orien_entropy))

            filePath = os.path.join(dirPath, "userInput.txt")
            if os.path.isfile(filePath):
                with open(filePath, 'r') as f:
                    userInput.append(f.read().replace("\n", ""))

    # plotParams(userInput, laplacianArr, "User Input", "Laplacian", "Graph of User Input vs. Laplacian")
    # plotParams(userInput, cornersArr, "User Input", "Corners", "Graph of User Input vs. Corners")

    with open('temp.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the laplacian
        writer.writerow(entropyArr)

        # write the corners
        writer.writerow(goEntropyArr)



def parseData():
    positionInWorldSpace = []
    distanceFromCameraToObject = []
    changeInViewingAngle = []
    magnitudeOfDrift = []
    userInput = []

    for dist in sorted(os.listdir(rootdir)):
        path = os.path.join(rootdir, dist)
        for directory in sorted(os.listdir(path)):
            dirPath = os.path.join(path, directory)
            # print(dirPath)

            # checking if it is a file, read position1.txt
            filePath = os.path.join(dirPath, "position1.txt")
            if os.path.isfile(filePath):
                with open(filePath, 'r') as f:
                    dataArr = []
                    dataArr.append(f.read())
                    for data in dataArr:
                        splitArr = data.split("\n")
                        positionInWorldSpace.append(splitArr[0])
                        distanceFromCameraToObject.append(splitArr[1])
                        changeInViewingAngle.append(splitArr[2])
                        magnitudeOfDrift.append(splitArr[3])

            #checking if it is a file, read userInput.txt
            filePath = os.path.join(dirPath, "userInput.txt")
            if os.path.isfile(filePath):
                with open(filePath, 'r') as f:
                    userInput.append(f.read().replace("\n", ""))


    print("Position in world space: " + str(positionInWorldSpace))
    print("Distance from Camera to Object: " + str(distanceFromCameraToObject))
    print("Change in viewing angle: " + str(changeInViewingAngle))
    print("Magnitude of drift: " + str(magnitudeOfDrift))
    print("User Input: " + str(userInput))

    #Plotting
    plotParams(userInput, magnitudeOfDrift, "User Input", "Magnitude of Drift", "Graph of User Input vs. Magnitude of Drift (All Trials)")

    userInputArr = []
    for directory in sorted(os.listdir(rootdir)):
        dirPath = os.path.join(rootdir, directory)
        print(dirPath)
        filePath = os.path.join(dirPath, "userInput.txt")
        if (os.path.isfile(filePath)):
            with open(filePath, 'r') as f:
                userInputArr.append(f.read())

    with open('temp.csv', 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the laplacian
        writer.writerow(userInputArr)


def plotParams(xaxis, yaxis, xlabel, ylabel, title):
    int_x = [int(numeric_string) for numeric_string in xaxis]
    float_y = [float(numeric_string) for numeric_string in yaxis]

    plt.scatter(int_x, float_y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    os.system("find . -name '.DS_Store' -type f -delete") #Delete .DS_Store files that come from macOS
    parseData()
    getPictureDetails()


