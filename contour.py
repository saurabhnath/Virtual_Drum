import cv2

def findContour(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)[1]
    cntr, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(cntr)