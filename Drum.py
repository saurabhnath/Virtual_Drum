# organized imports
import cv2
import numpy as np
from susic import kick
from susic import snare
from susic import hihat
from contour import findContour

print("Start your virtual drum, created by saurav")

# color to detect - drum stick
lower = [17, 15, 100]
upper = [80, 76, 220]

# region coordinates
k_top, k_bottom, k_right, k_left = 180, 200, 540, 640
h_top, h_bottom, h_right, h_left = 140, 240, 300, 400
s_top, s_bottom, s_right, s_left = 140, 240, 750, 850

#bool for each drum
e_snare = 0
e_kick = 0
e_hihat = 0

#-------------------------------
# main function
#--------------------------------

if __name__ == "__main__":
    #accumulated weight
    aWeight = 0.5

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 60)

    while True:
        status, frame = cam.read()
        clone = frame.copy()
        clone = cv2.flip(clone, 1)

        #get the three drum region
        reg_kick = clone[k_top:k_bottom, k_right:k_left]
        reg_hihat = clone[h_top:h_bottom, h_right:h_left]
        reg_snare = clone[s_top:s_bottom, s_right:s_left]

        #blur the regions
        reg_kick = cv2.GaussianBlur(reg_kick, (7, 7), 0)
        reg_hihat = cv2.GaussianBlur(reg_hihat, (7, 7), 0)
        reg_snare = cv2.GaussianBlur(reg_snare, (7, 7), 0)

        l = np.array(lower, dtype="uint8")
        u = np.array(upper, dtype="uint8")

        mask_kick = cv2.inRange(reg_kick, l, u)
        mask_hihat = cv2.inRange(reg_hihat, l, u)
        mask_snare = cv2.inRange(reg_snare, l, u)

        out_kick = cv2.bitwise_and(reg_kick, reg_kick, mask=mask_kick)
        out_hihat = cv2.bitwise_and(reg_hihat, reg_hihat, mask=mask_hihat)
        out_snare = cv2.bitwise_and(reg_snare, reg_snare, mask=mask_snare)

        cnts_kick = findContour(out_kick)
        cnts_hihat = findContour(out_hihat)
        cnts_snare = findContour(out_snare)

        if (cnts_kick > 0) and (e_kick == 0):
            kick()

            e_kick = 1
        elif cnts_kick == 0:
            e_kick = 0

        if (cnts_hihat > 0) and (e_hihat == 0):
            hihat()
            e_hihat = 1
        elif cnts_hihat == 0:
            e_hihat = 0

        if (cnts_snare > 0) and (e_snare == 0):
            snare()
            e_snare = 1
        elif cnts_snare == 0:
            e_snare = 0

         # draw the drum regions
        cv2.rectangle(clone, (k_left, k_top), (k_right, k_bottom), (0, 255, 0, 0.5), 5)
        cv2.rectangle(clone, (h_left, h_top), (h_right, h_bottom), (255, 0, 0, 0.5), 5)
        cv2.rectangle(clone, (s_left, s_top), (s_right, s_bottom), (0, 0, 255, 0.5), 5)

        cv2.imshow("video", clone)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
