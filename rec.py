import cv2
import numpy as np
from operator import itemgetter
from math import atan2, pi

# PARAMETERS TO CHANGE AROUND
filename = 'input.png'
distance = 25 # What should this be?
mark_color_hsv = (0, 255, 255) # Color of the marks in HSV format

img = cv2.imread(filename)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Find pixels that have specified color in the image
mask = cv2.inRange(hsv, mark_color_hsv, mark_color_hsv)

## slice the red
imask = mask>0
red = np.zeros_like(img, np.uint8)
red[imask] = img[imask]

## save
cv2.imwrite("red mask.png", red)

_, _, gray = cv2.split(red)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("thresh.png", thresh)

centers = []
contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
# Find and record the center of each contour
for c in contours:
    mom = cv2.moments(c)
    c_x = int(mom["m10"] / mom["m00"])
    c_y = int(mom["m01"] / mom["m00"])
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    centers.append((c_x, c_y))

cv2.imwrite("contours.png", img)
print("Found", len(centers), "marked shots.")

#(0, 0) is at top left corner, not bottom left
#increases positive downwards and right
centers.sort(key=itemgetter(1), reverse=True)
recoil = []
for i in range(len(centers)-1):
    x1, y1 = centers[i]
    x2, y2 = centers[i+1]
    img = cv2.line(img, centers[i], centers[i+1], (255, 0, 0), 2)

    # Negative sign in front of delta y because y increases going down
    # but we want the angle in sane coords
    hrec = atan2(x2-x1, distance)
    vrec = atan2(y1-y2, distance)
    recoil.append((hrec, vrec))

cv2.imwrite("lines.png", img)
# write recoil values to a file
with open('recoil.csv', 'w') as f:
    f.write("HREC,VREC\n")
    i = 0
    for hrec, vrec in recoil:
        # This is Kovaak's format
        f.write("PSR{}={},{}\n".format(i, vrec, hrec))
        i += 1
