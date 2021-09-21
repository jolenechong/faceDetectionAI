# data structures for creating image arrays
import  numpy as np
import matplotlib.pyplot as plt
import cv2


# bw_arr = np.array([
#     # each of this represents one pixel
#     [0,0,0],
#     [100,100,100],
#     [255,255,255]
# ])

# colorful_arr = np.array([
#     # each of this represents one pixel
#     # notice the double square
#     [[255,0,0],[0,255,0]],
#     [[0,0,0],[0,0,255]]
# ])

# # plt.imshow(bw_arr, cmap="gray")
# plt.imshow(colorful_arr)
# plt.show()

# img_bgr = cv2.imread("face.jpg")
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# # cv2.waitKey()
# plt.imshow(img_rgb)
# # print the pixels thingy of an image
# print(img_rgb)

# # giving video feedback
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# while True:
#     ret, frame = cap.read()
#     cv2.imshow("face camera", frame)
#     print(frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# facial detection
# use cascade classifier, reads as blue red format
img = cv2.imread('face.jpg')
face_cascade = cv2.CascadeClassifier('haarcascade frontalface default.xml')
# must convert to greyscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #bgr blue green red to grey
boxes = face_cascade.detectMultiScale(gray_img, 1.3 ,5)

for box in boxes:
    x,y,width, height = box
    # x and y are not always at the 0 coordinate
    cv2.rectange(img, (x,y), (x+width,y+height), (255,0,0),2)

cv2.imshow("detection", img)
cv2.waitKey()