import cv2
import numpy as np
img = cv2.imread('./findContours.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
kernel = np.ones((7,7))
opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)

image,contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
#cv2.drawContours(img,contours,-1,(0,0,255),3) 

for cnt in contours:
    minAreaRect = cv2.minAreaRect(cnt)
    # 将浮点数坐标转换成整数
    rectCnt = np.int64(cv2.boxPoints(minAreaRect))
    cv2.drawContours(img, [rectCnt], 0, (0,255,0), 3)

'''
for i in contours:
    mrect = cv2.minAreaRect(i)
    rect = cv2.boxPoints(mrect)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print('RECT: x={}, y={}, w={}, h={}'.format(x, y, w, h))
'''
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




  
