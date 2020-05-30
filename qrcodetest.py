
import cv2 as cv

img = cv.imread('C:\\tmp\\s2.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 高斯模糊
gaussian_bulr = cv.GaussianBlur(gray, (5, 5), 0)
# 边缘检测,灰度值小于2参这个值的会被丢弃，大于3参这个值会被当成边缘，在中间的部分，自动检测
edged = cv.Canny(gaussian_bulr, 50, 400)
# 寻找轮廓
cts, hierarchy = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# 给轮廓加标记，便于我们在原图里面观察，注意必须是原图才能画出红色，灰度图是没有颜色的
cv.drawContours(img, cts, -1, (0, 0, 255), 3)
# 按面积大小对所有的轮廓排序
list = sorted(cts, key=cv.contourArea, reverse=True)
cv.imshow("draw_contours", img)
print("寻找轮廓的个数：", len(cts))


cv.waitKey(0)
