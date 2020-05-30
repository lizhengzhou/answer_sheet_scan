
from imutils.perspective import four_point_transform
import numpy as np
import cv2 as cv

# 加载一个图片到opencv中
img = cv.imread('C:\\tmp\\r2.jpg')
# 转化成灰度图片
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gaussian_bulr = cv.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
edged = cv.Canny(gaussian_bulr, 40, 200)  # 边缘检测,灰度值小于2参这个值的会被丢弃，大于3参这个值会被当成边缘，在中间的部分，自动检测
# 寻找轮廓
cts, hierarchy = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# 给轮廓加标记，便于我们在原图里面观察，注意必须是原图才能画出红色，灰度图是没有颜色的
cv.drawContours(img, cts, -1, (0, 0, 255), 3)
# 按面积大小对所有的轮廓排序
list = sorted(cts, key=cv.contourArea, reverse=True)
cv.imshow("draw_contours", img)
print("寻找轮廓的个数：", len(cts))

area=[]
arc = []

for c in list:
    # 周长，第1个参数是轮廓，第二个参数代表是否是闭环的图形
    peri = 0.01 * cv.arcLength(c, True)
    # 获取多边形的所有定点，如果是四个定点，就代表是矩形
    approx = cv.approxPolyDP(c, peri, True)
    # 打印定点个数
    # print("顶点个数：", len(approx))

    if len(approx) == 4 and peri<5 :  # 矩形
        arc.append(approx)
        area.append(c)
        # print(approx)
        # print(peri)



cv.drawContours(img, area, -1, (0, 255, 0), 3)
cv.imshow("draw_contours", img)

print(arc)

rect = []

for a in arc:
    rect.append(a[0])

print(rect)

# 透视变换提取原图内容部分
ox_sheet = four_point_transform(img, np.array(rect).reshape(4, 2))
# 透视变换提取灰度图内容部分
tx_sheet = four_point_transform(gray,np.array(rect).reshape(4, 2))

cv.imshow("ox", ox_sheet)
cv.imshow("tx", tx_sheet)



cv.waitKey(0)