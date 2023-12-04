import cv2
import numpy as np
from imutils import contours
import myutils

# 读取数字模板
img = cv2.imread('template.png')

# 数字模板转为灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图进行二值化处理
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# 计算二值化处理过后的模板轮廓
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 对轮廓进行排序
#将模板的轮廓从左至右进行排序,将排序后的结果赋值给refCnts.sort_contour函数中，返回的结果是一个元组，其中包含了排序后的轮廓列表和排序的标志
#[0]表示获取元组中的第一个元素，也就是排序后的轮廓列表。
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

# 绘制模板的轮廓矩形： 分别提取refCnts中存储的轮廓，传入cv2.boundingRect（这里将存储的轮廓设置为c）x与y代表这个轮廓矩形左上角的坐标
# w代表轮廓矩形的宽度，h代表轮廓矩形的高度
#for c in refCnts:
#    (x, y, w, h) = cv2.boundingRect(c)
#    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0,255), 2)

digits = {}

#分别遍历refCnts中的轮廓索引(i)已经轮廓列表（c），使用boundingRect函数分别超出每一个轮廓的外接矩形，再进行切除（左上角为（x，y），右下角
#为（x+w，y+h）得到roi。设置roi大小为57*88，将roi存储到digits【】中
for (i,c) in enumerate(refCnts):
     (x,y,w,h) = cv2.boundingRect(c)
     roi = ref[y:y +h , x:x + w]
     roi = cv2.resize(roi,(57,88))

     digits[i] = roi


#处理图像

#初始化卷积核
rectKernal = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernal = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#读取银行卡
image = cv2.imread("creaditcard.jpg")

#设置银行卡大小
image = myutils.resize(image,width=300)

#将银行卡转为灰度图
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#进行顶帽操作，突出明亮的部分
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernal)

gradX = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
gradX = np.absolute(gradX)
(minVal,maxVal) = (np.min(gradX),np.max(gradX))
gradX = (255*((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")


#闭操作（先膨胀，再腐蚀）将数字连在一次
gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernal)

#进行二值化处理 cv2.threshold函数返回一个元组，其中第一个元素是计算得到的阈值，第二个元素是应用阈值后的输出图像。通过[1]可以取得输出图像。
thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#继续闭操作，将缝隙填充。
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernal)

#计算轮廓
threshCnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()

#绘制轮廓
#cv2.drawContours(cur_img,cnts,-1,(0,0,255),2)

locs = []

#遍历轮廓 需要自己根据实际银行卡设置不同的参数
for(i,c) in enumerate(cnts):
    #计算轮廓的外接矩形
    (x,y,w,h) = cv2.boundingRect(c)
    #计算矩形宽度与高度之比
    ar = w / float(h)

    #宽度与高度之比满足2.5-4.0
    if ar >2.5 and ar < 4.0:

        #宽度满足40-55 高度满足10-20
        if(w>40 and w<55) and (h>10 and h<20):

            #符合的留下来
            locs.append((x,y,w,h))

#将符合的轮廓从左到右排序
locs = sorted(locs,key = lambda  x:x[0])
output = []

#遍历每一个轮廓的数字
for(i,(gX,gY,gW,gH)) in enumerate(locs):
    groupOutput = []
    group = gray[gY-5:gY+gH+5,gX-5:gX+gW+5]

    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    digitCnts,hierarchy = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]

    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        scores = []

        for (digits_index,digitROI) in digits.items():
            result = cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
            (_,score,_,_) = cv2.minMaxLoc(result)
            scores.append(score)

        groupOutput.append(str(np.argmax(scores)))

    cv2.rectangle(image,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX,gY-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)

    output.extend(groupOutput)

print("Credit Card id:{}".format("".join(output)))
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()