import tkinter as tk
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import numpy as np
import math
import time


def zoSuboru(subor):
    matica = []
    riadokSuboru = subor.readline()

    while riadokSuboru:

        cislaRiadkuSuboru = []
        for i in riadokSuboru.split(","):
            cislaRiadkuSuboru.append(i)
        matica.append(cislaRiadkuSuboru)
        riadokSuboru = subor.readline()
    subor.close()
    return matica


def openImage():
    global img
    global imgHistEq
    global imgGaussBlur
    global imgCannyEdge
    global imgHoughTran
    global path
    path = filedialog.askopenfilename()
    print(path)
    if path:
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # cv2.circle(img, (x1, y1), r1, (255, 255, 0), 2)
        image = Image.fromarray(img)
        image_tk = ImageTk.PhotoImage(image)
        panelSRC.configure(image=image_tk)
        panelSRC.image = image_tk

    imgHistEq = img.copy()
    imgGaussBlur = img.copy()
    imgCannyEdge = img.copy()
    imgHoughTran = img.copy()


def histEq():
    global imgHistEq
    global imgGaussBlur
    global imgCannyEdge
    global imgHoughTran

    if imgHistEq.any():  # check whether image exists
        if varHistEq.get() == 1:
            image = cv2.equalizeHist(imgHistEq)

            imgGaussBlur = image.copy()
            imgCannyEdge = image.copy()
            imgHoughTran = image.copy()

            image = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image)

            panelSRC.configure(image=image_tk)
            panelSRC.image = image_tk
        else:
            imgGaussBlur = imgHistEq.copy()
            imgCannyEdge = imgHistEq.copy()
            imgHoughTran = imgHistEq.copy()

            image = Image.fromarray(imgHistEq)
            image_tk = ImageTk.PhotoImage(image)

            panelSRC.configure(image=image_tk)
            panelSRC.image = image_tk


def gausianBlur():
    global imgGaussBlur
    global imgCannyEdge
    global imgHoughTran

    if imgGaussBlur.any():  # check whether image exists
        kernel = 0
        sigma = 0
        if kernelGaus.get() != '':
            kernel = int(kernelGaus.get())
        if sigmaGaus.get() != '':
            sigma = int(sigmaGaus.get())
        image = cv2.GaussianBlur(imgGaussBlur, (kernel, kernel), sigma)

        imgCannyEdge = image.copy()
        imgHoughTran = image.copy()

        # img = image

        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)

        panelSRC.configure(image=image_tk)
        panelSRC.image = image_tk


def cannyEdge():
    global imgCannyEdge
    global imgHoughTran

    if imgCannyEdge.any():  # check whether image exists
        tresVal1 = 0
        tresVal2 = 0

        tresVal1 = int(treshold1.get())
        tresVal2 = int(treshold2.get())
        image = cv2.Canny(imgCannyEdge, tresVal1, tresVal2)

        imgHoughTran = image.copy()
        # img = image

        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)

        panelSRC.configure(image=image_tk)
        panelSRC.image = image_tk


def houghTransf():
    global imgHoughTran
    global circles
    if imgHoughTran.any():  # check whether image exists

        dp = int(dpAkumHoughTra.get())
        minDist = int(minDistHoughTra.get())
        param1 = int(param1HoughTra.get())
        param2 = int(param2HoughTra.get())
        minRadius = int(minRadiusHoughTra.get())
        maxRadius = int(maxRadiusHoughTra.get())
        circles = cv2.HoughCircles(imgHoughTran, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                   param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        circles = np.uint16(np.around(circles))
        image = cv2.cvtColor(imgHoughTran, cv2.COLOR_GRAY2BGR)
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        # img = image

        image = Image.fromarray(image)
        image_tk = ImageTk.PhotoImage(image)

        panelSRC.configure(image=image_tk)
        panelSRC.image = image_tk


def vypocetVsetko():
    kernel = 0
    sigma = 0
    if kernelGaus.get() != '':
        kernel = int(kernelGaus.get())
    if sigmaGaus.get() != '':
        sigma = int(sigmaGaus.get())

    dp = int(dpAkumHoughTra.get())
    minDist = int(minDistHoughTra.get())
    param1 = int(param1HoughTra.get())
    param2 = int(param2HoughTra.get())
    minRadius = int(minRadiusHoughTra.get())
    maxRadius = int(maxRadiusHoughTra.get())
    mat = zoSuboru(open("iris_annotation.csv"))
    poleVysledkov = []
    i = 0
    truePositivePrecision = 0
    falsePositivePrecision = 0

    for x in mat[1:]:
        # nacitanie
        print(i)
        try:
            img = cv2.imread('C:/Users/Luky/Documents/Skola/BIOM/duhovky/' + str(x[0]), cv2.COLOR_BGR2RGB)
            x1 = int(x[1])
            y1 = int(x[2])
            r1 = int(x[3])
            # gaus
            image = cv2.GaussianBlur(img, (kernel, kernel), sigma)
            # hough

            circlesAll = cv2.HoughCircles(imgHoughTran, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                                          param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
            circlesAll = np.uint16(np.around(circlesAll))
            # vypocet IoU
            precPreImg = []
            for cir in circlesAll[0, :]:
                if r1 > cir[2]:
                    iou = iouKruhov(x1, y1, r1, cir[0], cir[1], cir[2])
                    poleVysledkov.append(iou)
                    precPreImg.append(iou)
                else:
                    iou2 = iouKruhov(cir[0], cir[1], cir[2], x1, y1, r1)
                    poleVysledkov.append(iou2)
                    precPreImg.append(iou2)

            ### vypocet precision
            nasloAsponJedno = 0
            for pre in precPreImg:
                if pre >= 0.75:
                    if nasloAsponJedno == 0:
                        truePositivePrecision += 1
                        nasloAsponJedno += 1
                    else:
                        falsePositivePrecision += 1
                else:
                    falsePositivePrecision += 1
        except Exception:
            print('C:/Users/Luky/Documents/Skola/BIOM/duhovky/' + str(x[0]))
        i += 1
    time.sleep(2)
    print("TP")
    print(truePositivePrecision)
    print("FP")
    print(falsePositivePrecision)
    print("Precision")
    print(truePositivePrecision / (truePositivePrecision + falsePositivePrecision))
    print("Recall")
    falseNegative = 2655 - truePositivePrecision  # len pre zrenicku
    print(truePositivePrecision / (truePositivePrecision + falseNegative))
    print("_______________________")
    print(sum(poleVysledkov) / len(poleVysledkov))


def vypocet():
    global path
    global circles
    x1 = 0
    y1 = 0
    r1 = 0

    mat = zoSuboru(open("iris_annotation.csv"))
    for x in mat:
        if x[0] in str(path):
            x1 = int(x[1])
            y1 = int(x[2])
            r1 = int(x[3])
            break
    precPreImg = []
    for cir in circles[0, :]:
        if r1 > cir[2]:
            print(iouKruhov(x1, y1, r1, cir[0], cir[1], cir[2]))
            precPreImg.append(iouKruhov(x1, y1, r1, cir[0], cir[1], cir[2]))
        else:
            print(iouKruhov(cir[0], cir[1], cir[2], x1, y1, r1))
            precPreImg.append(iouKruhov(cir[0], cir[1], cir[2], x1, y1, r1))

    truePositive = 0
    falsePositive = 0
    nasloAsponJedno = 0
    for pre in precPreImg:
        if pre >= 0.75:
            if nasloAsponJedno == 0:
                truePositive += 1
                nasloAsponJedno += 1
            else:
                falsePositive += 1
        else:
            falsePositive += 1

    print(truePositive)
    print(falsePositive)
    print("Precision")
    print(truePositive / (truePositive + falsePositive))
    print("Recall")
    falseNegative = 1 - truePositive  # len pre zrenicku
    print(truePositive / (truePositive + falseNegative))


def iouKruhov(x1, y1, r1, x2, y2, r2):
    vzdialenost = math.sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))

    if vzdialenost >= r1 + r2:
        return 0
    elif vzdialenost + r2 <= r1:
        prienik = math.pi * pow(r2, 2)
        zjednotenie = math.pi * pow(r1, 2) + math.pi * pow(r2, 2) - prienik
        return prienik / zjednotenie
    else:
        d1 = (pow(r1, 2) - pow(r2, 2) + pow(vzdialenost, 2)) / (vzdialenost * 2)
        d2 = vzdialenost - d1
        prienikD1 = pow(r1, 2) * math.acos(d1 / r1) - d1 * math.sqrt((pow(r1, 2) - pow(d1, 2)))
        prienikD2 = pow(r2, 2) * math.acos(d2 / r2) - d2 * math.sqrt((pow(r2, 2) - pow(d2, 2)))
        prienik = prienikD1 + prienikD2
        zjednotenie = math.pi * pow(r1, 2) + math.pi * pow(r2, 2) - prienik
        return prienik / zjednotenie


root = tk.Tk()
root.geometry("1200x700")
root.title('toolbox')  # window title is toolbox

img = np.array([])  # set it to numpy array initially

w = tk.Label(root, text="Zadanie 1")
w.pack()
# otvorenie obrazku
btn = tk.Button(root, text="Otvor obrazok", command=openImage)
btn.pack(side="bottom", fill="both", padx="10", pady="10")

# histogramova ekvalizacia
varHistEq = tk.IntVar()
btnHisEq = tk.Checkbutton(root, text='Histogramova Ekvalizacia', variable=varHistEq, command=histEq)
btnHisEq.pack(side='top', padx=10, pady=10)

# Gaussove rozmazanie
tk.Label(root, text="Kernel Gaus").place(x=0, y=50)
kernelGaus = tk.Scale(root, from_=0, to=50, orient="horizontal")
kernelGaus.place(x=0, y=70)

tk.Label(root, text="Sigma Gaus").place(x=0, y=120)
sigmaGaus = tk.Scale(root, from_=0, to=20, orient="horizontal")
sigmaGaus.place(x=0, y=140)

btnGaus = tk.Button(root, text='Gaussian', command=gausianBlur)
btnGaus.place(x=0, y=190)

# Canyho hrany
tk.Label(root, text="Treshold 1").place(x=0, y=245)
treshold1 = tk.Scale(root, from_=0, to=300, orient="horizontal")
treshold1.place(x=0, y=265)

tk.Label(root, text="Treshold 2").place(x=0, y=305)
treshold2 = tk.Scale(root, from_=0, to=300, orient="horizontal")
treshold2.place(x=0, y=325)

btnCanny = tk.Button(root, text='Canny', command=cannyEdge)
btnCanny.place(x=0, y=375)

# Houghova transformacia

tk.Label(root, text="dp - akumulator").place(x=1000, y=50)
dpAkumHoughTra = tk.Scale(root, from_=1, to=10, orient="horizontal")
dpAkumHoughTra.place(x=1000, y=70)

tk.Label(root, text="Min vzdialenost kruhov").place(x=1000, y=120)
minDistHoughTra = tk.Scale(root, from_=0, to=50, orient="horizontal")
minDistHoughTra.place(x=1000, y=140)

tk.Label(root, text="Parameter 1").place(x=1000, y=190)
param1HoughTra = tk.Scale(root, from_=0, to=300, orient="horizontal")
param1HoughTra.place(x=1000, y=210)

tk.Label(root, text="Parameter 2").place(x=1000, y=260)
param2HoughTra = tk.Scale(root, from_=0, to=300, orient="horizontal")
param2HoughTra.place(x=1000, y=280)

tk.Label(root, text="Min radius").place(x=1000, y=330)
minRadiusHoughTra = tk.Scale(root, from_=0, to=100, orient="horizontal")
minRadiusHoughTra.place(x=1000, y=350)

tk.Label(root, text="Max radius").place(x=1000, y=400)
maxRadiusHoughTra = tk.Scale(root, from_=0, to=100, orient="horizontal")
maxRadiusHoughTra.place(x=1000, y=420)

btnHougTra = tk.Button(root, text='HOUGHOVA TRANSFORMÁCIA', command=houghTransf)
btnHougTra.place(x=1000, y=470)

# vypocet 1
vypocetBtn = tk.Button(root, text="Vypocet akt", command=vypocet)
vypocetBtn.pack(side="bottom", fill="both", padx="10", pady="10")

# vypocet 1
vypocetAllBtn = tk.Button(root, text="Vypocet vsetko", command=vypocetVsetko)
vypocetAllBtn.pack(side="bottom", fill="both", padx="10", pady="10")
# exit
Exit1 = tk.Button(root, text="Exit", command=root.destroy)

Exit1.pack(side="bottom", fill="both", padx="10", pady="10")

panelSRC = tk.Label(root)
panelSRC.place(x=450, y=150)

root.mainloop()

##### ZDROJE #####
# https://diego.assencio.com/?index=8d6ca3d82151bad815f78addf9b5c1c6#mjx-eqn-post_8d6ca3d82151bad815f78addf9b5c1c6_A_intersection
# https://stackoverflow.com/questions/65508255/image-editing-software-with-tkinter-and-opencv-and-how-do-make-a-button-that-re
