import numpy as np
import pandas as pd
import cv2
import Tkinter as tk
from PIL import Image, ImageTk
import dlib
from tkFileDialog import askopenfilename
import os
import random
import string

detector = dlib.get_frontal_face_detector()  # Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark identifier.

# Initialize some variables:
PythonDir = os.getcwd()
makeup_template_path = '\\'.join([PythonDir, 'MakeupTemplate'])
pic_makeup_loc = '\\'.join([makeup_template_path, '6.jpg'])
# ImageList
ImageList=[]
for file in os.listdir(makeup_template_path):
    if file.endswith(".jpg"):
        ImageList.append(os.path.join(makeup_template_path, file))

modified_frame = pic_makeup = cv2.imread(pic_makeup_loc,1)
EndImg = cv2.imread(PythonDir+'\\End.png',1)


#-------------------- Functions ----------------------


def FaceLandmarkPoints (img): # img in BGR
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rects = detector(img_gray, 0)
        for img_rect in img_rects:
            shape_img_gray = predictor(img_gray, img_rect)
            xlist = []
            ylist = []
            for i in range(0, 68):  # There are 68 landmark points on each face
                xlist.append(float(shape_img_gray.part(i).x))
                ylist.append(float(shape_img_gray.part(i).y))
        return np.array(zip(xlist,ylist))
    except (UnboundLocalError, type(img.shape)==None):
        pass


def FaceMask(FaceImage, LandmarkPoints):
    try:
        OuterPoints = []
        for i in range(26, 16, -1): # add point 17 to 26 in descending order
            OuterPoints.append([LandmarkPoints[i][0], LandmarkPoints[i][1]])
        for i in range(0, 17): # add points 0 to 16
            OuterPoints.append([LandmarkPoints[i][0],LandmarkPoints[i][1]])
        OuterPoints = np.array(OuterPoints, np.int32)
        mask = np.zeros_like(FaceImage) # create empty mask
        cv2.fillPoly(mask, [OuterPoints], [255, 255, 255])  # fill polygon at the empty mask
        return mask
    except TypeError:
        pass


def MouthMask_Outer(FaceImage, LandmarkPoints):
    try:
        OuterPoints = []
        for i in range(48, 60):
            OuterPoints.append([LandmarkPoints[i][0], LandmarkPoints[i][1]])
        OuterPoints = np.array(OuterPoints, np.int32)
        mask = np.zeros_like(FaceImage) # create empty mask
        # cv2.fillPoly(mask, [InnerPoints], [0, 0, 0])  # fill polygon at the empty mask
        cv2.fillPoly(mask, [OuterPoints], [255,255,255])# fill polygon at the empty mask
        return mask
    except TypeError:
        pass


def MouthMask_Inner(FaceImage, LandmarkPoints):
    try:
        InnerPoints = []
        for i in range(60, 68):
            InnerPoints.append([LandmarkPoints[i][0], LandmarkPoints[i][1]])
        InnerPoints = np.array(InnerPoints, np.int32)
        mask = np.zeros_like(FaceImage)  # create empty mask
        cv2.fillPoly(mask, [InnerPoints], [255, 255, 255])
        return mask
    except TypeError:
        pass


def EyesMask(FaceImage, LandmarkPoints):
    try:
        LeftPoints = []
        for i in range(36, 42): # add point 17 to 26 in descending order
            LeftPoints.append([LandmarkPoints[i][0], LandmarkPoints[i][1]])
        RightsPoints = []
        for i in range(42, 48): # add points 0 to 16
            RightsPoints.append([LandmarkPoints[i][0],LandmarkPoints[i][1]])
        LeftPoints = np.array(LeftPoints, np.int32)
        RightsPoints = np.array(RightsPoints, np.int32)
        mask = np.zeros_like(FaceImage)  # create empty mask
        cv2.fillPoly(mask, [LeftPoints], [255,255,255])  # fill polygon at the empty mask
        cv2.fillPoly(mask, [RightsPoints], [255,255,255])  # fill polygon at the empty mask
        return mask
    except TypeError:
        pass


def applyAffineTransform(src, srcTri, dstTri, size):
    """
    Apply affine transform calculated using srcTri and dstTri to src and output an image of size
    """
    # Given a pair of triangles, find the affine transform matrix.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img
    """
    #-------- make sure catch the errors when some part of face is out of the frame!---------
    try:
        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        r = cv2.boundingRect(np.float32([t]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        tRect = []
        for i in xrange(0, 3):
            tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        # Get mask by filling triangle
        mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);
        # Apply warpImage to small rectangular patches
        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
        size = (r[2], r[3])
        warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
        warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)
        # Alpha blend rectangular patches
        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
        # Copy triangular region of the rectangular patch to the output image
        img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask
    except (type(img1)==None) | (type(img2)==None):
        pass


def MorphingImg(img1, img2, img1_LandMarks, img2_LandMarks, alpha, triangles):
    img1_ = np.float32(img1)
    img2_ = np.float32(img2)
    # Allocate space for final output
    imgMorph = np.zeros(img1_.shape, dtype=img2_.dtype)
    try:
        for i in range(0, len(triangles)):
            x = triangles.iloc[i][0];
            y = triangles.iloc[i][1];
            z = triangles.iloc[i][2]  # get 3 conners of triangle
            t1 = [img1_LandMarks[x], img1_LandMarks[y], img1_LandMarks[z]]
            t2 = [img2_LandMarks[x], img2_LandMarks[y], img2_LandMarks[z]]
            t = t1
            # Morph one triangle at a time.
            morphTriangle(img1_, img2_, imgMorph, t1, t2, t, alpha)

    except (ValueError, TypeError):
        pass
    return np.uint8(imgMorph)


def adjust_gamma(image, gamma=1.0):  # adjust plain picture illumination
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)


def ModifyFace(frame, alpha_face, alpha_lips, gamma_brightness):  # Modify face
    originalFrame = frame
    frame_LandMarks = FaceLandmarkPoints(frame)
    makeup_LandMarks = FaceLandmarkPoints(pic_makeup)
    if (type(frame_LandMarks)!= None)|(type(makeup_LandMarks)!= None):
        try:
            mask = FaceMask(frame, frame_LandMarks) - EyesMask(frame, frame_LandMarks)
            faceMorph = MorphingImg(frame, pic_makeup, frame_LandMarks, makeup_LandMarks, alpha_face, triangle_Face_NoEyesLips)
            lipMorph = MorphingImg(frame, pic_makeup, frame_LandMarks, makeup_LandMarks, alpha_lips, triangle_Lips)

            ### Adjust Intensity
            frame = adjust_gamma(frame, gamma=gamma_brightness)

            ### Seamless Cloning
            face_center = tuple(((np.max(frame_LandMarks, axis=0) + np.min(frame_LandMarks, axis=0)) / 2).astype(int))
            normal_clone = faceMorph+lipMorph
            normal_clone[np.where(((faceMorph != [0, 0, 0])&(lipMorph != [0, 0, 0])).all(axis=2))] = \
                lipMorph[np.where(((faceMorph != [0, 0, 0])&(lipMorph != [0, 0, 0])).all(axis=2))]
            normal_clone = cv2.seamlessClone(normal_clone, frame, mask, face_center, cv2.NORMAL_CLONE)

            # add inner mouth
            inner_mouth = cv2.bitwise_and(frame, MouthMask_Inner(frame, frame_LandMarks))
            normal_clone = normal_clone + inner_mouth
            normal_clone[np.where(((inner_mouth != [0, 0, 0])&(lipMorph != [0, 0, 0])).all(axis=2))] = \
                lipMorph[np.where(((inner_mouth != [0, 0, 0])&(lipMorph != [0, 0, 0])).all(axis=2))]

            # Gaussian blur edges of lips and inner mouth
            inner_mouth_edge_mask = MouthMask_Inner(normal_clone, frame_LandMarks)
            inner_mouth_edge_mask = inner_mouth_edge_mask - cv2.erode(inner_mouth_edge_mask , np.ones((5,5), np.uint8), iterations=1)

            normal_clone[np.where((inner_mouth_edge_mask != [0, 0, 0]).all(axis=2))] = cv2.GaussianBlur(normal_clone, (5, 5), 11)[
                np.where((inner_mouth_edge_mask != [0, 0, 0]).all(axis=2))]

            normal_clone = PasteOriginal(originalFrame, normal_clone, 0.3)  # paste the original frame on the makeup added frame
            return normal_clone
        except (cv2.error, AssertionError, TypeError, ValueError):
            return PasteOriginal(originalFrame, frame, 0.3)
    else:
        return frame


'''
# Crop image read from webcam
def CropImg(frame, LandmarkPoints):
    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)  # resize frame to reduce calculation
    # find the most top, bottom, left and right points of LandmarkPoints
    Left = np.min(LandmarkPoints, axis=0)[0]
    Right = np.max(LandmarkPoints, axis=0)[0]
    Top = np.min(LandmarkPoints, axis=0)[1]
    Bottom = np.max(LandmarkPoints, axis=0)[1]
    FaceWidth = Right - Left
    FaceHight = Bottom - Top

    x = int(max(Left - 0.5*FaceWidth, 0))
    w = int(FaceWidth*2.0)
    y = int(max(Top - FaceHight, 0))
    h = int(FaceHight*3)
    frame = frame[y:y+h, x:x+w]

    return cv2.flip(frame, 1)'''


def PasteOriginal(original, processed, shrinkFactor):
    """
    Paste original resizes image on the processed image. The shrinkFactor is in range [0,1]
    """
    Original_Resize = cv2.resize(original, (0, 0), fx=shrinkFactor, fy=shrinkFactor)
    l0,w0,c0 = processed.shape
    l1,w1,c1 = Original_Resize.shape
    processed[(l0-l1):l0, (w0-w1):w0, :] = Original_Resize
    return processed


def ResizeTemplate(Hight, template):
    """
    Resize image to desired hight, without changing the ratio of dimension
    """
    h, w, c = template.shape
    ResizeFactor = 1.0*Hight / h
    template = cv2.resize(template, (0, 0), fx=ResizeFactor, fy=ResizeFactor)
    return template


#-------------------- Import triangles  --------------------


triangle_Face_NoEyesLips = pd.read_csv('\\'.join([PythonDir, 'triangle_Face_NoEyesLips.csv']), header=None)
triangle_Lips = pd.read_csv('\\'.join([PythonDir, 'triangle_lips.csv']), header=None)


#########################################################################
#                                Setup GUI                              #
#########################################################################
# --------------------- Graphics window ---------------------
window = tk.Tk()  #Makes main window
window.wm_title("Beautify your face")
window.config(background="light grey")


def SelectMakeup():
    global pic_makeup_loc
    global pic_makeup
    global ImageList
    pic_makeup_loc = askopenfilename()
    pic_makeup = cv2.imread(pic_makeup_loc, 1)
    pic_makeup = ResizeTemplate(200, pic_makeup)

# ------------------ setup frame to display makeup template --------------
# Initialize makeup tamplate display frames
templateFrame1 = tk.Frame(window, height=200)
templateFrame1.grid(row=1, column=1, rowspan=5, columnspan=3)
tmain1 = tk.Label(templateFrame1)
tmain1.grid(row=0, column=0)
template1 = cv2.imread(pic_makeup_loc, 1)
timgtk1 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(ResizeTemplate(200, template1), cv2.COLOR_BGR2RGBA)))
tmain1.imgtk = timgtk1
tmain1.configure(image=timgtk1)

# set sliders -----------------
# alpha_face
tk.Label(window, text="Face Blend").grid(row=7, column=0, sticky=tk.E)
AlphaFace = tk.Scale(window, from_=0, to=100, length=400, orient=tk.HORIZONTAL, showvalue=False)
AlphaFace.grid(row=7, column=1, padx=10, pady=2, columnspan=3)
AlphaFace.set(50)
# alpha_lips
tk.Label(window, text="Lips Blend").grid(row=8, column=0, sticky=tk.E)
AlphaLips = tk.Scale(window, from_=0, to=100, length=400, orient=tk.HORIZONTAL, showvalue=False)
AlphaLips.grid(row=8, column=1, padx=10, pady=2, columnspan=3)
AlphaLips.set(50)
# gamma_brightness
tk.Label(window, text="Brightness").grid(row=9, column=0, sticky=tk.E)
Gamma = tk.Scale(window, from_=0, to=100, length=400, orient=tk.HORIZONTAL, showvalue=False)
Gamma.grid(row=9, column=1, padx=10, pady=2, columnspan=3)
Gamma.set(50)

textImg = tk.PhotoImage(file="Text_bg.gif")
TextDisplay = tk.Label(window, image=textImg, text="Select Template", compound=tk.CENTER, font=("Helvetica", 16))
TextDisplay.grid(row=0, column=2)


# get all image files under the same path as pic_makeup -------------------


def GetAllImgsFromPath():
    global pic_makeup_loc
    global makeup_template_path
    global ImageList
    makeup_template_path1, filename = os.path.split(pic_makeup_loc)
    if makeup_template_path1!= makeup_template_path:
        makeup_template_path = makeup_template_path1
        ImageList=[]
        for file in os.listdir(makeup_template_path):
            if file.endswith(".jpg"):
                ImageList.append(makeup_template_path.replace('\\', '/') + '/' + file)


def DisplayTemplate():
    """
    change the displayed template
    """
    global pic_makeup_loc
    global ImageList
    global pic_makeup
    currentIndex = ImageList.index(pic_makeup_loc)

    if currentIndex == 0:
        displayImg = np.concatenate((EndImg,
                                     ResizeTemplate(EndImg.shape[0], cv2.imread(ImageList[currentIndex],1)),
                                     ResizeTemplate(EndImg.shape[0], cv2.imread(ImageList[currentIndex+1],1))), axis=1)
    elif currentIndex == len(ImageList)-1:
        displayImg = np.concatenate((ResizeTemplate(EndImg.shape[0], cv2.imread(ImageList[currentIndex-1],1)),
                                     ResizeTemplate(EndImg.shape[0], cv2.imread(ImageList[currentIndex],1)),
                                     EndImg), axis=1)
    else:
        displayImg = np.concatenate((ResizeTemplate(EndImg.shape[0], cv2.imread(ImageList[currentIndex-1],1)),
                                     ResizeTemplate(EndImg.shape[0], cv2.imread(ImageList[currentIndex],1)),
                                     ResizeTemplate(EndImg.shape[0], cv2.imread(ImageList[currentIndex+1], 1))), axis=1)

    timgtk1 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(displayImg, cv2.COLOR_BGR2RGBA)))
    tmain1.imgtk = timgtk1
    tmain1.configure(image=timgtk1)
    # Change text label
    TextDisplay = tk.Label(window, image=textImg, text="Beautify Your Face", compound=tk.CENTER, font=("Helvetica", 14))
    TextDisplay.grid(row=0, column=2)


def GoPrevious():
    """
    move to previous template
    """
    global pic_makeup_loc
    global ImageList
    currentIndex = ImageList.index(pic_makeup_loc)
    if currentIndex > 0:
        pic_makeup_loc = ImageList[currentIndex-1]


def GoNext():
    """
    move to next template
    """
    global pic_makeup_loc
    global ImageList
    currentIndex = ImageList.index(pic_makeup_loc)
    if currentIndex < len(ImageList)-1:
        pic_makeup_loc = ImageList[currentIndex+1]


def SavePhoto():
    """
    save the photo
    """
    try:
        PhotoDir = '\\'.join([PythonDir, 'Photos'])
        PhotoName = ''.join(random.choice(string.digits) for _ in range(9)) + '.jpg'
        cv2.imwrite(os.path.join(PhotoDir, PhotoName), modified_frame)
        TextDisplay = tk.Label(window, image=textImg, text="Photo Saved", compound=tk.CENTER,
                               font=("Helvetica", 16))
        TextDisplay.grid(row=0, column=2)
    except TypeError:
        pass


def ConfirmSelection():
    """
    confirm selection of template
    """
    global pic_makeup_loc
    global pic_makeup
    pic_makeup = cv2.imread(pic_makeup_loc, 1)


# ------------------- Setup webcam -------------------------
cap = cv2.VideoCapture(0)  # Enable video camera
if not cap.isOpened():
    exit('The Camera is not opened')

# ------------------- setup frame to display video ------------
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=4, padx=10, pady=2, rowspan=10)
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)


def ShowWebcam():
    global modified_frame
    _, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
    frame = cv2.flip(frame, 1)

    alpha_face = AlphaFace.get()*1.0/100
    alpha_lips = AlphaLips.get()*1.0/100
    gamma_brightness = Gamma.get()*1.0/50+0.1 # gamma cannot be 0

    img = ModifyFace(frame, alpha_face, alpha_lips, gamma_brightness)
    modified_frame = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(5, ShowWebcam)


# ---------------- Setup buttons -----------------
buttonImg_Brow = tk.PhotoImage(file="Button_Brow.gif")
tk.Button(window, image=buttonImg_Brow,  command=lambda:[SelectMakeup(),GetAllImgsFromPath(),DisplayTemplate()]).grid(row=0, column=1, padx=10, pady=2)

buttonImg_Shot = tk.PhotoImage(file="Button_Shot.gif")
tk.Button(window, image=buttonImg_Shot, command=lambda:[SavePhoto()]).grid(row=0, column=3, padx=10, pady=2)

buttonImg_Prev = tk.PhotoImage(file="Button_Prev.gif")
tk.Button(window, image=buttonImg_Prev, command=lambda:[GoPrevious(), DisplayTemplate()]).grid(row=6, column=1, padx=10, pady=2)

buttonImg_Sel = tk.PhotoImage(file="Button_Sel.gif")
tk.Button(window, image=buttonImg_Sel, command=lambda:[ConfirmSelection()]).grid(row=6, column=2, padx=10, pady=2)

buttonImg_Next = tk.PhotoImage(file="Button_Next.gif")
tk.Button(window, image=buttonImg_Next, command=lambda:[GoNext(), DisplayTemplate()]).grid(row=6, column=3, padx=10, pady=2)


DisplayTemplate()
ShowWebcam()
window.mainloop()
