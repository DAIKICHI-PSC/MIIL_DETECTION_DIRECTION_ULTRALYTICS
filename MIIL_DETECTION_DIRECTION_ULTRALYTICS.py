# -*- coding: utf-8 -*-

import os #OS関連処理用モジュールの読込
import sys #システム関連処理用モジュールの読込
import time #時間関連処理用モジュールの読込
import numpy as np #行列処理用モジュールの読込
import math as mt #各種計算用モジュールの読込
import cv2 #画像処理用モジュールの読込
from PySide6 import QtCore, QtGui, QtWidgets #GUI関連処理用モジュールの読込
from MIIL_DETECTION_DIRECTION_ULTRALYTICS_GUI import Ui_MainWindow #QT Designerで作成し変換したファイルの読込
from getRectanglePos import getRectanglePos #２点の何れかが選択領域の開始点（左上）になり、終点（左下）になるか判定し、さらに終点が指定した範囲にあるかるか確認するライブラリ

from ctypes import * #C言語処理用モジュールの読込
import random #乱数処理用モジュールの読込
import re #正規表現処理用モジュールの読込
import multiprocessing

from ultralytics import YOLO
import yaml

#####グローバル変数########################################
cap = 0 #キャプチャー画像取得用変数
capLoop = 0 #動画を表示中か判定するフラグ
camWidth = 0 #動画の横サイズ
camHeight = 0 #動画の縦サイズ
sStartFlag = 0 #領域選択開始フラグ
mX1 = 0 #マウスボタンを押した時の横方向の座標
mY1 = 0 #マウスボタンを押した時の縦方向の座標
mX2 = 0 #マウスボタンを離した時の横方向の座標
mY2 = 0 #マウスボタンを離した時の縦方向の座標
ssX = 0 #選択領域開始点（左上）の横方向座標
ssY = 0 #選択領域開始点（左上）の縦方向座標
sXL = 0 #選択領域終点（右下）の横方向座標
sYL = 0 #選択領域終点（右下）の縦方向座標

######フレームワーク以外のグローバル変数変数########################################
label_color ={} #各ラベルのカラーコード保存用ディクショナリ
color_code = 200 #カラーコード保存用変数
label_pos = {} #各ラベルのカラーパターン保存用ディクショナリ
color_pos = 0 #カラーパターン保存用変数
trimMode = 0 #トリムモード用フラグ
CapWidth = 320 #キャプチャー用Ｗｉｄｔｈ
CapHeight = 240 #キャプチャー用Ｈｅｉｇｈｔ
resizeWidth = 1280 #取得画像リサイズ用横サイズ
resizeHeight = 960 #取得画像リサイズ用縦サイズ
sensor_x = -1 #検出した領域が指定した座標にあるか確認用
sensor_y = -1 #検出した領域が指定した座標にあるか確認用
DirPath = "" #画像保存用パス
FileNum = 0 #画像ファイルネーム番号
outName = "" #外部出力するラベル名
outPut = 0 #外部出力するフラグ

model = ""
det_target = []

#####各種処理用関数########################################
#=====メインループ処理========================================
##########
#カメラから画像を取得し物体を認識
##########
#スタートボタンで開始
def mainLoop():

    global capLoop
    global camHeight
    global sStartFlag
    global label_color
    global label_pos
    global color_pos
    global color_code
    global MX1
    global MY1
    global FileNum
    global outPut
    thresh = float(win.ui.comboBox1.currentText())
    det_size = int(win.ui.comboBox8.currentText())
    while(True):
        ret, frame = cap.read() #カメラ画像を取得
        if ret == True and capLoop == 1: #カメラ画像取得成功かつループモードの場合
            frameB = np.copy(frame) #画像を画像にコピー
            if win.ui.checkBox3.isChecked() == True:
                frameB = cv2.resize(frameB, (resizeWidth, resizeHeight)) #画像サイズ変更
        #!!!!!!!!!!openCVの処理は此処で行う!!!!!!!!!!
            if capLoop == 1: #ループモードの場合

                if win.ui.lineEdit6.text() != '' and trimMode == 1:
                    frameB = frameB[int(win.ui.lineEdit7.text()):int(win.ui.lineEdit7.text()) + int(win.ui.lineEdit2.text()), int(win.ui.lineEdit6.text()):int(win.ui.lineEdit6.text()) + int(win.ui.lineEdit1.text())] #指定したサイズに画像をトリム
                if sStartFlag == 1: #####領域選択開始後の処理（赤枠を描画）
                    frameB = cv2.rectangle(frameB, (ssX, ssY), (sXL, sYL), (0, 0, 255), 1)

                if win.ui.checkBox4.isChecked() == False:
                    if len(det_target) > 0:
                        results = model.predict(source = frameB, classes = det_target, imgsz = det_size, conf = thresh, device = "intel:gpu", save = False,  project  = "", name = "", exist_ok = True)
                    else:
                        results = model.predict(source = frameB, imgsz = det_size, conf = thresh, device = "intel:gpu", save = False,  project  = "", name = "", exist_ok = True)
                    for result in results:
                        # 検出された各オブジェクトのバウンディングボックス、クラスID、信頼度を取得
                        boxes = result.boxes
                        for box in boxes:
                            # クラスID (数値) を取得
                            #CONFIDENCE = int((float(box.conf) + 0.05) * 10) / 10
                            class_id = int(box.cls[0])
                            LABEL = result.names[class_id]
                            TX = int(box.xyxy[0,0])
                            TY = int(box.xyxy[0,1])
                            BX = int(box.xyxy[0,2])
                            BY = int(box.xyxy[0,3])
                            if(LABEL in label_color) == False: #ラベル名がラベル色保存用ディクショナリにないか確認
                                label_color[LABEL] = color_code #ラベルに対する色を保存
                                label_pos[LABEL] = color_pos #ラベルに対するカラーパターンを保存
                                color_pos += 1
                                if color_pos == 6: #6パターン毎に色の明度を下げる
                                    color_pos = 0
                                    color_code -= 10
                            cv2.rectangle(frameB, (TX + 1, TY + 1), (BX, BY), (0, 0, 0), 1) #検出領域に枠の影を描画
                            cv2.rectangle(frameB, (TX, TY), (BX - 1, BY - 1), (256, 256, 256), 1) #検出領域に枠を描画
                            font_size = 1 #フォントサイズを指定
                            #pix_size = 10
                            font = cv2.FONT_HERSHEY_PLAIN #フォントを指定
                            if label_pos[LABEL] == 0: #パターン０の色設定
                                cv2.rectangle(frameB, (TX, TY - 20), (TX + 100, TY - 1), (label_color[LABEL], 0, 0), -1) #ラベル名描画領域を塗りつぶし
                            elif label_pos[LABEL] == 1: #パターン１の色設定
                                cv2.rectangle(frameB, (TX, TY - 20), (TX + 100, TY - 1), (0, label_color[LABEL], 0), -1) #ラベル名描画領域を塗りつぶし
                            elif label_pos[LABEL] == 2: #パターン２の色設定
                                cv2.rectangle(frameB, (TX, TY - 20), (TX + 100, TY - 1), (0, 0, label_color[LABEL]), -1) #ラベル名描画領域を塗りつぶし
                            elif label_pos[LABEL] == 3: #パターン３の色設定
                                cv2.rectangle(frameB, (TX, TY - 20), (TX + 100, TY - 1), (label_color[LABEL], label_color[LABEL], 0), -1) #ラベル名描画領域を塗りつぶし
                            elif label_pos[LABEL] == 4: #パターン４の色設定
                                cv2.rectangle(frameB, (TX, TY - 20), (TX + 100, TY - 1), (label_color[LABEL], 0, label_color[LABEL]), -1) #ラベル名描画領域を塗りつぶし
                            elif label_pos[LABEL] == 5: #パターン５の色設定
                                cv2.rectangle(frameB, (TX, TY - 20), (TX + 100, TY - 1), (0, label_color[LABEL], label_color[LABEL]), -1) #ラベル名描画領域を塗りつぶし
                            cv2.rectangle(frameB, (TX + 1, TY - 20), (TX + 100, TY - 1), (0, 0, 0), 1) #ラベル名描画領域に枠の影を描画
                            cv2.rectangle(frameB, (TX, TY - 21), (TX + 100, TY - 1), (256, 256, 256), 1) #ラベル名描画領域に枠を描画
                            cv2.putText(frameB, LABEL,(TX + 3, TY - 6), font, font_size, (0, 0, 0), 1) #ラベル名の影を描画
                            cv2.putText(frameB, LABEL,(TX + 2, TY - 7), font, font_size, (256, 256, 256), 1) #ラベル名を描画
                            if LABEL in outName:
                                if win.ui.checkBox2.isChecked() == True:
                                    if is_point_in_rectangle(sensor_x, sensor_y, TX, TY, BX, BY):
                                        outPut = 1
                                    else:
                                        outPut = 0
                                else:
                                    outPut = 1
                            else:
                                outPut = 0
                else:
                    cvKey = cv2.waitKey(1)
                    if cvKey == 32: ##########SPACE KEY##########
                        cv2.imwrite(DirPath + '/' + str(FileNum) + '.jpg', frameB)
                        if capLoop == 1:
                            font_size = 2
                            font = cv2.FONT_HERSHEY_PLAIN
                            cv2.putText(frameB, str(FileNum) + '.jpg SAVED.' , (5, 25), font, font_size,(0, 0, 255), 1)
                            app.processEvents()
                        FileNum += 1
                if win.ui.checkBox2.isChecked() == True and sensor_x >= 0: #画像保存モードの場合
                    cv2.line(frameB, (sensor_x - 10, sensor_y), (sensor_x + 10, sensor_y), (0, 0, 255), 3, cv2.LINE_8)
                    cv2.line(frameB, (sensor_x, sensor_y - 10), (sensor_x, sensor_y + 10), (0, 0, 255), 3, cv2.LINE_8)

            if capLoop == 1: #ループモードの場合
                cv2.imshow("MIIL DETECTION",frameB) #画像を表示
                cv2.setMouseCallback("MIIL DETECTION",onMouse) #画像に対するマウス入力を取得
                
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            app.processEvents() #ループ中に他のイベントを実行
        else:
            break

#####Pysideのウィンドウ処理クラス########################################
class MainWindow1(QtWidgets.QMainWindow, Ui_MainWindow): #QtWidgets.QMainWindowを継承
#=====GUI用クラス継承の定型文========================================
    def __init__(self, parent = None): 
        
        
        
        #クラス初期化時にのみ実行される関数（コンストラクタと呼ばれる）
        super(MainWindow1, self).__init__(parent) #親クラスのコンストラクタを呼び出す（親クラスのコンストラクタを再利用したい場合）　指定する引数は、親クラスのコンストラクタの引数からselfを除いた引数
        self.ui = Ui_MainWindow() #uiクラスの作成。Ui_MainWindowのMainWindowは、QT DesignerのobjectNameで設定した名前
        self.ui.setupUi(self) #uiクラスの設定
        self.ui.comboBox1.addItems(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]) #コンボボックスにアイテムを追加
        self.ui.comboBox1.setCurrentIndex(4) #コンボボックスのアイテムを選択
        self.ui.comboBox2.addItems(["320x240", "640x480", "800x600", "1024x768", "1280x960", "1400x1050", "2448x2048", "2592x1944", "320x180", "640x360", "1280x720", "1600x900", "1920x1080"]) #コンボボックスにアイテムを追加
        self.ui.comboBox2.setCurrentIndex(0) #コンボボックスのアイテムを選択
        self.ui.comboBox3.addItems(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]) #コンボボックスにアイテムを追加
        self.ui.comboBox3.setCurrentIndex(0) #コンボボックスのアイテムを選択
        self.ui.comboBox4.addItems(["320x240", "640x480", "800x600", "1024x768", "1280x960", "1400x1050", "2448x2048", "2592x1944", "320x180", "640x360", "1280x720", "1600x900", "1920x1080"]) #コンボボックスにアイテムを追加
        self.ui.comboBox4.setCurrentIndex(0) #コンボボックスのアイテムを選択
        '''
        CamNum = int(self.ui.comboBox3.currentText()) #カメラを選択
        for i in range(1,2600):
            cap=cv2.VideoCapture(CamNum,i)
            ret, _=cap.read()
            if ret:
                self.ui.comboBox6.addItem(str(i))
            cap.release()
        if self.ui.comboBox6.count() > 0:
            self.ui.comboBox6.setCurrentIndex(0) #コンボボックスのアイテムを選択
        '''
        self.ui.comboBox6.addItems(["700", "1400"]) #####コンボボックスにアイテムを追加
        self.ui.comboBox6.setCurrentIndex(0) #コンボボックスのアイテムを選択
        self.ui.comboBox7.addItems(["1", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]) #コンボボックスにアイテムを追加
        self.ui.comboBox7.setCurrentIndex(12) #コンボボックスのアイテムを選択
        self.ui.comboBox8.addItems(["640"]) #####コンボボックスにアイテムを追加
        self.ui.comboBox8.setCurrentIndex(0) #コンボボックスのアイテムを選択
        #-----シグナルにメッソドを関連付け----------------------------------------
        self.ui.checkBox1.clicked.connect(self.checkBox1_clicked) #checkBox1_clickedは任意
        self.ui.checkBox2.clicked.connect(self.checkBox2_clicked) #checkBox2_clickedは任意
        self.ui.checkBox3.clicked.connect(self.checkBox3_clicked) #checkBox2_clickedは任意
        self.ui.comboBox2.currentIndexChanged.connect(self.comboBox2_changed) #comboBox2_changedは任意
        #self.ui.comboBox3.currentIndexChanged.connect(self.comboBox3_changed) #comboBox3_changedは任意
        self.ui.comboBox4.currentIndexChanged.connect(self.comboBox4_changed) #comboBox4_changedは任意
        self.ui.pushButton1.clicked.connect(self.pushButton1_clicked) #pushButton1_clickedは任意
        self.ui.pushButton2.clicked.connect(self.pushButton2_clicked) #pushButton2_clickedは任意
        self.ui.pushButton3.clicked.connect(self.pushButton3_clicked) #pushButton3_clickedは任意
        self.ui.pushButton4.clicked.connect(self.pushButton4_clicked) #pushButton4_clickedは任意
        self.ui.pushButton5.clicked.connect(self.pushButton5_clicked) #pushButton5_clickedは任意
        self.ui.pushButton6.clicked.connect(self.pushButton6_clicked) #pushButton6_clickedは任意
        self.ui.pushButton7.clicked.connect(self.pushButton7_clicked) #pushButton7_clickedは任意
        self.ui.pushButton8.clicked.connect(self.pushButton8_clicked) #pushButton8_clickedは任意
        self.ui.pushButton9.clicked.connect(self.pushButton9_clicked) #pushButton7_clickedは任意
        self.ui.pushButton10.clicked.connect(self.pushButton10_clicked) #pushButton8_clickedは任意

#=====ウィジットのシグナル処理用メッソド========================================
    #-----checkBox1用イベント処理----------------------------------------
    ##########
    #トリムモードが変更された際の処理
    ##########
    def checkBox1_clicked(self):
        global trimMode
        global sensor_x
        global sensor_y
        if self.ui.checkBox1.isChecked() == True: #トリムモードの場合
            self.ui.pushButton8.setEnabled(False)
            self.ui.comboBox2.setEnabled(False)
            self.ui.comboBox4.setEnabled(False)
            self.ui.checkBox3.setEnabled(False)
            trimMode = 1
        else: #トリムモードでない場合
            self.ui.lineEdit1.setText('')
            self.ui.lineEdit2.setText('')
            self.ui.lineEdit6.setText("")
            self.ui.lineEdit7.setText("")
            self.ui.pushButton8.setEnabled(True)
            self.ui.comboBox2.setEnabled(True)
            self.ui.comboBox4.setEnabled(True)
            self.ui.checkBox3.setEnabled(True)
            trimMode = 0
        self.ui.checkBox2.setChecked(False)
        sensor_x = -1
        sensor_y = -1

    #-----checkBox2用イベント処理----------------------------------------
    ##########
    #センサーモードが変更された場合の処理
    ##########
    def checkBox2_clicked(self):
        global sensor_x
        global sensor_y
        if self.ui.checkBox2.isChecked() == False:
            sensor_x = -1
            sensor_y = -1

    #-----checkBox3用イベント処理----------------------------------------
    ##########
    #画像サイズ変更モードが変更された際の処理
    ##########
    def checkBox3_clicked(self):
        global sensor_x
        global sensor_y
        self.ui.lineEdit1.setText('')
        self.ui.lineEdit2.setText('')
        self.ui.lineEdit6.setText("")
        self.ui.lineEdit7.setText("")
        self.ui.checkBox2.setChecked(False)
        sensor_x = -1
        sensor_y = -1

    #-----comboBox2用イベント処理----------------------------------------
    ##########
    #画像サイズが変更された際の処理
    ##########
    def comboBox2_changed(self):
        global CapWidth
        global CapHeight
        global trimMode
        global sensor_x
        global sensor_y
        res = self.ui.comboBox2.currentText() #キャプチャーサイズを取得
        rx, ry = res.split('x') #キャプチャーサイズを代入
        CapWidth = int(rx) #キャプチャー幅を記憶
        CapHeight = int(ry) #キャプチャー高さを記憶
        self.ui.checkBox1.setChecked(False)
        self.ui.lineEdit1.setText('')
        self.ui.lineEdit2.setText('')
        self.ui.lineEdit6.setText("")
        self.ui.lineEdit7.setText("")
        self.ui.checkBox2.setChecked(False)
        sensor_x = -1
        sensor_y = -1

    '''
    #-----comboBox3用イベント処理----------------------------------------
    ##########
    #カメラ番号が変更された際の処理
    ##########
    def comboBox3_changed(self):
        self.ui.comboBox6.clear()
        CamNum = int(self.ui.comboBox3.currentText()) #カメラを選択
        for i in range(1,2600):
            cap=cv2.VideoCapture(CamNum,i)
            ret, _=cap.read()
            if ret:
                self.ui.comboBox6.addItem(str(i))
            cap.release()
        if self.ui.comboBox6.count() > 0:
            self.ui.comboBox6.setCurrentIndex(0) #コンボボックスのアイテムを選択
    '''

    #-----comboBox4用イベント処理----------------------------------------
    ##########
    #画像サイズ変更モードが変更された際の処理
    ##########
    def comboBox4_changed(self):
        global resizeWidth
        global resizeHeight
        global trimMode
        global sensor_x
        global sensor_y
        res = self.ui.comboBox4.currentText() #キャプチャーサイズを取得
        rx, ry = res.split('x') #キャプチャーサイズを代入
        resizeWidth = int(rx) #キャプチャー幅を記憶
        resizeHeight = int(ry) #キャプチャー高さを記憶
        self.ui.checkBox1.setChecked(False)
        self.ui.lineEdit1.setText('')
        self.ui.lineEdit2.setText('')
        self.ui.lineEdit6.setText("")
        self.ui.lineEdit7.setText("")
        self.ui.checkBox2.setChecked(False)
        sensor_x = -1
        sensor_y = -1

    #-----pushButton1用イベント処理----------------------------------------
    ##########
    #検出を開始
    ##########
    def pushButton1_clicked(self):
        global capLoop
        global cap
        global CapWidth
        global CapHeight
        global resizeWidth
        global resizeHeight
        global DirPath
        global DirPath
        global FileNum
        global model
        global det_target

        if self.ui.lineEdit3.text() == '' and self.ui.checkBox4.isChecked() == False:
            msgbox = QtWidgets.QMessageBox(self)
            msgbox.setWindowTitle("MDC")
            msgbox.setText("PLEASE SET PARAMETER FILE.")
            ret = msgbox.exec()
        else:
            if self.ui.checkBox4.isChecked() == True:
                DirPath = ""
                DirPath = QtWidgets.QFileDialog.getExistingDirectory(self) #写真が保存してあるフォルダを選択
                if DirPath != "":
                    FileNum = 0
                    res = self.ui.comboBox2.currentText() #キャプチャーサイズを決定
                    rx, ry = res.split('x') #キャプチャーサイズを代入
                    CamNum = int(self.ui.comboBox3.currentText()) #カメラを選択
                    cap = cv2.VideoCapture(CamNum) #キャプチャーオブジェクトを作成
                    cap.set(3, int(rx)) #キャプチャー幅を決定
                    cap.set(4, int(ry)) #キャプチャー高さを決定
                    cap.set(5, 10)
                    CapWidth = int(rx) #キャプチャー幅を記憶
                    CapHeight = int(ry) #キャプチャー高さを記憶
                    res = self.ui.comboBox4.currentText()
                    cx, cy = res.split('x')
                    resizeWidth = int(cx)
                    resizeHeight = int(cy)
                    self.ui.pushButton1.setEnabled(False)
                    self.ui.pushButton2.setEnabled(True)
                    self.ui.pushButton3.setEnabled(False)
                    self.ui.pushButton4.setEnabled(False)
                    self.ui.pushButton5.setEnabled(False)
                    self.ui.pushButton6.setEnabled(False)
                    self.ui.pushButton7.setEnabled(False)
                    self.ui.pushButton8.setEnabled(False)
                    self.ui.pushButton9.setEnabled(False)
                    self.ui.pushButton10.setEnabled(False)
                    self.ui.comboBox1.setEnabled(False)
                    self.ui.comboBox2.setEnabled(False)
                    self.ui.comboBox3.setEnabled(False)
                    self.ui.comboBox4.setEnabled(False)
                    self.ui.comboBox5.setEnabled(False)
                    self.ui.checkBox1.setEnabled(False)
                    self.ui.checkBox2.setEnabled(False)
                    self.ui.checkBox3.setEnabled(False)
                    self.ui.checkBox4.setEnabled(False)
                    self.ui.lineEdit8.setEnabled(False)
                    if capLoop == 0:
                        capLoop = 1 #ループ中とする
                        mainLoop() #メインループ用関数を実行
            else:
                '''
                currentListIndex = self.ui.comboBox6.currentIndex()
                if currentListIndex == -1:
                    msgbox = QtWidgets.QMessageBox(self)
                    msgbox.setWindowTitle("MDC")
                    msgbox.setText("NO CAMERA FOUND.")
                    ret = msgbox.exec()
                else:
                '''
                res = self.ui.comboBox2.currentText() #キャプチャーサイズを決定
                rx, ry = res.split('x') #キャプチャーサイズを代入
                CamNum = int(self.ui.comboBox3.currentText()) #カメラを選択
                ##########
                CAM_PIPE = self.ui.comboBox6.currentText()
                if CAM_PIPE == "700":
                    CAM_PIPE = cv2.CAP_DSHOW
                elif CAM_PIPE == "1400":
                    CAM_PIPE = "cv2.CAP_MSMF"
                ##########
                cap = cv2.VideoCapture(CamNum, int(CAM_PIPE))
                cap.set(3, int(rx)) #キャプチャー幅を決定
                cap.set(4, int(ry)) #キャプチャー高さを決定
                cap.set(5, 10)
                CapWidth = int(rx) #キャプチャー幅を記憶
                CapHeight = int(ry) #キャプチャー高さを記憶
                res = self.ui.comboBox4.currentText()
                cx, cy = res.split('x')
                resizeWidth = int(cx)
                resizeHeight = int(cy)
                FPS = self.ui.comboBox7.currentText()
                cap.set(cv2.CAP_PROP_FPS, int(FPS))
                w_dir = self.ui.lineEdit3.text()
                model = YOLO(w_dir, task="detect")
                with open(w_dir + "/metadata.yaml", 'r') as file:
                    det_names = yaml.safe_load(file)
                det_target =[]
                for x in det_names["names"]:
                    if det_names["names"][x] in outName:
                        det_target.append(x)
                count = self.ui.comboBox5.count()
                if len(det_target) == 0 and count > 1:
                    msgbox = QtWidgets.QMessageBox(self)
                    msgbox.setWindowTitle("MDC")
                    msgbox.setText("[OUTPUT NAME] was NOT FOUND IN [DETECTION NAMES].")
                    ret = msgbox.exec()
                self.ui.pushButton1.setEnabled(False)
                self.ui.pushButton2.setEnabled(True)
                self.ui.pushButton3.setEnabled(False)
                self.ui.pushButton4.setEnabled(False)
                self.ui.pushButton5.setEnabled(False)
                self.ui.pushButton6.setEnabled(False)
                self.ui.pushButton7.setEnabled(False)
                self.ui.pushButton8.setEnabled(False)
                self.ui.pushButton9.setEnabled(False)
                self.ui.pushButton10.setEnabled(False)
                self.ui.comboBox1.setEnabled(False)
                self.ui.comboBox2.setEnabled(False)
                self.ui.comboBox3.setEnabled(False)
                self.ui.comboBox4.setEnabled(False)
                self.ui.comboBox5.setEnabled(False)
                self.ui.comboBox6.setEnabled(False)
                self.ui.comboBox7.setEnabled(False)
                self.ui.comboBox8.setEnabled(False)
                self.ui.checkBox1.setEnabled(False)
                self.ui.checkBox2.setEnabled(False)
                self.ui.checkBox3.setEnabled(False)
                self.ui.checkBox4.setEnabled(False)
                self.ui.lineEdit8.setEnabled(False)
                if capLoop == 0:
                    capLoop = 1 #ループ中とする
                    mainLoop() #メインループ用関数を実行
    #-----pushButton2用イベント処理----------------------------------------
    ##########
    #検出を終了
    ##########
    def pushButton2_clicked(self):
        global cap
        global capLoop
        global metaMain
        global netMain
        global altNames
        if capLoop == 1:
            capLoop = 0 #ループ中ではないとする
            time.sleep(0.2)
        self.ui.pushButton1.setEnabled(True)
        self.ui.pushButton2.setEnabled(False)
        self.ui.pushButton3.setEnabled(True)
        self.ui.pushButton4.setEnabled(True)
        self.ui.pushButton5.setEnabled(True)
        self.ui.pushButton6.setEnabled(True)
        self.ui.pushButton7.setEnabled(True)
        self.ui.pushButton9.setEnabled(True)
        self.ui.pushButton10.setEnabled(True)
        if trimMode == 0:
            self.ui.comboBox2.setEnabled(True)
            self.ui.comboBox4.setEnabled(True)
            self.ui.pushButton8.setEnabled(True)
        self.ui.comboBox1.setEnabled(True)
        self.ui.comboBox3.setEnabled(True)
        self.ui.comboBox4.setEnabled(True)
        self.ui.comboBox5.setEnabled(True)
        self.ui.comboBox6.setEnabled(True)
        self.ui.comboBox7.setEnabled(True)
        self.ui.comboBox8.setEnabled(True)
        self.ui.checkBox1.setEnabled(True)
        self.ui.checkBox2.setEnabled(True)
        self.ui.checkBox3.setEnabled(True)
        self.ui.checkBox4.setEnabled(True)
        self.ui.lineEdit8.setEnabled(True)
        cap.release()
        netMain = None
        metaMain = None
        altNames = None
        cap.release() #キャプチャー用オブジェクトを廃棄
        cv2.destroyAllWindows() #画像表示用のウィンドウを全て閉じる

    #-----pushButton3用イベント処理----------------------------------------
    ##########
    #設定読込み処理
    ##########
    def pushButton3_clicked(self):
    #####ファイル読込
        global cap
        global CapWidth
        global CapHeight
        global trimMode
        global tw
        global th
        global resizeWidth
        global resizeHeight
        global sensor_x
        global sensor_y
        global outName
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "",'mdy File (*.mdy)') #設定ファイルを選択
        if filepath: #ファイルが存在するか確認
            #####ファイル名のみの取得
            filename1 = filepath.rsplit(".", 1) #ファイルパスの文字列右側から指定文字列で分割
            filename2 = filename1[0].rsplit("/", 1) #ファイルパスの文字列右側から指定文字列で分割
            f = open(filepath, "r") #ファイルの読み込み開始
            text = f.readlines() #テキストを一行ずつ配列として読込む（行の終わりの改行コードも含めて読込む）
            f.close() #ファイルの読み込み終了
            self.ui.comboBox1.setCurrentIndex(int(text[0].replace("\n", ""))) #改行コードを削除してデータを読込む
            self.ui.comboBox2.setCurrentIndex(int(text[1].replace("\n", ""))) #改行コードを削除してデータを読込む
            res = self.ui.comboBox2.currentText() #キャプチャーサイズを取得
            rx, ry = res.split('x') #キャプチャーサイズを代入
            CapWidth = int(rx) #キャプチャー幅を記憶
            CapHeight = int(ry) #キャプチャー高さを記憶
            self.ui.comboBox3.setCurrentIndex(int(text[2].replace("\n", ""))) #改行コードを削除してデータを読込む
            self.ui.lineEdit1.setText(text[3].replace("\n", "")) #改行コードを削除してデータを読込む
            self.ui.lineEdit2.setText(text[4].replace("\n", "")) #改行コードを削除してデータを読込む
            self.ui.lineEdit3.setText(text[5].replace("\n", "")) #改行コードを削除してデータを読込む
            self.ui.lineEdit4.setText(text[6].replace("\n", "")) #改行コードを削除してデータを読込む
            self.ui.lineEdit5.setText(text[7].replace("\n", "")) #改行コードを削除してデータを読込む
            self.ui.lineEdit6.setText(text[8].replace("\n", "")) #改行コードを削除してデータを読込む
            self.ui.lineEdit7.setText(text[9].replace("\n", "")) #改行コードを削除してデータを読込む
            cFlag = int(text[10])
            if cFlag == 1:
                self.ui.checkBox1.setChecked(True)
                self.ui.checkBox3.setEnabled(False)
                self.ui.comboBox2.setEnabled(False)
                self.ui.comboBox4.setEnabled(False)
                self.ui.pushButton8.setEnabled(False)
                trimMode = 1
            else:
                self.ui.checkBox1.setChecked(False)
                self.ui.checkBox3.setEnabled(True)
                self.ui.pushButton8.setEnabled(True)
                self.ui.comboBox2.setEnabled(True)
                self.ui.comboBox4.setEnabled(True)
                trimMode = 0
            if int(text[11]) == 0:
                self.ui.checkBox2.setChecked(False)
            else:
                self.ui.checkBox2.setChecked(True)
            self.ui.comboBox4.setCurrentIndex(int(text[12].replace("\n", ""))) #改行コードを削除してデータを読込む
            res = self.ui.comboBox4.currentText() #キャプチャーサイズを取得
            rx, ry = res.split('x') #キャプチャーサイズを代入
            resizeWidth = int(rx) #キャプチャー幅を記憶
            resizeHeight = int(ry) #キャプチャー高さを記憶
            if int(text[13]) == 0:
                self.ui.checkBox3.setChecked(False)
            else:
                self.ui.checkBox3.setChecked(True)
            sensor_x = int(text[14])
            sensor_y = int(text[15])
            textlist = text[16].replace("\n", "")
            textlist = textlist.split(" ")
            if len(textlist) > 0:
                self.ui.comboBox5.clear() #Combo Boxの内容を全て消去
                for x in textlist:
                    self.ui.comboBox5.addItem(x) #textListの各配列要素をCombo Boxに追加
            outName = ""
            lCount = self.ui.comboBox5.count()
            for CurIndex in range(lCount):
                outName = outName + self.ui.comboBox5.itemText(CurIndex) + " " #指定したインデックスのテキストを取得
            self.ui.comboBox6.setCurrentIndex(int(text[17].replace("\n", ""))) #改行コードを削除してデータを読込む
            self.ui.comboBox7.setCurrentIndex(int(text[18].replace("\n", ""))) #改行コードを削除してデータを読込む
            self.ui.comboBox8.setCurrentIndex(int(text[19].replace("\n", ""))) #改行コードを削除してデータを読込む
            #####
    #####

    #-----pushButton4用イベント処理----------------------------------------
    ##########
    #設定書込み処理
    ##########
    def pushButton4_clicked(self):
    #####ファイル書込み
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "",'mdy File (*.mdy)')
        if filepath: #ファイルが存在するか確認
            #####ファイル名のみの取得
            filename1 = filepath.rsplit(".", 1) #ファイルパスの文字列右側から指定文字列で分割
            filename2 = filename1[0].rsplit("/", 1) #ファイルパスの文字列右側から指定文字列で分割
            #os.chdir(filename2[0] + "/") #カレントディレクトリをファイルパスへ変更
            f = open(filepath, "w") #ファイルの書き込み開始
            if self.ui.checkBox1.isChecked() == True:
                cFlag = 1
            else:
                cFlag = 0
            text = str(self.ui.comboBox1.currentIndex()) + "\n" + \
            str(self.ui.comboBox2.currentIndex()) + "\n" + \
            str(self.ui.comboBox3.currentIndex()) + "\n" + \
            self.ui.lineEdit1.text() + "\n" + \
            self.ui.lineEdit2.text() + "\n" + \
            self.ui.lineEdit3.text() + "\n" + \
            self.ui.lineEdit4.text() + "\n" + \
            self.ui.lineEdit5.text() + "\n" + \
            self.ui.lineEdit6.text() + "\n" + \
            self.ui.lineEdit7.text() + "\n" + \
            str(cFlag) + "\n"
            if self.ui.checkBox2.isChecked() == True:
                text = text + "1\n"
            else:
                text = text + "0\n"
            text = text + str(self.ui.comboBox4.currentIndex()) + "\n"
            if self.ui.checkBox3.isChecked() == True:
                text = text + "1\n"
            else:
                text = text + "0\n"
            text = text + str(sensor_x) + "\n"
            text = text + str(sensor_y) + "\n"
            lCount = self.ui.comboBox5.count()
            if lCount > 0:
                for CurIndex in range(lCount):
                    text = text + self.ui.comboBox5.itemText(CurIndex) + " " #指定したインデックスのテキストを取得
                text = text[:-1] + "\n"
            else:
                text = text + "\n"
            text = text + str(self.ui.comboBox6.currentIndex()) + "\n"
            text = text + str(self.ui.comboBox7.currentIndex()) + "\n"
            text = text + str(self.ui.comboBox8.currentIndex()) + "\n"
            f.writelines(text) #空ファイルとして書込み
            f.close() #ファイルの書き込み終了
            msgbox = QtWidgets.QMessageBox(self) #####メッセージボックスを準備
            msgbox.setWindowTitle("MDC")
            msgbox.setText("FILE : Saved.") #####メッセージボックスのテキストを設定
            ret = msgbox.exec() #####メッセージボックスを表示
            #####
    #####

    #-----pushButton5用イベント処理----------------------------------------
    def pushButton5_clicked(self):
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(self)
        if dirpath: #ファイルが存在するか確認
            self.ui.lineEdit3.setText(dirpath)

    #-----pushButton6用イベント処理----------------------------------------
    def pushButton6_clicked(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "",'cfg File (*.cfg)')
        if filepath: #ファイルが存在するか確認
            self.ui.lineEdit4.setText(filepath)

    #-----pushButton7用イベント処理----------------------------------------
    def pushButton7_clicked(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "",'weights File (*.weights)')
        if filepath: #ファイルが存在するか確認
            self.ui.lineEdit5.setText(filepath)

    #-----pushButton8用イベント処理----------------------------------------
    def pushButton8_clicked(self):
        win.ui.lineEdit1.setText('')
        win.ui.lineEdit2.setText('')
        win.ui.lineEdit6.setText('')
        win.ui.lineEdit7.setText('')

    #-----pushButton9用イベント処理----------------------------------------
    def pushButton9_clicked(self):
        global outName
        if self.ui.lineEdit8.text() == "":
            msgbox = QtWidgets.QMessageBox(self) #####メッセージボックスを準備
            msgbox.setWindowTitle("MDC")
            msgbox.setText("NO TEXT.") #####メッセージボックスのテキストを設定
            ret = msgbox.exec() #####メッセージボックスを表示
        else:
            text = ""
            lCount = self.ui.comboBox5.count()
            for CurIndex in range(lCount):
                text = text + self.ui.comboBox5.itemText(CurIndex) + " " #指定したインデックスのテキストを取得
            if self.ui.lineEdit8.text() in text:
                msgbox = QtWidgets.QMessageBox(self) #####メッセージボックスを準備
                msgbox.setWindowTitle("MDC")
                msgbox.setText("NAME IS ALREADY ADDED.") #####メッセージボックスのテキストを設定
                ret = msgbox.exec() #####メッセージボックスを表示
            else:
                self.ui.comboBox5.addItem(self.ui.lineEdit8.text())
                self.ui.comboBox5.setCurrentIndex(self.ui.comboBox5.count() - 1)
                self.ui.comboBox5.setFocus()
                win.ui.lineEdit8.setText("")
                outName = ""
                lCount = self.ui.comboBox5.count()
                for CurIndex in range(lCount):
                    outName = outName + self.ui.comboBox5.itemText(CurIndex) + " " #指定したインデックスのテキストを取得

    #-----pushButton10用イベント処理----------------------------------------
    def pushButton10_clicked(self):
    #####現在選択されているアイテムを削除
        global outName
        currentListIndex = self.ui.comboBox5.currentIndex()
        if currentListIndex == -1:
            msgbox = QtWidgets.QMessageBox(self)
            msgbox.setWindowTitle("MDC")
            msgbox.setText("NO ITEM.")
            ret = msgbox.exec()
            self.ui.comboBox5.setFocus()
        else:
            self.ui.comboBox5.removeItem(currentListIndex)
            self.ui.comboBox5.setFocus()
            outName = ""
            lCount = self.ui.comboBox5.count()
            for CurIndex in range(lCount):
                outName = outName + self.ui.comboBox5.itemText(CurIndex) + " " #指定したインデックスのテキストを取得

#=====メインウィンドウのイベント処理========================================
    #-----ウィンドウ終了イベントのフック----------------------------------------
    def closeEvent(self, event): #event.accept() event.ignore()で処理を選択可能
        global capLoop
        if capLoop == 1: #ループ実行中の場合
            event.ignore() #メインウィンドウの終了イベントをキャンセル
        else: #ループが実行中でない場合
            event.accept() #メインウィンドウの終了イベントを実行

#=====メインウィンドウで取得したウィジットのイベント処理========================================
def onMouse(event, x, y, flags, param):  
        global capLoop
        global sStartFlag
        global mX1
        global mY1
        global mX2
        global mY2
        global ssX
        global ssY
        global sXL
        global sYL
        global sensor_x
        global sensor_y
        #マウスが移動た時の処理
        #マウスボタンがクリックされた時の処理
        if event == cv2.EVENT_LBUTTONDOWN and win.ui.lineEdit6.text() == '' and trimMode == 1:
            if sStartFlag == 0 and capLoop == 1:
                sStartFlag = 1
                #マウス位置の取得
                mX1 = x
                mY1 = y
                mX2 = x
                mY2 = y
                ret, ssX, ssY, sXL, sYL, _ , _ = getRectanglePos(mX1, mY1, mX2, mY2, CapWidth, CapHeight)
        #マウスボタンがリリースされた時の処理
        elif event == cv2.EVENT_LBUTTONUP and win.ui.lineEdit6.text() == '' and trimMode == 1:
            if sStartFlag == 1:
                sStartFlag = 0
                #マウス位置の取得
                mX2 = x
                mY2 = y
                ret, ssX, ssY, sXL, sYL, W , H = getRectanglePos(mX1, mY1, mX2, mY2, CapWidth, CapHeight)
                if W > 100 and H > 100 and ret == 1: #選択領域が小さければエラーを表示
                    #選択領域をオリジナルの座標に変換して記憶
                    win.ui.lineEdit6.setText(str(ssX))
                    win.ui.lineEdit7.setText(str(ssY))
                    win.ui.lineEdit1.setText(str(W))
                    win.ui.lineEdit2.setText(str(H))
                    win.ui.checkBox2.setChecked(False)
                    sensor_x = -1
                    sensor_y = -1
                else:
                    msgbox = QtWidgets.QMessageBox() #####メッセージボックスを準備
                    msgbox.setWindowTitle("MDC")
                    msgbox.setText("The region width and height must be greater than 100 pixel.") #####メッセージボックスのテキストを設定
                    ret = msgbox.exec() #####メッセージボックスを表示
        #マウスボタンが移動した時の処理
        elif event == cv2.EVENT_MOUSEMOVE and win.ui.lineEdit6.text() == '' and trimMode == 1:
            if sStartFlag == 1:
                #マウス位置の取得
                mX2 = x
                mY2 = y
                ret, ssX, ssY, sXL, sYL, _ , _ = getRectanglePos(mX1, mY1, mX2, mY2, CapWidth, CapHeight)
            else:
                mX1 = x
                mY1 = y
        elif event == cv2.EVENT_RBUTTONDOWN and win.ui.checkBox2.isChecked() == True and sensor_x == -1:
            sensor_x = x
            sensor_y = y

#=====点が四角の範囲内か調べる関数========================================
def is_point_in_rectangle(point_x, point_y, rect_x1, rect_y1, rect_x2, rect_y2):
    # 四角形のx座標とy座標の範囲を正規化（大小関係が逆転しても対応できるようにする）
    min_x = min(rect_x1, rect_x2)
    max_x = max(rect_x1, rect_x2)
    min_y = min(rect_y1, rect_y2)
    max_y = max(rect_y1, rect_y2)
    return min_x <= point_x <= max_x and min_y <= point_y <= max_y

#####メイン処理（グローバル）########################################
#=====メイン処理定型文========================================
if __name__ == '__main__': #C言語のmain()に相当。このファイルが実行された場合、以下の行が実行される（モジュールとして読込まれた場合は、実行されない）
    app = QtWidgets.QApplication(sys.argv) #アプリケーションオブジェクト作成（sys.argvはコマンドライン引数のリスト）
    win = MainWindow1() #MainWindow1クラスのインスタンスを作成
    win.show() #ウィンドウを表示　win.showFullScreen()やwin.showEvent()を指定する事でウィンドウの状態を変える事が出来る
    sys.exit(app.exec()) #引数が関数の場合は、関数が終了するまで待ち、その関数の返値を上位プロセスに返す
