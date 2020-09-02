import cv2
import os
from Face_Train import Model
import tensorflow
import dlib
import cv2
import sys
import keras

ageModel = 'models//age_net.caffemodel'
ageProto = 'models//age_deploy.prototxt'

genderModel = 'models//gender_net.caffemodel'
genderProto = 'models//gender_deploy.prototxt'

# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
def CatchPICFromVideo(window_name,camera_idx,catch_pic_num,path_name):
    model = Model()
    model.load_model(file_path='C:\\Users\\57261\\Desktop\\recognition\\aggregate.face.model.h5')
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    genderList = ['Male', 'Female']

    detector = dlib.get_frontal_face_detector()

    num = 0
    while cap.isOpened():
        print('open')
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            print("action")
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
        faces = detector(grey, 1)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        for face in faces:
            #脸部坐标
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            #截取人脸
            image = frame[top: bottom, left: right]

            label = model.face_predict(image)

            print(label)

            # 从label对应出预测名字结果
            for dir in range(len(os.listdir('trainset'))):
                if label == dir:
                    cv2.putText(frame, os.listdir('trainset')[dir],
                                (left + 30, top + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1, (255, 0, 255), 2)# colour

            #性别识别
            blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            cv2.putText(frame, gender,
                        (left + 30, top - 10),  # 坐标
                        cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                        1, (255, 0, 255), 2)  # colour
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
            img_name = '%s\%d.jpg' % (path_name, num)
            cv2.imwrite(img_name, image)
            num += 1
            if num > (catch_pic_num):  # 如果超过指定最大保存数量退出循环
                break
            # font = cv2.FONT_HERSHEY_SIMPLEXre

        # 显示图像
        cv2.imshow(window_name, frame)
        #按键盘‘Q’中断采集
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

#判断本程序是独立运行还是被调用
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("截取人脸", 0, 200, 'dataset')