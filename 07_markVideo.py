import time
import cv2
import robomaster
from robomaster import robot, flight
from numpy import *


# YOLO Begin

#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
# import time

# import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

# YOLO END

if __name__ == '__main__':

    # fill in your lan address:
    # robomaster.config.LOCAL_IP_STR = "192.168.0.22"
    # fill in robot lan address:
    # robomaster.config.ROBOT_IP_STR = "192.168.0.24"

    # TT无人机默认 udp 模式，不设置也不影响
    # DEFAULT_PROTO_TYPE = "udp"

    # yolo = YOLO()  # For YOLO

    robomaster.config.LOCAL_IP_STR = "192.168.10.2"

    tl_drone = robot.Drone()
    tl_drone.initialize()
    # tl_drone.initialize(conn_type="sta")

    # 获取飞机电池电量信息
    tl_battery = tl_drone.battery
    battery_info = tl_battery.get_battery()
    print("Drone battery soc: {0}".format(battery_info))

    # stop motor spinning
    # tl_flight = tl_drone.flight
    # tl_flight.motor_off()

    # initialize the camera
    tl_camera = tl_drone.camera
    t = time.time()
    # t_list = [0]*20

    # 显示302帧图传
    # tl_camera.start_video_stream(display=False)
    # tl_camera.set_fps("high")
    # tl_camera.set_resolution("high")
    # tl_camera.set_bitrate(6)
    # for i in range(0, 302):
    #     img = tl_camera.read_cv2_image()
    #
    #     cv2.imshow("Drone", img)
    #     cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # tl_camera.stop_video_stream()


    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("low")  # 默认high
    tl_camera.set_resolution("low")  # 默认high
    tl_camera.set_bitrate(6)  # 默认6


    yolo = YOLO()
    # -------------------------------------#
    #   调用摄像头
    #   capture=cv2.VideoCapture("1.mp4")
    # -------------------------------------#
    fps = 0.0
    while (True):
        # 读取某一帧
        img = tl_camera.read_cv2_image(strategy="newest")

        t1 = time.time()

        # 格式转变，BGRtoRGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 转变成Image
        img = Image.fromarray(np.uint8(img))
        # 进行检测
        img = np.array(yolo.detect_image(img))
        # RGBtoBGR满足opencv显示格式
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 缩小原图
        # height, width = img.shape[:2]
        # size = (int(width*0.3), int(height*0.5))
        # img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # 放大原图，输入尺寸格式为（宽，高）
        fx = 3
        fy = 2.1
        img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        img = cv2.putText(img, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", img)

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break

    cv2.destroyAllWindows()
    tl_camera.stop_video_stream()

    tl_drone.close()
