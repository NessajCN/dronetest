import robomaster
import cv2
from robomaster import robot
import time


if __name__ == '__main__':

    # fill in your lan address:
    robomaster.config.LOCAL_IP_STR = "192.168.10.2"
    # robomaster.config.ROBOT_IP_STR = "192.168.31.143"
    # robomaster.config.DEFAULT_CONN_TYPE = "sta"
    tl_drone = robot.Drone()
    tl_drone.initialize()

    # 获取飞机电池电量信息
    tl_battery = tl_drone.battery
    battery_info = tl_battery.get_battery()
    print("Drone battery soc: {0}".format(battery_info))

    # start motor spinning
    tl_flight = tl_drone.flight
    tl_flight.motor_on()

    # initialize the camera
    tl_camera = tl_drone.camera

    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("high")
    tl_camera.set_resolution("low")
    tl_camera.set_bitrate(6)
    for i in range(0, 302):
        img = tl_camera.read_cv2_image()
        cv2.imshow("Drone", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    tl_camera.stop_video_stream()

    #stop motor spinning
    tl_flight.motor_off()

    print("test successfully!")

    tl_drone.close()
