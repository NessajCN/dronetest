import robomaster
import cv2
from robomaster import robot


if __name__ == '__main__':

    # fill in your lan address:
    robomaster.config.LOCAL_IP_STR = "192.168.3.50"
    tl_drone = robot.Drone()
    tl_drone.initialize(conn_type="sta")

    # 获取飞机电池电量信息
    tl_battery = tl_drone.battery
    battery_info = tl_battery.get_battery()
    print("Drone battery soc: {0}".format(battery_info))

    # stop motor spinning
    tl_flight = tl_drone.flight
    tl_flight.motor_off()

    # initialize the camera
    tl_camera = tl_drone.camera

    # 显示302帧图传
    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("high")
    tl_camera.set_resolution("high")
    tl_camera.set_bitrate(6)
    for i in range(0, 302):
        img = tl_camera.read_cv2_image()
        cv2.imshow("Drone", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    tl_camera.stop_video_stream()

    tl_drone.close()
