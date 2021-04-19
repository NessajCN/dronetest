import robomaster
from robomaster import robot


if __name__ == '__main__':

    # power on the drone
    # attach the extending module and switch to AP mode
    # connect your wifi to RMTT-D25BD0
    robomaster.config.LOCAL_IP_STR = "192.168.10.2"
    tl_drone = robot.Drone()
    tl_drone.initialize()

    # 指定路由器SSID和密码
    # Run 00_set_sta.py
    tl_drone.config_sta(ssid="robomasterTT", password="tj66157001")
    print("Switch to STA mode")
    print("Then wait until the motor start spinning")
    print("which indicates that your drone has been successfully connected to your Wifi.")
    print("Connect your wifi to the same one with the drone.")
    print("Run 01_test1.py for further testing after properly configured.")

    tl_drone.close()