# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import robomaster
from robomaster import robot
import time


if __name__ == '__main__':
    robomaster.config.LOCAL_IP_STR = "192.168.31.44"
    tl_drone = robot.Drone()
    tl_drone.initialize(conn_type="sta")

    tl_flight = tl_drone.flight

    # 起飞
    tl_flight.takeoff().wait_for_completed()

    # patrol
    for i in range(0,1):
        tl_flight.go(x=150, y=0, z=0, speed=50).wait_for_completed()
        tl_flight.go(x=0, y=40, z=0, speed=50).wait_for_completed()
        tl_flight.go(x=-150, y=0, z=0, speed=50).wait_for_completed()
        tl_flight.go(x=0, y=40, z=0, speed=50).wait_for_completed()
    
    # back to origin
    tl_flight.go(x=0, y=-200, z=0, speed=50).wait_for_completed()

    # look for nearest mid card 
    tl_flight.go(x=0, y=0, z=100, speed=30, mid1="m-2").wait_for_completed()
    
    # pre-landing
    tl_flight.go(x=0, y=0, z=70, speed=20, mid1="m-2").wait_for_completed()
    time.sleep(2)
    tl_flight.go(x=0, y=0, z=40, speed=20, mid1="m-2").wait_for_completed()
    time.sleep(2)

    # 降落
    tl_flight.land().wait_for_completed()

    tl_drone.close()

