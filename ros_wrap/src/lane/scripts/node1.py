#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def talker():
    pub = rospy.Publisher('videotopic',Image,queue_size=50)
    rospy.init_node('node1',anonymous=True)
    rate = rospy.Rate(10)
    cap = cv2.VideoCapture('/home/sb/manas_tp/opencv/lane detection/lane.mp4')
    bridge = CvBridge()
    ax = np.array([5])
    while not rospy.is_shutdown():
        ret,frame = cap.read()
        if not ret:
            print("ret false")
            #print(ax)
            break
        try:
            image = bridge.cv2_to_imgmsg(frame,"bgr8")
        except CvBridgeError as err:
            print(err)
        pub.publish(image)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
