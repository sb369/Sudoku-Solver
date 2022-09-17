#!/usr/bin/env python
import rospy
from lane.msg import Coeff

def callback(data):
    pass
    
def listener():
    rospy.init_node('node3', anonymous=True)

    rospy.Subscriber("coeffs", Coeff, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
