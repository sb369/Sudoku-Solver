#!/usr/bin/env python
from __future__ import print_function
import cv2 as cv
import rospy
import numpy as np
from sensor_msgs.msg import Image
from lane.msg import Coeff
from cv_bridge import CvBridge, CvBridgeError

#### LANE DETECTION FUNCTIONS #####
def process2(frame):
    l1 = np.array([100,100,200],dtype=np.uint8)
    u1 = np.array([255,255,255],dtype=np.uint8)
    l2 = np.array([0,180,255],dtype=np.uint8)
    u2 = np.array([170,255,255],dtype=np.uint8)
    l3 = np.array([20,120,80],dtype=np.uint8)
    u3 = np.array([45,200,255],dtype=np.uint8)
    
    mask1 = cv.inRange(frame,l1,u1)
    mask2 = cv.inRange(frame,l2,u2)
    mask3 = cv.inRange(cv.cvtColor(frame,cv.COLOR_BGR2HLS),l3,u3)
    mask = cv.bitwise_or(mask1,mask2)
    mask = cv.bitwise_or(mask,mask3)
    
    blur = cv.GaussianBlur(mask,(5,5),0)
    canny = cv.Canny(blur,50,100)
    i = canny
    return mask

def threshold_rel(img,l,h):
    vmin = np.min(img)
    vmax = np.max(img)
    lo = vmin + (vmax-vmin)*l
    hi = vmin + (vmax-vmin)*h
    return np.uint8((img>=lo)&(img<=hi))*255

def threshold_abs(img,l,h):
    return np.uint8((img>=l)&(img<=h))*255

def process3(frame):
    hls = cv.cvtColor(frame,cv.COLOR_BGR2HLS)
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]
    v = hsv[:,:,2]
    
    right_lane = threshold_rel(l, 0.8, 1.0)
    right_lane[:,:750] = 0
    
    left_lane = threshold_abs(h, 20, 30)
    left_lane &= threshold_rel(v, 0.7, 1.0)
    left_lane[:,550:] = 0
    
    mask = left_lane | right_lane
    
    
    blur = cv.GaussianBlur(mask,(5,5),0)
    canny = cv.Canny(blur,50,100)
    i = canny
    return mask

def transform_backward(img,pts=None):
    x_size = img.shape[1]
    y_size = img.shape[0]
    if pts is None:
        mid_x = x_size//2
        top_y = 2*y_size//3
        top_margin = 93
        bottom_margin = 450
        pts = ([(mid_x-top_margin,top_y),(mid_x+top_margin,top_y),(mid_x-bottom_margin,y_size),(mid_x+bottom_margin,y_size)],
               [(mid_x-bottom_margin,0),(mid_x+bottom_margin,0),(mid_x-bottom_margin,y_size),(mid_x+bottom_margin,y_size)])
        
    pts1 = np.float32(pts[1])
    pts2 = np.float32(pts[0])
    M = cv.getPerspectiveTransform(pts1,pts2)
    new = cv.warpPerspective(img,M,(x_size,y_size),flags=cv.INTER_LINEAR)
    
    return new

def transform_forward(img,pts=None):
    x_size = img.shape[1]
    y_size = img.shape[0]
    if pts is None:
        mid_x = x_size//2
        top_y = 2*y_size//3
        top_margin = 93
        bottom_margin = 450
        pts = ([(mid_x-top_margin,top_y),(mid_x+top_margin,top_y),(mid_x-bottom_margin,y_size),(mid_x+bottom_margin,y_size)],
               [(mid_x-bottom_margin,0),(mid_x+bottom_margin,0),(mid_x-bottom_margin,y_size),(mid_x+bottom_margin,y_size)])
        
    pts1 = np.float32(pts[0])
    pts2 = np.float32(pts[1])
    M = cv.getPerspectiveTransform(pts1,pts2)
    new = cv.warpPerspective(img,M,(x_size,y_size),flags=cv.INTER_LINEAR)
    
    return new 

def find_lane(img,nwindows=10,margin=80,minpix=None):
    if minpix is None:
        minpix = margin*2
    assert (img.ndim == 2)
    new_bin = np.zeros_like(img)
    new_bin[img>0] = 1
    
    out = np.copy(cv.cvtColor(img,cv.COLOR_GRAY2BGR))
    out[out>0] = 255
    
    histogram = np.sum(new_bin[new_bin.shape[0]//2000:,:],axis=0)
    mid = histogram.shape[0]//2
    left_x = np.argmax(histogram[:mid])
    right_x = np.argmax(histogram[mid:]) + mid
    
    #find nonzero pixel index number in image (coordinates of bright)
    win_height = new_bin.shape[0]//nwindows
    nonzero = new_bin.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])
    
    left_lane_ind = []
    right_lane_ind = []
    #find windows
    for win in range(nwindows):
        win_y_low = new_bin.shape[0] - (win+1)*win_height
        win_y_high = new_bin.shape[0] - win*win_height
        win_x_left_low = max(0,left_x-margin)
        win_x_left_high = left_x+margin
        win_x_right_low = right_x - margin
        win_x_right_high = min(new_bin.shape[1]-1,right_x+margin)
        
        #draw rectangel:
        cv.rectangle(out,(win_x_left_low,win_y_high),(win_x_left_high,win_y_low),(0,255,0),3)
        cv.rectangle(out,(win_x_right_low,win_y_high),(win_x_right_high,win_y_low),(0,255,0),3)
        
        #non-zero pixel within the windows, returned as positions(index) in nonzero_x 
        left_ind = ((nonzero_y>=win_y_low) & (nonzero_y<=win_y_high) &
                    (nonzero_x>=win_x_left_low) & (nonzero_x<=win_x_left_high)).nonzero()[0]
        right_ind = ((nonzero_y>=win_y_low) & (nonzero_y<=win_y_high) &
                     (nonzero_x>=win_x_right_low) & (nonzero_x<=win_x_right_high)).nonzero()[0]
        left_lane_ind.append(left_ind)
        right_lane_ind.append(right_ind)
        
        #re-align centre
        if len(left_ind)>= minpix:
            left_x = int(np.mean(nonzero_x[left_ind]))
        if len(right_ind)>=minpix:
            right_x = int(np.mean(nonzero_x[right_ind]))
            
    #concatenate array of index of pixels inside all boxes
    left_lane_ind = np.concatenate(left_lane_ind)
    right_lane_ind = np.concatenate(right_lane_ind)
    
    #coordinates of final "good pixels" i.e inside the boxes
    left_x = nonzero_x[left_lane_ind]
    left_y = nonzero_y[left_lane_ind]
    right_x = nonzero_x[right_lane_ind]
    right_y = nonzero_y[right_lane_ind]
    
    #color the good pixels
    out[left_y,left_x] = [255,0,0]
    out[right_y,right_x] = [0,0,255]
    
    #fit 2nd order poly
    left_fit,right_fit = np.array([]),np.array([])
    
    if (left_y.size!=0) and (left_x.size!=0):
        left_fit = np.polyfit(left_y,left_x,2)
    if (right_y.size!=0) and (right_x.size!=0):
        right_fit = np.polyfit(right_y,right_x,2)
    
    coeffs = (np.array(left_fit,np.float64),np.array(right_fit,np.float64))
    
    #get coordinates of poly
    y_val = np.linspace(0,new_bin.shape[0]-1,new_bin.shape[0])
    left_x_val,right_x_val = np.array([]),np.array([])
    if left_fit.size!=0:
        left_x_val = left_fit[0]*y_val**2 + left_fit[1]*y_val + left_fit[2]
    if right_fit.size!=0:
        right_x_val = right_fit[0]*y_val**2 + right_fit[1]*y_val + right_fit[2]
    
    #draw poly
    #left_points,right_points = np.array([]),np.array([])
    if left_x_val.size!=0:
        left_points = np.array([np.vstack([left_x_val,y_val]).T])
        pts = np.array(left_points.squeeze(),np.int32)
        pts.reshape((-1,1,2))
        cv.polylines(out,[pts],False,(254,0,0),50)
    if right_x_val.size!=0:
        right_points = np.array([np.flipud(np.vstack([right_x_val,y_val]).T)])
        pts = np.array(right_points.squeeze(),np.int32)
        pts.reshape((-1,1,2))
        cv.polylines(out,[pts],False,(254,0,0),50)
    lane = np.copy(out)
    lane[out==255] = 0
    if (left_x_val.size!=0) and (right_x_val.size!=0):
        pts = np.hstack((left_points,right_points))
        cv.fillPoly(lane,np.int_([pts]),(0,254,0))
    return out,lane,coeffs



def get_coeffs(frame):
    new = process3(transform_forward(frame))
    out,lanes,coeffs = find_lane(new)
    lanes = transform_backward(lanes)
    final = cv.addWeighted(frame,1,lanes,0.35,1)
    return final,coeffs


############# ROS NODE ##############

pub = rospy.Publisher('coeffs',Coeff,queue_size=10)

def callback(data):
    bridge = CvBridge()
    #pub = rospy.Publisher('coeffs',Coeff,queue_size=10)
    try:
        frame = bridge.imgmsg_to_cv2(data,"bgr8")
    except CvBridge as err:
        print(err)
    final,c = get_coeffs(frame)
    coeffs = Coeff()
    coeffs.header = data.header
    coeffs.left_lane = tuple(c[0])
    coeffs.right_lane = tuple(c[1])
    #print(coeffs.left_lane)
    pub.publish(coeffs)
    cv.imshow('ffffff',final)
    cv.waitKey(1)

def listener():
    rospy.init_node('node2',anonymous=True)
    rospy.Subscriber('videotopic',Image,callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
    cv.destroyAllWindows()