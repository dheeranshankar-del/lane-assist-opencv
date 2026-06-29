import cv2
import numpy as np
import time

#converts the lane equation like slope and intercept into two endpoints
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1= image.shape[0]
    y2=int(y1*(3/5))
    x1= int((y1 - intercept)/slope)
    x2= int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

#Creates a smooth left and right lane line from multiple line segments
def average_slope_intercept (image, lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line= make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

#Converts image to grayscale and reduces noise with gaussian blur adn detects edges with Canny
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #add gaussian blur
    blur = cv2.GaussianBlur(gray, (5,5),0)
    #identify the edges
    canny=cv2.Canny(blur, 50, 150)
    return canny
    
#Draws the detected lane lines into a blank image to put on the video frame
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

#function ignores everything except where the road is likely to be
def region_of_interest(image):
    #setting points in a triangle shape of area of interest
    height = image.shape[0] #gets image height #shape[1] gets image width but you don't need it here
    polygons = np.array([[(200,height), (1100, height), (550,250)]])
    #height is basically 700
    mask = np.zeros_like(image) #makes the whole image black
    cv2.fillPoly(mask, polygons, 255) #selects the traingle area and sets it to white
    masked_image= cv2.bitwise_and(image, mask)
    return masked_image

#measures program FPS
frame_count = 0
fps_list = []

prev_time = time.time()

FRAME_DELAY_MS =0

#opens the input video file
cap = cv2.VideoCapture(r"C:\Users\dheer\Downloads\vs code\Lane Assist\test2.mp4")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
        
#Processes each video frame through the lane detection pipeline and displays the result
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    Line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, Line_image, 1, 1)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    fps_list.append(fps)
    avg_fps = sum(fps_list) / len(fps_list)

    cv2.putText(combo_image, f"FPS: {fps:.1f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(combo_image, f"Avg FPS: {avg_fps:.1f}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("result", combo_image)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
