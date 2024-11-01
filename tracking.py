import cv2
import numpy as np

def track_yellow_object(video_path):
   #create video capture object to read video
   cap = cv2.VideoCapture(video_path)

   max_displacement = 0
   max_frame = 0
   frame_count = 0
   
   #get initial pos
   ret, first_frame = cap.read()
   if not ret:
       print("Error: Couldn't read the video file")
       return 0

   #get spring width
   print("Enter the actual width of spring in mm:")
   spring_width_mm = float(input())
   
   #list to store clicked points
   points = []
   #function that handles mouse clicks for calibration
   def click_event(event, x, y, flags, params):
       if event == cv2.EVENT_LBUTTONDOWN:
           points.append((x, y))
           #draw red dot where clicked
           cv2.circle(first_frame, (x, y), 3, (0, 0, 255), -1)
           if len(points) == 2:
               #draw line between two points
               cv2.line(first_frame, points[0], points[1], (0, 0, 255), 2)
           cv2.imshow('Calibration', first_frame)
   
   print("Click the left and right edges of spring, then space after")
   #show first frame for calibration
   cv2.imshow('Calibration', first_frame)
   #set up mouse callback
   cv2.setMouseCallback('Calibration', click_event)
   #wait for key press
   cv2.waitKey(0)
   #close calibration window
   cv2.destroyWindow('Calibration')
   
   #calculate distance between points in pixels
   spring_width_pixels = np.sqrt((points[1][0] - points[0][0])**2 + 
                                (points[1][1] - points[0][1])**2)
   #calculate conversion ratio from pixels to mm
   mm_per_pixel = spring_width_mm / spring_width_pixels
       
   #convert frame to hsv color space for better color detection
   hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
   #define range of yellow color in hsv
   yellow_lower = np.array([20, 100, 100])
   yellow_upper = np.array([30, 255, 255])
   
   #create binary mask of yellow pixels
   initial_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
   #calculate center of mass of yellow object
   initial_position = cv2.moments(initial_mask)
   
   #if yellow object found, get its initial position
   if initial_position['m00'] != 0:
       initial_x = int(initial_position['m10']/initial_position['m00'])
       initial_y = int(initial_position['m01']/initial_position['m00'])
   else:
       print("Error: No yellow object detected in first frame")
       return 0

   #main processing loop
   while True:
       #read next frame
       ret, frame = cap.read()
       if not ret:
           break
           
       #convert frame to hsv and create yellow mask
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
       
       #find center of yellow object in current frame
       M = cv2.moments(mask)
       if M['m00'] != 0:
           current_x = int(M['m10']/M['m00'])
           current_y = int(M['m01']/M['m00'])
           
           #calculate distance from initial position
           displacement = np.sqrt((current_x - initial_x)**2 + (current_y - initial_y)**2)
           
           #convert pixel displacement to mm
           displacement_mm = displacement * mm_per_pixel
           
           #update maximum displacement if current is larger
           if displacement_mm > max_displacement:
               max_displacement = displacement_mm
               max_frame = frame_count
           
           #draw green dot at center and display measurements
           cv2.circle(frame, (current_x, current_y), 5, (0, 255, 0), -1)
           cv2.putText(frame, f"Displacement: {displacement_mm:.2f} mm", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           cv2.putText(frame, f"Max Displacement: {max_displacement:.2f} mm", (10, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           
       #show processed frame and mask
       cv2.imshow('Original', frame)
       cv2.imshow('Mask', mask)
       
       frame_count += 1
       
       #check for q key to quit
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
   #cleanup
   cap.release()
   cv2.destroyAllWindows()
   print(f"Maximum displacement: {max_displacement:.2f} mm at frame {max_frame}")
   return max_frame

print(track_yellow_object("videos/yellow_spring2.mp4"))
