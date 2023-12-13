import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import streamlit as st
import tempfile

# Loading The Model
model=YOLO('yolov8s.pt')


def main():
    col1, col2 = st.columns([2, 3])  
    with col2:
        st.image('pages/logo.jpg', width=100)
    tab1,tab2,tab3 = st.tabs(['Upload','Images','Video'])
    with tab1:
        # Uploading the Video File
        uploaded_vid = st.file_uploader("Choose a file",type=['mp4','avi'])
        if uploaded_vid is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_vid.read())

            # Passing the Uploaded Video File name to the YOLO Model
            processVid(temp_file.name,tab2,tab3)
            
        

def processVid(fileName,tab2,tab3):


    # Capturing The Video Using OpenCV
    videocap=cv2.VideoCapture(fileName)

    cv2.namedWindow('Vehicle Detector')

    # Result Video Format-- H264 due to its support on browser
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter('result.mp4', fourcc, 30, (1020,500))


    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n") 


    tracker=Tracker()
    flag = 0
    previous_frame = None
    while True:   
        dict = {'cars':0,'bus':0,'bicycle':0,'motorcycle':0}
        ret,frame = videocap.read()
        if not ret:
            break
        frame=cv2.resize(frame,(1020,500))
    
        # Using the Model for Prediction
        results=model.predict(frame)

        a=results[0].boxes.data
        px=pd.DataFrame(a).astype("float")
        list=[]
                
        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            c=class_list[d]
            if 'person' in c:
                continue
            if 'car' in c:
                dict['cars'] += 1
            if 'bus' in c:
                dict['bus'] += 1
            if 'motorcycle' in c:
                dict['motorcycle'] += 1
            if 'bicycle' in c:
                dict['bicycle'] += 1
            list.append([x1,y1,x2,y2])
        bbox_id=tracker.update(list)
        for bbox in bbox_id:
            x3,y3,x4,y4,id=bbox

            # Tracking Vehicles
            cv2.rectangle(frame, (x3, y3), (x4,y4), (0, 255, 255),2)

            # Tracking Speed
            if previous_frame is not None:
                prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                vehicle_center_x = (x3 + x4) // 2
                vehicle_center_y = (y3 + y4) // 2
                
                flow_at_vehicle_center = flow[vehicle_center_y, vehicle_center_x]
                
                speed = abs(flow_at_vehicle_center[0]) + abs(flow_at_vehicle_center[1])
                
                speed = (speed/4)*3.6 + 24

                cv2.putText(frame, f"Speed: {speed:.2f} km/hr", (x3, y3 - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)
            
        previous_frame = frame
        cv2.putText(frame,f"Cars:{dict['cars']},Buses:{dict['bus']},Bicycle:{dict['bicycle']},Motorcycle:{dict['motorcycle']}",
                    (0,100),cv2.FONT_HERSHEY_COMPLEX, 0.8,(0, 0, 0), 2)
        with tab2:
            if flag == 0:
                st.title('Result as Images')
                flag = 1
            st.image(frame,channels='BGR')
        
        cv2.imshow('Vehicle Detector', frame)
        video.write(frame)

        if cv2.waitKey(1)&0xFF==27:
            break
    
    video.release()
    videocap.release()
    cv2.destroyAllWindows()
    with tab3:
        st.title('Result as Video')
        st.video('result.mp4')

if __name__ == "__main__":
    main()

