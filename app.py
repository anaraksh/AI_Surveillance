import streamlit as st
import cv2
import tempfile
import os
import time
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("yolov8n.pt")
object_tracks = defaultdict(list)
loiter_threshold = 50   
abandon_threshold = 80  

def detect_and_analyze(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    alerts = []
    frame_count = 0
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        video_time = frame_count / fps  # time in seconds
        minutes = int(video_time // 60)
        seconds = int(video_time % 60)
        ts = f"{minutes:02d}:{seconds:02d}"

        results = model(frame)
        current_objects = []
        alert_this_frame = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                current_objects.append((label, cx, cy))

                
                color = (0, 255, 0)
                alert_text = None

                
                object_tracks[label].append((frame_count, cx, cy))

                if label == "person":
                    if len(object_tracks[label]) > loiter_threshold:
                        recent = object_tracks[label][-loiter_threshold:]
                        if max(abs(px - recent[0][1]) for _, px, _ in recent) < 20 and \
                           max(abs(py - recent[0][2]) for _, _, py in recent) < 20:
                            alert_text = "Loitering"
                            alerts.append((ts, "Loitering detected"))
                            color = (0, 0, 255)  # red box
                            alert_this_frame.append(alert_text)


                if label in ["backpack", "suitcase", "handbag"]:
                    if len(object_tracks[label]) > abandon_threshold:
                        alert_text = "Abandoned"
                        alerts.append((ts, f"Object abandonment: {label}"))
                        color = (0, 0, 255)  # red box
                        alert_this_frame.append(alert_text)

                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if alert_text:
                    cv2.putText(frame, alert_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        
        if len(current_objects) > 5:
            alerts.append((ts, "Unusual activity detected"))
            cv2.putText(frame, "Unusual Activity", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Save frame with drawings for later display
        processed_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return alerts, processed_frames

st.title("AI Surveillance System 🚨")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded:
    temp_dir = tempfile.gettempdir()
    tpath = os.path.join(temp_dir, f"{int(time.time())}_{uploaded.name}")

    with open(tpath, "wb") as f:
        f.write(uploaded.read())

    st.info("⏳ Processing video... please wait")
    alerts, frames = detect_and_analyze(tpath)
    st.success(" Video processing complete!")

    
    stframe = st.empty()
    for f in frames:
        stframe.image(f, channels="RGB")
        time.sleep(0.03)

    
    if alerts:
        st.subheader("🚨 Alerts Detected")
        df_alerts = pd.DataFrame(alerts, columns=["Time", "Alert"])
        st.table(df_alerts)

       
        excel_path = os.path.join(tempfile.gettempdir(), "alerts_log.xlsx")
        df_alerts.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            st.download_button("Download Alerts Log (Excel)", f, file_name="alerts_log.xlsx")
    else:
        st.write("No unusual activity detected.")
