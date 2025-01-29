import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
import imageio.v3 as iio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        return YOLO('yolov8n.pt')
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_model()

# Virtual boundary
LINE_Y_COORD = 200
LINE_COLOR = (0, 0, 255)  # Red line in RGB
LINE_WIDTH = 5

# Email configuration
SENDER_EMAIL = "notifysystemclg@gmail.com"
RECEIVER_EMAIL = "ssayeeswarnan@gmail.com"
EMAIL_PASSWORD = "wzgu ktek roma hkgl"  # Use app-specific password if 2FA is enabled

def send_email():
    """Send an alert email when a prison break is detected."""
    subject = "Prison Break Detected!"
    body = "Alert: A prison break has been detected by the system."

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Streamlit UI
st.title("YOLO Video Detection Without OpenCV")

# Video file uploader
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file:
    # Save uploaded video temporarily
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())

    stframe = st.empty()
    email_sent = False
    alarm_triggered = False

    # Read frames from video
    reader = iio.imiter("uploaded_video.mp4", plugin="pyav")

    for frame in reader:
        # Convert numpy array to PIL Image for drawing
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)

        try:
            # Run YOLO model on the frame
            results = model(frame)
        except Exception as e:
            st.error(f"Error during model inference: {e}")
            break

        if results[0].boxes:
            detections = results[0].boxes.data.cpu().numpy()
            for box in detections:
                x_min, y_min, x_max, y_max, conf, cls = box
                if int(cls) == 0:  # Class 0 is 'person'
                    bottom_center_y = int(y_max)
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="green", width=3)
                    draw.ellipse([(x_min + x_max) // 2 - 5, bottom_center_y - 5,
                                  (x_min + x_max) // 2 + 5, bottom_center_y + 5], fill="red")
                    if bottom_center_y > LINE_Y_COORD:
                        alarm_triggered = True

        # Draw the virtual boundary line
        draw.line([(0, LINE_Y_COORD), (frame.shape[1], LINE_Y_COORD)], fill=LINE_COLOR, width=LINE_WIDTH)

        if alarm_triggered and not email_sent:
            send_email()
            email_sent = True

        # Display the processed frame in Streamlit
        stframe.image(pil_image)

else:
    st.info("Please upload a video file to start detection.")
