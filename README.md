## What does this app do?

**rPPG Demo** is a desktop demo app that uses a **webcam** to estimate your **heart rate (BPM)** based on **rPPG** (*remote Photoplethysmography*). Instead of using a sensor on the skin, the app analyzes subtle, natural color changes in the face that are related to blood flow.

At a high level, the app works as follows:

- Captures live video from the webcam.
- Detects and tracks the **face** (and relevant skin regions) in the image.
- Extracts an **rPPG signal** (a time series that reflects the pulse) from the video stream.
- Processes this signal to calculate and display a **heart-rate estimate**.

### What is this for?

This application is intended as a **demo/example** to showcase rPPG signal analysis using common computer-vision tooling. The results are **indicative** and can be affected by factors such as lighting, motion, camera quality, and which skin region is visible.
The demo is not validated for use in practice.

### What do you need?

- A working **webcam**
- Sufficient and stable **lighting**
- A person sitting relatively still in front of the camera (less movement generally yields better measurements)


### How to build from source?

Run the following commands:
```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --collect-all mediapipe --name "rPPG Demo" --icon="icon.png" main
```