import cv2
import mediapipe as mp

from analysis import (ROI_LANDMARK_LOCATIONS, extract_bvp_signal_pos, bvp_signal_to_heart_rate,
                      get_roi_values, extract_bvp_signal_green)


# GLOBAL DISPLAY SETTINGS
MAX_NR_OF_FACES = 1
ROI_LANDMARK_COLOR = (0, 255, 0) # GREEN in BGR
DISPLAY_FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_WEIGHT = 1
BVP_EXTRACTION_METHOD = 'pos' # 'green' or 'pos'
FRAME_BUFFER_SIZE = 300

# Create face mesh tracker
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=MAX_NR_OF_FACES,
                                  refine_landmarks=True)

# Select correct BVP signal extraction method
if BVP_EXTRACTION_METHOD == 'green':
    extract_bvp_signal = extract_bvp_signal_green
elif BVP_EXTRACTION_METHOD == 'pos':
    extract_bvp_signal = extract_bvp_signal_pos
else:
    raise ValueError("Unknown BVP signal extraction method!")

# Get webcam stream
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

# Create dictionary to keep track of face mesh landmark locations per face
face_data = {}

# Continue for as long as there is a webcam feed
while cap.isOpened():
    # Read frame
    ret, frame = cap.read()
    if not ret: break

    # Mirror the image for natural display experience
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert from bgr to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extract face mesh landmark points
    results = face_mesh.process(rgb_frame)

    # If no faces were found
    if not results.multi_face_landmarks:
        # Clear face data
        face_data.clear()

        # Show no faces found
        cv2.putText(frame,"Zoeken naar gezichten...", (50, 50),
                    DISPLAY_FONT, 0.8, ROI_LANDMARK_COLOR, FONT_WEIGHT)

    # If faces found
    else:
        # Loop over faces (and landmarks)
        for i, face_lms in enumerate(results.multi_face_landmarks):

            # Create new store for face if needed
            if i not in face_data:
                face_data[i] = {"frames": [], "bpm": "Hartslag bepalen..."}

            # Collect RGB values at landmark locations
            roi_img = get_roi_values(frame, face_lms, h, w)
            if roi_img.size > 0:
                face_data[i]["frames"].append(roi_img)

            # Calculate heart rate (after about FRAME_BUFFER_SIZE frames of data)
            if len(face_data[i]["frames"]) >= FRAME_BUFFER_SIZE:

                # Keep the buffer at most 180 frames long
                face_data[i]["frames"] = face_data[i]["frames"][-FRAME_BUFFER_SIZE:]

                # Extract bvp signal
                bvp = extract_bvp_signal(face_data[i]["frames"], fps)

                # Extract heart rate from BVP signal
                bpm = bvp_signal_to_heart_rate(bvp, fps)

                # Store data in an array
                estimated_bpm = f"{int(bpm)} BPM"
                face_data[i]["bpm"] = estimated_bpm

            # Draw face boundary landmark positions
            for idx in ROI_LANDMARK_LOCATIONS:
                pt = face_lms.landmark[idx]
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (cx, cy), 2, ROI_LANDMARK_COLOR, -1)

            # Show estimated heart rate for every face
            tx, ty = int(face_lms.landmark[10].x * w), int(face_lms.landmark[10].y * h) - 40
            cv2.putText(frame, face_data[i]["bpm"], (tx - 50, ty),
                        DISPLAY_FONT, 0.8, ROI_LANDMARK_COLOR, FONT_WEIGHT)

    # Window settings
    cv2.imshow("rPPG Demo", frame)

    # Allow closing via button or window cross
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    if cv2.getWindowProperty("rPPG Demo", cv2.WND_PROP_VISIBLE) < 1: break

# Stop video stream
cap.release()
cv2.destroyAllWindows()