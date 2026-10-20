import cv2
import time
import mediapipe as mp
import os
import urllib.request

# ============================================================
# MODEL DOWNLOAD
# ============================================================

# BlazeFace Short Range model
model_url = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)

model_path = "face_detection/blaze_face_short_range.tflite"

# Create folder if it doesn't exist
os.makedirs("face_detection", exist_ok=True)

# Download model if it doesn't already exist
if not os.path.exists(model_path):
    print("Downloading BlazeFace Short Range model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded!")
else:
    print("BlazeFace model already exists.")


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

# Hand detector
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

# Face detector
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

# Running mode
VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# HAND LANDMARKER
# ============================================================

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_wave_detection-1/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = HandLandmarker.create_from_options(hand_options)


# ============================================================
# FACE DETECTOR
# ============================================================

face_options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.5
)

face_detector = FaceDetector.create_from_options(face_options)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ============================================================
# WAVE ZONES AND PARAMETERS
# ============================================================

zones = {
    "left": (0.05, 0.35),
    "right": (0.65, 0.95)
}

wiggle_distance = 0.015
crosses_min = 1
smooth_alpha = 0.3
display_time = 1.5

# Static hand parameters
static_time_held = 2.0
static_move_threshold = 0.005


# ============================================================
# HAND WAVE DETECTOR CLASS
# ============================================================

class HandWaveDetector:

    def __init__(self):

        self.counters = {
            "left": 0,
            "right": 0
        }

        self.last_x = {
            "left": None,
            "right": None
        }

        self.direction = {
            "left": None,
            "right": None
        }

    def update(self, x, zone_name):

        if self.last_x[zone_name] is None:

            self.last_x[zone_name] = x

            return False

        waving_detected = False

        delta = x - self.last_x[zone_name]

        if abs(delta) > wiggle_distance:

            new_dir = "right" if delta > 0 else "left"

            if (
                self.direction[zone_name]
                and new_dir != self.direction[zone_name]
            ):

                self.counters[zone_name] += 1

                if self.counters[zone_name] >= crosses_min:

                    waving_detected = True

                    self.counters[zone_name] = 0

                    self.direction[zone_name] = None

            else:

                self.direction[zone_name] = new_dir

        self.last_x[zone_name] = x

        return waving_detected


# ============================================================
# STATE VARIABLES
# ============================================================

wave_detector = HandWaveDetector()

smoothed_x = None

last_wave_time = {
    "left": 0,
    "right": 0
}

static_start_time = {
    "left": None,
    "right": None
}


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while cap.isOpened():

    # --------------------------------------------------------
    # Get camera frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    # ========================================================
    # HAND DETECTION
    # ========================================================

    result = landmarker.detect(mp_image)


    # ========================================================
    # FACE DETECTION
    # ========================================================

    face_result = face_detector.detect(mp_image)

    if face_result.detections:

        # Display face detected
        cv2.putText(
            frame,
            "FACE DETECTED",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3
        )

        # Draw face bounding boxes
        for detection in face_result.detections:

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y

            w = bbox.width
            h = bbox.height

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )


    # ========================================================
    # DRAW WAVE ZONES
    # ========================================================

    for name, (x_min, x_max) in zones.items():

        zone_x1 = int(
            x_min * frame_width
        )

        zone_x2 = int(
            x_max * frame_width
        )

        cv2.rectangle(
            frame,
            (zone_x1, 0),
            (zone_x2, frame_height),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{name.upper()} ZONE",
            (zone_x1 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # HAND WAVE PROCESSING
    # ========================================================

    if result.hand_landmarks:

        # Use first detected hand
        landmarks = result.hand_landmarks[0]

        # Palm landmarks
        palm_ids = [
            0,
            1,
            5,
            9,
            13,
            17
        ]

        # Average X coordinate of palm
        x = sum(
            landmarks[i].x
            for i in palm_ids
        ) / len(palm_ids)


        # ----------------------------------------------------
        # Smooth X position
        # ----------------------------------------------------

        if smoothed_x is None:

            smoothed_x = x

        else:

            smoothed_x = (
                smooth_alpha * x
                + (1 - smooth_alpha) * smoothed_x
            )


        # ----------------------------------------------------
        # Check zones
        # ----------------------------------------------------

        for name, (x_min, x_max) in zones.items():

            if x_min <= smoothed_x <= x_max:

                # --------------------------------------------
                # Wave detection
                # --------------------------------------------

                if wave_detector.update(
                    smoothed_x,
                    name
                ):

                    last_wave_time[name] = time.time()

                    static_start_time[name] = None


                # --------------------------------------------
                # Static hand detection
                # --------------------------------------------

                else:

                    last_x = wave_detector.last_x[name]

                    if (
                        last_x is not None
                        and abs(smoothed_x - last_x)
                        < static_move_threshold
                    ):

                        if static_start_time[name] is None:

                            static_start_time[name] = time.time()

                        elif (
                            time.time()
                            - static_start_time[name]
                            >= static_time_held
                        ):

                            last_wave_time[name] = time.time()

                            static_start_time[name] = None

                    else:

                        static_start_time[name] = None

            else:

                static_start_time[name] = None


        # ----------------------------------------------------
        # Draw palm landmarks
        # ----------------------------------------------------

        for i in palm_ids:

            px = int(
                landmarks[i].x
                * frame_width
            )

            py = int(
                landmarks[i].y
                * frame_height
            )

            cv2.circle(
                frame,
                (px, py),
                8,
                (0, 255, 255),
                -1
            )


    # ========================================================
    # SHOW WAVING
    # ========================================================

    for idx, name in enumerate(zones.keys()):

        if (
            time.time()
            - last_wave_time[name]
            < display_time
        ):

            cv2.putText(
                frame,
                "WAVING",
                (50, 100 + 50 * idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3
            )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "MediaPipe Face + Palm Wave Detector",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

face_detector.close()
