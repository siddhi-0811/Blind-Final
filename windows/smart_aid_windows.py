"""
Smart Aid for Windows: real-time object detection + OCR + voice assistance.

Controls:
  SPACE = capture the current frame, read visible text, and speak it
  D     = toggle object detection on/off
  V     = toggle automatic object/direction announcements on/off
  Q/ESC = quit

Notes:
- This is a Windows/laptop adaptation, not the original Raspberry Pi GPIO program.
- YOLO downloads yolov8n.pt on first run if it is not already present (internet required).
- EasyOCR model files may download on first run. Google Text-to-Speech requires internet.
- Object detection is not a reliable distance sensor or collision-avoidance system.
"""

import os
import time
import queue
import logging
import threading
import urllib.request
from pathlib import Path

import cv2
import pytesseract  # kept installed, but EasyOCR handles text recognition now
from PIL import Image
import easyocr
from gtts import gTTS
# Translation and language-detection libraries are not needed in English-only mode.
import pygame
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("SmartAidWindows")

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "smart_aid_audio"
AUDIO_DIR.mkdir(exist_ok=True)

# Optional known-person recognition: known_faces/<Name>/*.jpg
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
MODELS_DIR = BASE_DIR / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
FACE_MATCH_THRESHOLD = 0.40
FACE_ANNOUNCEMENT_RESET_SECONDS = 3.0

# Configure Tesseract on common Windows installations.
tesseract_env = os.environ.get("TESSERACT_EXE")
if tesseract_env:
    pytesseract.pytesseract.tesseract_cmd = tesseract_env
else:
    common_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if common_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(common_tesseract)

# English-only configuration to keep setup simple.
LANGUAGES = {
    "en": {"name": "English", "tts": "en"},
}
OCR_CODES = ("eng",)

# Only show/announce objects relevant to everyday assistance.
# Objects outside this list (for example, sandwich or refrigerator) are ignored.
RELEVANT_OBJECTS = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "traffic light", "fire hydrant", "stop sign", "bench",
    "cat", "dog",
    "backpack", "umbrella", "handbag", "suitcase",
    "bottle", "cup", "bowl",
    "chair", "couch", "bed", "dining table", "toilet",
    "potted plant", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "book", "clock", "scissors",
}

speech_queue = queue.Queue()
last_object_announcement = {}
announce_objects = True

# Direction is estimated from horizontal position in the webcam image.
# These boundaries leave a central zone for "ahead".
LEFT_BOUNDARY = 0.35
RIGHT_BOUNDARY = 0.65


def available_ocr_languages():
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR was not found. Check the Tesseract installation/path."
        ) from exc

    available = [code for code in OCR_CODES if code in installed]
    if not available:
        raise RuntimeError("No supported OCR language data found (eng/hin/mar).")
    absent = [code for code in OCR_CODES if code not in installed]
    if absent:
        log.warning("OCR language data missing: %s", ", ".join(absent))
    return available


def speech_worker():
    """Generate and play queued speech without freezing the webcam display."""
    audio_path = AUDIO_DIR / "speech.mp3"
    while True:
        item = speech_queue.get()
        if item is None:
            speech_queue.task_done()
            break
        text, lang_code = item
        try:
            lang_code = lang_code if lang_code in LANGUAGES else "en"
            gTTS(text=text, lang=LANGUAGES[lang_code]["tts"]).save(str(audio_path))
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
        except Exception as exc:
            log.warning("Speech failed (internet/audio may be unavailable): %s", exc)
        finally:
            speech_queue.task_done()


def say(text, lang_code="en"):
    text = " ".join(str(text).split())
    if text:
        speech_queue.put((text[:700], lang_code))


def get_ocr_language(text):
    # This version intentionally supports English only.
    return "en"


def read_text_from_frame(frame, ocr_reader):
    """
    Detect and recognize English text with EasyOCR.
    EasyOCR returns detected text regions and a recognition confidence for each.
    """
    try:
        # Upscale modestly for small text; do not apply aggressive thresholding.
        height, width = frame.shape[:2]
        if max(height, width) < 1400:
            scale = min(1.7, 1400 / max(height, width))
            image = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        else:
            image = frame

        detections = ocr_reader.readtext(
            image,
            detail=1,
            paragraph=False,
            decoder="greedy",
            text_threshold=0.65,
            low_text=0.35,
            link_threshold=0.35,
            mag_ratio=1.2,
        )

        # Keep detected words in approximate top-to-bottom, left-to-right order.
        items = []
        for bbox, text, confidence in detections:
            text = text.strip()
            if not text or confidence < 0.30:
                continue
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            items.append((int(min(ys) // 15), int(min(xs)), text, float(confidence)))

        items.sort(key=lambda item: (item[0], item[1]))
        recognized_text = " ".join(item[2] for item in items).strip()

        print("\\n" + "=" * 60)
        if not recognized_text:
            print("OCR: No clear English text detected. Try holding the text steady, closer, and in good light.")
            print("=" * 60)
            say("No clear text detected. Hold the text steady, move closer, and improve the lighting.")
            return

        print("OCR TEXT (EasyOCR):\\n" + recognized_text)
        print("\\nRecognition confidence by text region:")
        for _, _, text, confidence in items:
            print(f"  {confidence:.0%}  {text}")
        print("=" * 60)
        say(recognized_text, "en")

    except Exception as exc:
        log.exception("EasyOCR failed")
        print(f"OCR ERROR: {exc}")



def get_object_direction(x1, x2, frame_width):
    """
    Estimate whether a detected object is left, ahead, or right in the image.
    This describes image position only; it does not estimate distance or safety.
    """
    center_x = (x1 + x2) / 2.0
    relative_x = center_x / max(frame_width, 1)

    # Swapped to match the left/right orientation observed on this webcam.
    if relative_x < LEFT_BOUNDARY:
        return "on your right"
    if relative_x > RIGHT_BOUNDARY:
        return "on your left"
    return "ahead"


# Non-person object categories are announced once while visible.
# People are tracked individually so a second person can be announced separately.
OBJECT_ABSENCE_RESET_SECONDS = 3.0
last_seen_by_class = {}
announced_classes = set()
last_seen_person_tracks = {}
announced_person_tracks = set()


def announce_detected_objects(detections):
    """
    Each person is announced once per YOLO tracking ID while visible.
    Other object categories are announced once while that category remains visible.
    detections items are (name, direction, track_id).
    """
    now = time.time()
    visible_names = set()
    visible_person_tracks = set()

    for name, direction, track_id in detections:
        if name == "person" and track_id is not None:
            visible_person_tracks.add(track_id)
        else:
            visible_names.add(name)

    # Re-arm non-person object categories after they have been absent for a while.
    for name in list(announced_classes):
        if name not in visible_names:
            last_seen = last_seen_by_class.get(name, 0)
            if now - last_seen >= OBJECT_ABSENCE_RESET_SECONDS:
                announced_classes.discard(name)

    # Re-arm individual people after their tracking IDs have disappeared long enough.
    for track_id in list(announced_person_tracks):
        if track_id not in visible_person_tracks:
            last_seen = last_seen_person_tracks.get(track_id, 0)
            if now - last_seen >= OBJECT_ABSENCE_RESET_SECONDS:
                announced_person_tracks.discard(track_id)

    for name, direction, track_id in detections:
        if name == "person" and track_id is not None:
            last_seen_person_tracks[track_id] = now
            if track_id not in announced_person_tracks:
                say(f"Person {direction}", "en")
                announced_person_tracks.add(track_id)
        elif name != "person":
            last_seen_by_class[name] = now
            if name not in announced_classes:
                say(f"{name} {direction}", "en")
                announced_classes.add(name)


def download_model_if_needed(destination, url):
    if destination.exists() and destination.stat().st_size > 10_000:
        return
    MODELS_DIR.mkdir(exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading face model: {destination.name} (first run only)...")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "SmartAidWindows/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as out:
            out.write(response.read())
        if temporary.stat().st_size < 10_000:
            raise RuntimeError("Downloaded model file is unexpectedly small.")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_face_recognition():
    """Load consenting registered faces; gracefully fall back if not configured."""
    if not KNOWN_FACES_DIR.exists():
        print(f"Known-face recognition is not configured. Create folders such as {KNOWN_FACES_DIR}\\Siddhi and {KNOWN_FACES_DIR}\\Siya.")
        return None

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folders = [p for p in KNOWN_FACES_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not any(f.is_file() and f.suffix.lower() in extensions for d in folders for f in d.iterdir()):
        print("No known-face photos found; generic person announcements remain enabled.")
        return None

    try:
        download_model_if_needed(YUNET_PATH, YUNET_URL)
        download_model_if_needed(SFACE_PATH, SFACE_URL)
        if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
            raise RuntimeError("This OpenCV build lacks face APIs. Install opencv-contrib-python matching your OpenCV version.")

        detector = cv2.FaceDetectorYN_create(str(YUNET_PATH), "", (320, 320), 0.65, 0.3, 5000)
        recognizer = cv2.FaceRecognizerSF_create(str(SFACE_PATH), "")
        known = {}

        for folder in folders:
            embeddings = []
            for image_path in folder.iterdir():
                if not image_path.is_file() or image_path.suffix.lower() not in extensions:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    log.warning("Could not read face image: %s", image_path.name)
                    continue
                h, w = image.shape[:2]
                detector.setInputSize((w, h))
                _, faces = detector.detect(image)
                if faces is None or len(faces) == 0:
                    log.warning("No face found in registration image: %s", image_path.name)
                    continue
                face = max(faces, key=lambda row: float(row[2] * row[3]))
                embeddings.append(recognizer.feature(recognizer.alignCrop(image, face)))
            if embeddings:
                known[folder.name.strip()] = embeddings
            else:
                log.warning("No usable face photos found for %s.", folder.name)

        if not known:
            print("No usable face photos; generic person announcements remain enabled.")
            return None
        print("Registered people loaded: " + ", ".join(sorted(known)))
        print("Face matching is approximate; uncertain matches are labelled unknown.")
        return detector, recognizer, known
    except Exception as exc:
        log.warning("Known-face recognition could not start: %s", exc)
        print("Continuing with generic person announcements.")
        return None


def recognize_faces_in_frame(frame, face_system):
    if face_system is None:
        return []
    detector, recognizer, known = face_system
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []

    output = []
    for face in faces:
        x, y, fw, fh = [int(v) for v in face[:4]]
        feature = recognizer.feature(recognizer.alignCrop(frame, face))
        best_name, best_score = None, -1.0
        for person_name, embeddings in known.items():
            for embedding in embeddings:
                score = float(recognizer.match(feature, embedding, cv2.FaceRecognizerSF_FR_COSINE))
                if score > best_score:
                    best_name, best_score = person_name, score
        if best_score < FACE_MATCH_THRESHOLD:
            best_name = None
        direction = get_object_direction(x, x + fw, w)
        output.append((best_name, direction, (x, y, fw, fh), best_score))
    return output


face_last_seen = {}
face_announced = set()
last_pair_seen = {}
announced_pairs = set()


def announce_recognized_faces(recognitions):
    """Announce individual known people, then announce known pairs once while together."""
    now = time.time()
    visible_names = {name for name, _direction, _box, _score in recognitions if name}
    visible_names = set(visible_names)

    # Re-arm individual announcements only after a person has been absent briefly.
    for name in list(face_announced):
        if name not in visible_names and now - face_last_seen.get(name, 0) >= FACE_ANNOUNCEMENT_RESET_SECONDS:
            face_announced.discard(name)

    for name, direction, _box, _score in recognitions:
        if name:
            face_last_seen[name] = now
            if name not in face_announced:
                say(f"{name} is {direction}", "en")
                face_announced.add(name)

    # If two or more registered people are visible together, announce each pair once.
    names = sorted(visible_names)
    current_pairs = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = (names[i], names[j])
            current_pairs.add(pair)
            last_pair_seen[pair] = now
            if pair not in announced_pairs:
                say(f"{pair[0]} is with {pair[1]}", "en")
                announced_pairs.add(pair)

    for pair in list(announced_pairs):
        if pair not in current_pairs and now - last_pair_seen.get(pair, 0) >= FACE_ANNOUNCEMENT_RESET_SECONDS:
            announced_pairs.discard(pair)


def match_recognized_faces_to_tracks(recognitions, person_boxes):
    """Match recognized faces to the most plausible YOLO person track."""
    matched = {}
    used_tracks = set()

    known_faces = [(name, box, score) for name, _direction, box, score in recognitions if name]
    for name, (fx, fy, fw, fh), _score in known_faces:
        face_cx = fx + fw / 2
        face_cy = fy + fh / 2
        best_track = None
        best_distance = float("inf")

        for track_id, (x1, y1, x2, y2), _direction in person_boxes:
            if track_id in used_tracks:
                continue

            # A face normally sits in the upper part of the full-body person box.
            head_region_bottom = y1 + (y2 - y1) * 0.55
            if not (x1 - fw * 0.5 <= face_cx <= x2 + fw * 0.5):
                continue
            if not (y1 - fh <= face_cy <= head_region_bottom + fh * 0.25):
                continue

            person_cx = (x1 + x2) / 2
            person_head_y = y1 + (y2 - y1) * 0.2
            scale = max(fw, fh, 1)
            distance = ((face_cx - person_cx) ** 2 + (face_cy - person_head_y) ** 2) ** 0.5 / scale
            if distance < best_distance:
                best_distance = distance
                best_track = track_id

        if best_track is not None:
            matched[best_track] = name
            used_tracks.add(best_track)

    # Reliable fallback for the common case: exactly one person and one known face.
    # This prevents "person" being announced just because the two detectors' boxes differ.
    if len(person_boxes) == 1 and len(known_faces) == 1 and not matched:
        matched[person_boxes[0][0]] = known_faces[0][0]

    return matched


def main():
    global announce_objects
    print("Starting Smart Aid for Windows...")
    print("Features: filtered YOLO object detection, left/right/ahead guidance, EasyOCR English OCR, and English voice output.")
    print("SPACE = read text | D = toggle detection | V = toggle voice alerts | Q = quit")
    print("Object announcements include approximate image position: left, ahead, or right.")
    print("Each tracked person is announced separately; non-person categories are announced once while visible.")
    print("Optional face recognition: add consenting photos to known_faces/<Name>/ to enable names.")
    print("Only selected useful object classes are shown; confidence threshold is 55%.")
    print("Directions describe camera-image position, not physical distance or a safe route.")

    print("Loading English OCR model (EasyOCR, CPU mode)...")
    try:
        ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:
        print(f"Could not initialize EasyOCR: {exc}")
        print("Try: python -m pip install easyocr")
        input("Press Enter to exit...")
        return

    print("Loading YOLO object-detection model (yolov8n.pt)...")
    try:
        model = YOLO("yolov8n.pt")
    except Exception as exc:
        print(f"Could not load YOLO. Check internet/model installation. Details: {exc}")
        input("Press Enter to exit...")
        return

    face_system = initialize_face_recognition()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Close other camera apps and check Windows camera permissions.")
        return

    threading.Thread(target=speech_worker, daemon=True).start()
    detection_enabled = True
    frame_count = 0
    last_names = []
    print("Webcam ready.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            frame_count += 1
            display = frame.copy()
            current_detections = []
            person_boxes = []

            # Run inference every third frame to keep the laptop responsive.
            if detection_enabled and frame_count % 3 == 0:
                try:
                    result = model.track(frame, imgsz=640, conf=0.55, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
                    if result.boxes is not None:
                        for box_index, box in enumerate(result.boxes):
                            cls_id = int(box.cls[0].item())
                            confidence = float(box.conf[0].item())
                            name = model.names.get(cls_id, str(cls_id))
                            # Ignore classes outside our assistance-focused allowlist.
                            if name not in RELEVANT_OBJECTS:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            direction = get_object_direction(x1, x2, frame.shape[1])
                            direction_label = {
                                "on your left": "LEFT",
                                "ahead": "AHEAD",
                                "on your right": "RIGHT",
                            }[direction]
                            track_id = None
                            if name == "person" and result.boxes.id is not None:
                                track_id = int(result.boxes.id[box_index].item())
                            current_detections.append((name, direction, track_id))
                            if name == "person" and track_id is not None:
                                person_boxes.append((track_id, (x1, y1, x2, y2), direction))
                            # Defer drawing person boxes until face recognition has
                            # a chance to replace the generic "person" label with a name.
                            if name != "person":
                                cv2.rectangle(display, (x1, y1), (x2, y2), (40, 220, 40), 2)
                                label = f"{name} {confidence:.0%} - {direction_label}"
                                cv2.putText(
                                    display, label, (x1, max(24, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 40), 2
                                )
                    if face_system is not None:
                        face_results = recognize_faces_in_frame(frame, face_system)
                        if announce_objects:
                            announce_recognized_faces(face_results)
                        if face_results:
                            visible_known = [name for name, _direction, _box, _score in face_results if name]
                            print(f"Faces detected: {len(face_results)} | Recognized: {', '.join(visible_known) if visible_known else 'none'}")
                        recognized_tracks = match_recognized_faces_to_tracks(face_results, person_boxes)

                        # Draw exactly one box per detected person. If a face matches
                        # a registered identity, label the full person box with that name;
                        # otherwise keep the generic person label.
                        for track_id, (x1, y1, x2, y2), direction in person_boxes:
                            direction_label = {
                                "on your left": "LEFT",
                                "ahead": "AHEAD",
                                "on your right": "RIGHT",
                            }[direction]
                            person_name = recognized_tracks.get(track_id)
                            label = (f"{person_name} - {direction_label}" if person_name
                                     else f"person #{track_id} - {direction_label}")
                            cv2.rectangle(display, (x1, y1), (x2, y2), (40, 220, 40), 2)
                            cv2.putText(
                                display, label, (x1, max(24, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 40), 2
                            )

                        # Draw separate face boxes only for faces not matched to a person
                        # box, to avoid the duplicate name + person frames.
                        for face_name, _face_direction, (fx, fy, fw, fh), face_score in face_results:
                            if face_name and any(
                                recognized_tracks.get(track_id) == face_name
                                and x1 <= fx + fw / 2 <= x2 and y1 <= fy + fh / 2 <= y2
                                for track_id, (x1, y1, x2, y2), _direction in person_boxes
                            ):
                                continue
                            face_label = f"{face_name} {face_score:.0%}" if face_name else f"Unknown face {face_score:.0%}"
                            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (255, 180, 0), 2)
                            cv2.putText(display, face_label,
                                        (fx, max(24, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65, (255, 180, 0), 2)
                        # Suppress the generic announcement for a YOLO person
                        # when a registered face is spatially inside that person's
                        # head/upper-body region. This also works when the two
                        # detectors disagree about tracking IDs.
                        known_faces = [
                            (name, direction, box)
                            for name, direction, box, _score in face_results
                            if name
                        ]
                        filtered_detections = []

                        for detection in current_detections:
                            name, direction, track_id = detection
                            if name != "person":
                                filtered_detections.append(detection)
                                continue

                            # A direct track-to-name match is the strongest signal.
                            if track_id is not None and track_id in recognized_tracks:
                                continue

                            person_box = next(
                                (
                                    box
                                    for tid, box, _person_direction in person_boxes
                                    if tid == track_id and track_id is not None
                                ),
                                None,
                            )
                            matched_known_face = False

                            if person_box is not None:
                                x1, y1, x2, y2 = person_box
                                person_height = max(1, y2 - y1)
                                for _face_name, face_direction, (fx, fy, fw, fh) in known_faces:
                                    face_cx = fx + fw / 2
                                    face_cy = fy + fh / 2
                                    face_in_head_region = (
                                        x1 - fw * 0.5 <= face_cx <= x2 + fw * 0.5
                                        and y1 - fh <= face_cy <= y1 + person_height * 0.60
                                    )
                                    if face_in_head_region:
                                        matched_known_face = True
                                        break

                            # Fallback for a single detected person and one known identity.
                            if (
                                not matched_known_face
                                and len(person_boxes) == 1
                                and len(known_faces) == 1
                            ):
                                matched_known_face = True

                            if not matched_known_face:
                                filtered_detections.append(detection)
                    else:
                        recognized_tracks = {}
                        for track_id, (x1, y1, x2, y2), direction in person_boxes:
                            direction_label = {
                                "on your left": "LEFT",
                                "ahead": "AHEAD",
                                "on your right": "RIGHT",
                            }[direction]
                            cv2.rectangle(display, (x1, y1), (x2, y2), (40, 220, 40), 2)
                            cv2.putText(
                                display, f"person #{track_id} - {direction_label}",
                                (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (40, 220, 40), 2
                            )
                        filtered_detections = current_detections

                    if announce_objects:
                        # Call even with an empty list so disappeared objects can reset.
                        announce_detected_objects(filtered_detections)
                except Exception as exc:
                    log.warning("Object detection frame failed: %s", exc)

            status = "ON" if detection_enabled else "OFF"
            voice_status = "ON" if announce_objects else "OFF"
            cv2.putText(
                display, f"Detection: {status} | Voice alerts: {voice_status}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2
            )
            cv2.putText(
                display, "SPACE: OCR | D: detection | V: voice | Q: quit",
                (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 220, 255), 2
            )
            cv2.imshow("Smart Aid - Object Detection + OCR", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("d"):
                detection_enabled = not detection_enabled
                print(f"Object detection {'enabled' if detection_enabled else 'disabled'}.")
            elif key == ord("v"):
                announce_objects = not announce_objects
                print(f"Automatic object voice alerts {'enabled' if announce_objects else 'disabled'}.")
            elif key == 32:
                read_text_from_frame(frame.copy(), ocr_reader)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech_queue.put(None)
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        print("Smart Aid closed.")


if __name__ == "__main__":
    main()
