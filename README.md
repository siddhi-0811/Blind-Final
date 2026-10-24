# Blind Assist
### AI-Powered Environmental Awareness for the Visually Impaired

Blind Assist is an **AI + IoT assistive technology prototype** that combines computer vision, optical character recognition (OCR), voice announcements, optional face recognition, and ultrasonic proximity sensing. It is designed to provide additional audio-based awareness of selected objects, visible text, and nearby obstacles.

The project uses a **Windows laptop for AI processing** and a **Raspberry Pi 4 Model B for audio playback and ultrasonic sensing**.

> **Safety disclaimer:** Blind Assist is an experimental prototype, not a certified mobility aid. Its detections and sensor alerts may be delayed, inaccurate, or missed. It cannot guarantee collision avoidance and must not replace a white cane, guide dog, or established mobility practices.

---

## Table of Contents

- [Highlights](#highlights)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Stack](#software-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the System](#running-the-system)
- [Controls](#controls)
- [Ultrasonic Alert Logic](#ultrasonic-alert-logic)
- [Network Configuration](#network-configuration)
- [Troubleshooting](#troubleshooting)
- [Privacy and Safety](#privacy-and-safety)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Highlights

- **Object detection:** Detects selected objects in the camera view using Ultralytics YOLO.
- **Approximate directional cues:** Estimates whether a detected object appears to the left, ahead, or right in the image. This is a visual-position estimate, not a reliable physical bearing.
- **Text recognition:** Uses EasyOCR to read visible English text when requested.
- **Voice announcements:** Generates spoken descriptions with Google Text-to-Speech (gTTS).
- **Optional face recognition:** Uses OpenCV YuNet and SFace models to recognize locally registered people when configured.
- **Ultrasonic obstacle alerts:** Uses an HC-SR04 sensor to estimate distance and trigger configurable proximity messages.
- **Remote audio playback:** Sends generated MP3 audio from Windows to the Raspberry Pi over TCP for playback through the Pi's configured audio output.
- **Separated components:** Keeps AI inference on Windows and Raspberry Pi audio/sensor services in separate Python scripts.

## System Architecture

```text
                    CAMERA
                       |
                       v
             +---------------------+
             | Windows Laptop      |
             |---------------------|
             | YOLO object detection|
             | Approx. direction    |
             | EasyOCR text reading |
             | Optional face match  |
             | Speech generation    |
             +---------------------+
                       |
                 TCP port 5056
                 (MP3 audio bytes)
                       |
                       v
             +---------------------+
             | Raspberry Pi 4      |
             |---------------------|
             | Audio receiver      |----> Bluetooth / configured
             |                     |      audio output
             | HC-SR04 sensor      |
             | Obstacle alert logic|
             +---------------------+
```

**Processing split:** The primary AI model runs on Windows. The Raspberry Pi handles audio playback and the ultrasonic sensor bridge. The system is not a fully standalone Raspberry Pi AI device.

## Hardware Requirements

| Component | Purpose |
|---|---|
| Windows laptop | Runs object detection, OCR, face recognition, and speech generation |
| Raspberry Pi 4 Model B | Runs audio receiver and ultrasonic-sensor bridge |
| Camera | Supplies frames to the Windows AI application |
| HC-SR04 ultrasonic sensor | Estimates distance to nearby obstacles |
| Voltage divider or level shifter | Safely adapts the HC-SR04 Echo signal to Pi GPIO voltage |
| Jumper wires | Connects the sensor |
| Bluetooth earbuds or speaker | Provides audio feedback when paired/configured with the Pi |
| Suitable power supplies | Powers the laptop and Pi |

### HC-SR04 Wiring

The Raspberry Pi bridge uses **BCM GPIO numbering**:

| HC-SR04 pin | Raspberry Pi connection |
|---|---|
| VCC | 5V |
| GND | GND |
| TRIG | GPIO 23 (BCM) |
| ECHO | GPIO 24 (BCM), **through a voltage divider or level shifter** |

**Electrical safety:** Raspberry Pi GPIO pins use 3.3 V logic and are not 5 V tolerant. Never connect the HC-SR04 Echo pin directly to a Pi GPIO pin. Check the sensor module specifications and wiring before powering the circuit.

## Software Stack

| Technology | Role |
|---|---|
| Python | Core application language |
| Ultralytics YOLO | Object detection |
| OpenCV (contrib build) | Camera processing and optional face-recognition APIs |
| YuNet / SFace ONNX models | Optional face detection and face feature matching |
| EasyOCR | English text recognition |
| gTTS | Generates speech audio |
| pygame / paplay | Audio-related dependencies and Raspberry Pi playback, as configured |
| gpiozero | Raspberry Pi ultrasonic sensor interface |
| TCP sockets | Windows-to-Pi audio transfer and Pi bridge communication |
| Raspberry Pi OS / Linux | Runs embedded components |
| Windows | Runs the main AI application |

## Repository Structure

```text
Blind-Final/
├── README.md
├── .gitignore
├── windows/
│   ├── smart_aid_windows.py
│   ├── smart_aid_audio_bridge.py
│   ├── requirements.txt
│   ├── yolov8n.pt
│   ├── models/
│   │   ├── face_detection_yunet_2023mar.onnx
│   │   └── face_recognition_sface_2021dec.onnx
│   └── known_faces/             # Local, private face images; do not commit
└── raspberry_pi/
    ├── blind_assist_audio_receiver.py
    └── blind_assist_pi_bridge.py
```

The `known_faces/` directory is optional and should contain only images you are authorized to use. Face-model files may be downloaded by the application if they are not present locally, depending on the script configuration. Confirm the exact local folder structure before running.

## Getting Started

### 1. Prerequisites

- Git installed on Windows.
- Python 3.12 recommended for the current Windows environment and dependency set.
- A working Windows camera.
- Raspberry Pi OS on the Raspberry Pi 4.
- Both devices connected to a network that permits direct device-to-device communication.
- Bluetooth audio already paired/configured on the Raspberry Pi if earbuds are used.
- HC-SR04 wired safely if using obstacle alerts.

### 2. Clone This Repository

Run in PowerShell:

```powershell
git clone https://github.com/siddhi-0811/Blind-Final.git
cd Blind-Final
```

If you have already cloned this project into a local folder named `Blind-Final-Project`, you can keep using that folder; the local folder name does not have to match the GitHub repository name.

### 3. Set Up Windows Dependencies

From the repository root:

```powershell
cd windows
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you already have a working virtual environment, use it rather than creating another one. To activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

The first run may download model or OCR assets. YOLO, EasyOCR, and gTTS may require internet access for initial setup or speech generation. Check `windows/requirements.txt` for pinned package versions.

### 4. Configure Optional Face Recognition

Create a local directory structure such as:

```text
windows/known_faces/
├── PersonOne/
│   ├── front.jpg
│   └── side.jpg
└── PersonTwo/
    └── front.jpg
```

Use clear, reasonably well-lit face images and name each person's folder with the name the application should announce. Exact file/folder expectations depend on the current script. Check the startup logs to confirm the face images were loaded successfully.

Only register people who have given permission. Do not commit personal face images to a public repository.

### 5. Prepare the Raspberry Pi

Clone the repository on the Pi, or copy the two files in `raspberry_pi/` to the device. Install the packages required by the scripts, including `gpiozero` for the ultrasonic bridge and any audio playback tools used by your Raspberry Pi OS installation.

For the included audio receiver, the `paplay` command must be available and the Pi's audio output must be configured. Bluetooth earbuds should be paired and selected as an output device before testing.

The Pi scripts in the repository are:

- `raspberry_pi/blind_assist_audio_receiver.py`
- `raspberry_pi/blind_assist_pi_bridge.py`

## Running the System

Run each service in a separate terminal. Start the Raspberry Pi services first, then run the Windows application.

### Terminal 1 — Raspberry Pi Audio Receiver

On the Raspberry Pi, from the directory containing the repository:

```bash
python3 raspberry_pi/blind_assist_audio_receiver.py
```

The receiver listens on TCP port **5056**, saves incoming audio temporarily as an MP3, and invokes `paplay` for playback.

### Terminal 2 — Raspberry Pi Ultrasonic Bridge

```bash
python3 raspberry_pi/blind_assist_pi_bridge.py
```

The bridge uses the HC-SR04 sensor on BCM GPIO 23 and 24 and provides its configured TCP service on port **5055**. Hardware access and alert delivery require the associated GPIO wiring and network configuration to be correct.

### Terminal 3 — Windows AI Application with Pi Audio

From the repository root in Windows PowerShell, run:

```powershell
.\windows\.venv\Scripts\python.exe .\windows\smart_aid_audio_bridge.py
```

This script imports the Windows AI application and redirects generated voice announcements to the Raspberry Pi audio receiver. Before running, verify that `PI_HOST` in `windows/smart_aid_audio_bridge.py` matches the Pi's **current** IP address. The checked-in default may need to be changed when the network assigns a different address.

### Alternative — Run Windows AI Locally

To run the AI script without routing generated audio through the Pi, use:

```powershell
.\windows\.venv\Scripts\python.exe .\windows\smart_aid_windows.py
```

The Windows application is configured to open the laptop's default camera (index 0), with a fallback capture method. Make sure the intended camera is connected and not being used by another application.

## Controls

The Windows AI application displays these controls in its console:

| Key | Action |
|---|---|
| `SPACE` | Read text from the current frame |
| `D` | Toggle object detection |
| `V` | Toggle voice alerts |
| `Q` | Quit the application |

Controls apply while the Windows AI application is running. Exact behavior may change as the code evolves.

## Ultrasonic Alert Logic

The Raspberry Pi bridge currently uses these configurable example thresholds:

| Measured distance | Alert |
|---|---|
| 40 cm or less | Stop — obstacle very close |
| Over 40 cm and up to 100 cm | Warning — obstacle nearby |
| Over 100 cm and up to 180 cm | Caution — obstacle ahead |
| Over 180 cm | No proximity message within these thresholds |

These alerts are based on sensor measurements, not AI-based scene understanding. Ultrasonic readings can be affected by object size, texture, angle, placement, and the environment. They should not be treated as guaranteed safe distances.

## Network Configuration

| Service | Port | Purpose |
|---|---:|---|
| Raspberry Pi audio receiver | 5056 | Windows sends generated MP3 audio to the Pi |
| Raspberry Pi bridge service | 5055 | Configured TCP service used by the Pi bridge |

Network troubleshooting checklist:

1. Confirm both devices are connected to the same reachable local network. A phone hotspot may enable client isolation or limit peer-to-peer communication.
2. Check the Pi's current IP address; do not assume an old IP is still assigned.
3. Check that each required Python service is running on the Pi.
4. Confirm that `PI_HOST` in `smart_aid_audio_bridge.py` points to the current Pi IP.
5. Test the actual port, for example in PowerShell: `Test-NetConnection <PI_IP> -Port 5056`.
6. A successful ping does not guarantee that a TCP service is listening; similarly, a timeout can result from a stale IP, client isolation, a firewall, or a stopped service.

Replace `<PI_IP>` with the actual IP address; do not type the angle brackets literally.

## Troubleshooting

### Camera does not open
- Close other applications using the camera.
- Check Windows camera permissions.
- Confirm that the camera index configured in `smart_aid_windows.py` matches the camera to use.
- Test the webcam independently before running the AI pipeline.

### YOLO model is missing or fails to load
- Confirm that `windows/yolov8n.pt` exists.
- Run the application from the repository root or verify the path handling if launching from another folder.
- Check the console for model-download or dependency errors.

### OCR is slow or inaccurate
- Use good lighting and keep text steady and in focus.
- Move the text closer and avoid severe angles.
- Allow EasyOCR to finish downloading its model on the first run.
- Use a CPU-compatible environment with enough memory.

### No audio from the Raspberry Pi
- Confirm the audio receiver is running and listening on port 5056.
- Ensure `PI_HOST` is set to the Pi's current IP.
- Pair and select the intended audio output on the Pi.
- Check that `paplay` is installed and works with the chosen output.
- Check the Windows console for `Audio bridge error` messages.

### Ultrasonic alerts do not work
- Verify TRIG/ECHO wiring and the voltage divider.
- Confirm BCM GPIO 23 and 24 are the pins used in the script.
- Run the bridge on the Pi with the required GPIO permissions.
- Make sure the sensor is powered correctly and has a common ground.

### Face recognition does not identify registered people
- Check the folder names and image structure under `windows/known_faces/`.
- Use clear, frontal images.
- Confirm the YuNet and SFace ONNX files are present or can be downloaded.
- Review startup messages for missing folders or unusable photos.

## Privacy and Safety

- Keep registered face images private and out of public commits.
- Do not store passwords, access tokens, or private network details in tracked files.
- Get consent before registering or processing another person's face.
- Test the prototype in a controlled environment, with a sighted supervisor as appropriate.
- Do not use the prototype as the only means of navigating around traffic, stairs, moving vehicles, or other hazards.

## Limitations

- The AI pipeline runs on Windows and depends on camera quality and available CPU/GPU resources.
- Direction cues are estimated from object position in a two-dimensional image; they do not measure true distance or provide dependable spatial localization.
- Object detection and face recognition can produce false positives or miss targets.
- OCR may misread text, especially when it is small, blurred, angled, or poorly lit.
- gTTS speech generation requires internet access.
- Wi-Fi, TCP, Bluetooth, camera, or processing delays can postpone an announcement.
- An ultrasonic sensor covers a limited area and does not detect all obstacles reliably.
- The system has not been certified or validated as an assistive device.

## Future Improvements

- Move more AI inference onto edge hardware where feasible.
- Add robust distance estimation and sensor fusion.
- Improve audio prioritization and reduce repeated announcements.
- Add offline text-to-speech support.
- Improve OCR language support and environmental testing.
- Make network host/port settings configurable rather than hard-coded.
- Develop a compact, ergonomic wearable enclosure and optimize battery usage.
- Conduct structured usability and safety evaluations with appropriate oversight.

## Contributing

Issues and suggestions are welcome. Include the operating system, hardware, relevant console output, and reproducible steps when reporting a problem. Do not attach private face images, credentials, or other sensitive information.

## License

No license is currently specified in this repository. Until a license is added, do not assume that others have permission to reuse, modify, or distribute the code. Add a suitable `LICENSE` file if you intend to publish an explicit open-source license.

## Acknowledgements

Blind Assist explores how computer vision, embedded systems, and audio interfaces can be combined to make environmental information more accessible.
