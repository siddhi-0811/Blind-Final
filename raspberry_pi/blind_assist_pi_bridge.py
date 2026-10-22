
import socket
import threading
import subprocess
from gpiozero import DistanceSensor
from statistics import median
from time import sleep, monotonic

TRIG_PIN = 23
ECHO_PIN = 24
PORT = 5055

STOP_CM = 40
WARNING_CM = 100
CAUTION_CM = 180

sensor = DistanceSensor(
    echo=ECHO_PIN,
    trigger=TRIG_PIN,
    max_distance=2,
    queue_len=5
)

condition = threading.Condition()
pending_obstacle = None
pending_ai = None
last_category = None
last_stop_time = 0


def obstacle_message(distance):
    if distance <= STOP_CM:
        return "Stop! Obstacle very close!"
    if distance <= WARNING_CM:
        return "Warning! Obstacle nearby."
    if distance <= CAUTION_CM:
        return "Caution! Obstacle ahead."
    return None


def queue_obstacle(message):
    global pending_obstacle
    with condition:
        pending_obstacle = message
        condition.notify()


def queue_ai(message):
    global pending_ai
    with condition:
        pending_ai = message[:120]
        condition.notify()


def speech_worker():
    global pending_obstacle, pending_ai

    while True:
        with condition:
            condition.wait_for(
                lambda: pending_obstacle is not None
                or pending_ai is not None
            )
            if pending_obstacle is not None:
                message = pending_obstacle
                pending_obstacle = None
            else:
                message = pending_ai
                pending_ai = None

        subprocess.run(
            ["espeak-ng", "-v", "en", "-s", "165", message],
            check=False
        )


def receiver():
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", PORT))
    server.listen(5)
    print(f"Windows AI receiver listening on port {PORT}", flush=True)

    while True:
        connection, address = server.accept()
        with connection:
            message = connection.recv(512).decode(
                "utf-8", errors="ignore"
            ).strip()
        if message:
            print(f"AI from {address[0]}: {message}", flush=True)
            queue_ai(message)


def read_distance():
    readings = []
    for _ in range(7):
        try:
            value = sensor.distance * 100
            if 5 <= value <= 180:
                readings.append(value)
        except Exception:
            pass
        sleep(0.1)
    return median(readings) if len(readings) >= 4 else None


threading.Thread(target=speech_worker, daemon=True).start()
threading.Thread(target=receiver, daemon=True).start()

print("Blind Assist Pi bridge started.")
print("Ctrl+C stops the service.")

try:
    while True:
        distance = read_distance()
        if distance is None:
            sleep(0.3)
            continue

        message = obstacle_message(distance)
        now = monotonic()
        print(f"Distance: {distance:.1f} cm | {message or 'No alert'}")

        if message != last_category:
            last_category = message
            if message is not None:
                queue_obstacle(message)
            elif pending_obstacle if False else False:
                pass

        elif message and message.startswith("Stop!") and now - last_stop_time >= 2:
            last_stop_time = now
            queue_obstacle(message)

        sleep(0.2)

except KeyboardInterrupt:
    print("\nPi bridge stopped.")
finally:
    sensor.close()
