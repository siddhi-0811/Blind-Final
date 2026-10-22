import socket
import subprocess
import tempfile
import os

HOST = "0.0.0.0"
PORT = 5056
MAX_AUDIO_BYTES = 5_000_000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(5)

print(f"Audio receiver listening on port {PORT}", flush=True)

try:
    while True:
        connection, address = server.accept()

        with connection:
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as audio_file:
                path = audio_file.name
                total = 0

                while True:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break

                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        print("Audio too large; rejected.")
                        break

                    audio_file.write(chunk)

        if total == 0 or total > MAX_AUDIO_BYTES:
            os.unlink(path)
            continue

        print(f"Received {total} bytes from {address[0]}", flush=True)

        try:
            subprocess.run(["paplay", path], check=False)
        finally:
            os.unlink(path)

except KeyboardInterrupt:
    print("\nAudio receiver stopped.")

finally:
    server.close()
