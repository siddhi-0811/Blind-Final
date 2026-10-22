
import socket
from pathlib import Path
import smart_aid_windows as ai

PI_HOST = "192.168.19.40"
PI_PORT = 5056
AUDIO_PATH = Path(__file__).resolve().parent / "smart_aid_audio" / "speech.mp3"


def send_audio(text, lang_code="en"):
    # Generate the same natural voice used by the existing Windows AI.
    from gtts import gTTS

    try:
        gTTS(text=text, lang=lang_code).save(str(AUDIO_PATH))

        with socket.create_connection(
            (PI_HOST, PI_PORT), timeout=10
        ) as connection:
            with open(AUDIO_PATH, "rb") as audio:
                while True:
                    chunk = audio.read(4096)
                    if not chunk:
                        break
                    connection.sendall(chunk)

        print(f"Audio sent to Pi: {text[:100]}")

    except Exception as error:
        print(f"Audio bridge error: {error}")


# Redirect AI announcements to the Pi without editing the original file.
ai.say = send_audio

if __name__ == "__main__":
    ai.main()
