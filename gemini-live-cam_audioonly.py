import asyncio
import base64
import traceback
import logging

import pyaudio
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-live-001"

client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=os.getenv("GEMINI_API_KEY"))

tools = [
    types.Tool(google_search=types.GoogleSearch()),
]

CONFIG = types.LiveConnectConfig(
    response_modalities=[types.Modality.AUDIO],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda")
        )
    ),
    tools=types.ToolListUnion(tools),
)

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self):
        logger.info("Initializing AudioLoop class")
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        self.session = None

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                logger.info("User requested to quit.")
                break
            if self.session is not None:
                logger.info(f"Sending user text: {text}")
                await self.session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=text or ".")]
                    )
                )
            else:
                logger.warning("Session is not initialized. Unable to send text.")

    async def send_realtime(self):
        logger.info("send_realtime started") 
        while True:
            msg = await self.out_queue.get()
            logger.debug("send_realtime: got audio chunk from out_queue")
            if self.session is not None:
                logger.debug("Sending realtime audio data")
                await self.session.send_realtime_input(
                    media=types.Blob(data=msg["data"], mime_type=msg["mime_type"])
                )
            else:
                logger.warning("Session is not initialized. Unable to send message.")

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        logger.info(f"Using input device: {mic_info['name']}")
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=int(mic_info["index"]),
            frames_per_buffer=CHUNK_SIZE,
        )
        logger.info("listen_audio started")
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            logger.debug(f"listen_audio: Read {len(data)} bytes from mic")
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        while True:
            if self.session is None:
                await asyncio.sleep(1)
                continue
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    logger.debug("Received audio data from Gemini")
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    logger.info(f"Gemini says: {text.strip()}")
                    print(text, end="")
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        logger.info("Starting audio playback loop")
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            logger.info("Connecting to Gemini live session")
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            logger.info("Cancelled by user.")
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            logger.error("Exception occurred:")
            traceback.print_exception(EG)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')
    logger.info("Starting Gemini Audio Loop")
    main = AudioLoop()
    asyncio.run(main.run())