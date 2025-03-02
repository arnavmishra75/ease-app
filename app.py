import asyncio
import base64
import os
import threading
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for, session
from flask_socketio import SocketIO, emit
from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions
from hume import MicrophoneInterface, Stream

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this in production!
socketio = SocketIO(app, async_mode='threading')

# Load environment variables
load_dotenv()
HUME_API_KEY = os.getenv("HUME_API_KEY")
HUME_SECRET_KEY = os.getenv("HUME_SECRET_KEY")
HUME_CONFIG_ID = os.getenv("HUME_CONFIG_ID")

# Global Variables
hume_socket = None
hume_connected = False
main_loop = None
microphone_task = None
byte_strs = Stream.new()

# --- Utility Functions ---
def extract_top_n_emotions(emotion_scores: dict, n: int) -> dict:
    sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
    top_n_emotions = {emotion: score for emotion, score in sorted_emotions[:n]}
    return top_n_emotions

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def connect():
    print('Client connected')
    start_hume_connection()

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')

@socketio.on('stop_conversation')
def stop_conversation():
    print('Conversation stopped by client')
    stop_hume_connection()
    emit('redirect', {'url': url_for('evaluation')})

# --- Hume API Integration ---
async def hume_websocket_handler(message):
    scores = {}

    if message.type in ["user_message", "assistant_message"]:
        role = "E.A.S.E" if message.message.role.upper() == "ASSISTANT" else "You"
        message_text = message.message.content
        text = f"{role}: {message_text}"
        socketio.emit('update_transcript', {'data': text})

        if message.from_text is False and role == "You":
            scores = dict(message.models.prosody.scores)

    elif message.type == "audio_output":
        message_str: str = message.data
        message_bytes = base64.b64decode(message_str.encode("utf-8"))
        await byte_strs.put(message_bytes)
        return

    elif message.type == "error":
        error_message: str = message.message
        print(f"Error from Hume API: {error_message}")
        socketio.emit('hume_error', {'message': f"Hume API Error: {error_message}"})
        return

    if scores:
        top_3_emotions = extract_top_n_emotions(scores, 3)
        emotion_text = ' | '.join([f"{emotion} ({score:.2f})" for emotion, score in top_3_emotions.items()])
        socketio.emit('update_emotions', {'data': emotion_text})

async def hume_on_open():
    global hume_connected
    hume_connected = True
    print("Hume WebSocket connection opened.")

async def hume_on_close():
    global hume_connected
    hume_connected = False
    print("Hume WebSocket connection closed.")

async def hume_on_error(error):
    global hume_connected
    hume_connected = False
    print(f"Hume WebSocket Error: {error}")
    socketio.emit('hume_error', {'message': f"WebSocket Error: {error}"})

async def hume_microphone_handler(socket):
    global byte_strs
    try:
        await MicrophoneInterface.start(
            socket,
            allow_user_interrupt=False,
            byte_stream=byte_strs
        )
        print("Microphone Started")
    except asyncio.CancelledError:
        print("Microphone task cancelled")
    except RuntimeError as e:
        if "anext(): asynchronous generator is already running" in str(e):
            print("Microphone is already running, ignoring...")
        else:
            print(f"Error in microphone stream: {e}")
            socketio.emit('hume_error', {'message': f"Microphone Error: {e}"})
    except Exception as e:
        print(f"Error in microphone stream: {e}")
        socketio.emit('hume_error', {'message': f"Microphone Error: {e}"})


async def shutdown_microphone():
    global microphone_task
    if microphone_task:
        try:
            microphone_task.cancel()
            await microphone_task
        except asyncio.CancelledError:
            pass
    microphone_task = None

def start_hume_connection():
    global hume_connected
    if not hume_connected:
        threading.Thread(target=run_hume_client, daemon=True).start()

def stop_hume_connection():
    global hume_socket, main_loop, hume_connected, microphone_task
    hume_connected = False

    if main_loop and main_loop.is_running():
        for task in asyncio.all_tasks(main_loop):
            task.cancel()
        main_loop.call_soon_threadsafe(main_loop.stop)
        asyncio.run_coroutine_threadsafe(shutdown_microphone(), main_loop)

    if hume_socket:
        hume_socket = None

    reset_global_variables()

def reset_global_variables():
    global hume_socket, main_loop, microphone_task, byte_strs, hume_connected
    hume_socket = None
    hume_connected = False
    if main_loop:
        try:
            main_loop.stop()
            main_loop.close()
        except:
            pass
    main_loop = None
    microphone_task = None
    byte_strs = Stream.new()  # Reset the byte stream

def run_hume_client():
    global main_loop, hume_socket
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_loop = loop
    loop.run_until_complete(hume_client_task())
    loop.close()
    hume_socket = None

async def hume_client_task():
    global hume_socket, hume_connected, microphone_task
    client = AsyncHumeClient(api_key=HUME_API_KEY)
    options = ChatConnectOptions(config_id=HUME_CONFIG_ID, secret_key=HUME_SECRET_KEY)

    try:
        async with client.empathic_voice.chat.connect_with_callbacks(
            options=options,
            on_open=hume_on_open,
            on_message=hume_websocket_handler,
            on_close=hume_on_close,
            on_error=hume_on_error
        ) as socket:
            hume_socket = socket
            hume_connected = True
            print("Hume client connected.")
            microphone_task = asyncio.create_task(hume_microphone_handler(socket))
            await asyncio.Future()  # Keep the connection open indefinitely
    except Exception as e:
        hume_connected = False
        print(f"Error connecting to Hume: {e}")
        socketio.emit('hume_error', {'message': f"Hume Connection Error: {e}"})

# --- Flask Routes ---
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/new_conversation')
def new_conversation():
    session.clear()  # Clears Flask session variables
    stop_hume_connection()  # Stop the current Hume connection
    start_hume_connection()  # Start a new Hume connection
    return redirect(url_for('chat'))  # Redirect to the chat page

@app.route('/chat')
def chat():
    return render_template('index.html')  # Your existing chat page

# --- Main ---
if __name__ == '__main__':
    socketio.run(app, debug=True)
