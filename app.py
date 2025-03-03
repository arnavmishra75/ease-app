import asyncio
import base64
import io
import os
import threading
from dotenv import load_dotenv
from flask import Flask, render_template, redirect, url_for, session
from flask_socketio import SocketIO, emit
from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions
from hume import MicrophoneInterface, Stream
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import string
import re

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # change later
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
filler_word_count = 0  # Initialize filler word counter
all_emotion_scores_overalls = {}
positive_emotion_scores = {"sympathy": [], "calmness": [], "interest": []}
negative_emotion_scores = {"doubt": [], "anxiety": [], "distress": []}

# --- Utility Functions ---

def extract_top_n_emotions(emotion_scores: dict, n: int) -> dict:
    """Extracts the top N emotions from the emotion scores."""
    sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
    top_n_emotions = dict(sorted_emotions[:n])
    return top_n_emotions

def update_key_emotion_deltas(key_emotion_deltas: dict, emotion_scores: dict):
    """Updates the key emotion deltas based on the current emotion scores."""
    for key, values in key_emotion_deltas.items():
        values.append(emotion_scores[key])

def update_emotion_scores_overall(emotion_scores_overall: dict, emotion_scores: dict):
    """Updates the emotion scores overall dictionary."""
    for key, value in emotion_scores.items():
        if key in emotion_scores_overall:
            emotion_scores_overall[key] += value
        else:
            emotion_scores_overall[key] = value

def create_emotion_bar_chart(emotion_scores: dict):
    """Generates a bar chart of emotion scores and returns it as a base64 encoded string."""
    try:
        # Add Roboto font
        font_path = os.path.abspath("static/fonts/Roboto-Regular.ttf")  
        font_prop = font_manager.FontProperties(fname=font_path)
        
        top_3_emotions = extract_top_n_emotions(emotion_scores, 3)
        emotions = list(top_3_emotions.keys())
        scores = list(top_3_emotions.values())

        # Create the bar chart
        plt.figure(figsize=(8, 4))
        bars = plt.bar(emotions, scores, color='#84d8b7')  # green color
        plt.xlabel('Emotions', fontproperties=font_prop, fontsize=12, fontweight='bold')
        plt.ylabel('Scores', fontproperties=font_prop, fontsize=12, fontweight='bold')
        plt.title('Current Top 3 Emotions', fontproperties=font_prop, fontsize=14, fontweight='bold')
        plt.ylim(0, max(scores) + 0.1)  # Adjust y-axis limit to make space for value labels
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop, fontsize=10, fontweight='bold')  # Rotate emotion labels for readability
        plt.yticks(fontproperties=font_prop, fontsize=10, fontweight='bold')
        plt.tight_layout()

        # Add value labels above the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', 
                     fontproperties=font_prop, fontsize=10, fontweight='bold')

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()  # Close the plot to free memory

        # Encode as base64
        return base64.b64encode(img.read()).decode()
    except Exception as e:
        print(f"Error creating emotion chart: {e}")
        return None

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def connect():
    print('Client connected')
    start_hume_connection()
    global filler_word_count
    filler_word_count = 0

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
    global filler_word_count
    scores = {}
    filler_words = ["like", "well", "so", "um", "ah", "er", "you know", "i mean"]

    if message.type in ["user_message", "assistant_message"]:
        role = "E.A.S.E" if message.message.role.upper() == "ASSISTANT" else "You"
        message_text = message.message.content
        
        # Replace standalone "ease" variations with "E.A.S.E"
        ease_vars = [r'\bease\b', r'\bease\.\b', r'\bease\?\b', r'\bease\!\b', r'\bease\,\b',
                     r'\bEase\b', r'\bEase\.\b', r'\bEase\?\b', r'\bEase\!\b', r'\bEase\,\b']
        for var in ease_vars:
            message_text = re.sub(var, "E.A.S.E", message_text)
        
        # Count filler words (only for user messages)
        if role == "You":
            words = message_text.lower().split()
            # Remove punctuation from each word before checking
            words = [word.strip(string.punctuation) for word in words]
            filler_word_count += sum(1 for word in words if word in filler_words)

        text = f"{role}: {message_text}"
        socketio.emit('update_transcript', {'data': text})
        socketio.emit('update_filler_count', {'count': filler_word_count})

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
        # update emotion scores
        update_key_emotion_deltas(positive_emotion_scores, scores)
        update_key_emotion_deltas(negative_emotion_scores, scores)
        update_emotion_scores_overall(all_emotion_scores_overalls, scores)
        # generate the emotion bar chart
        img_data = create_emotion_bar_chart(scores)
        if img_data:
            socketio.emit('update_emotions', {'image': f'data:image/png;base64,{img_data}'})
        else:
            socketio.emit('hume_error', {'message': "Failed to create emotion chart"})
        #print(positive_emotion_scores, negative_emotion_scores, all_emotion_scores_overalls)

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
    global hume_socket, main_loop, microphone_task, byte_strs, hume_connected, filler_word_count
    hume_socket = None
    hume_connected = False
    if main_loop:
        try:
            main_loop.stop()
            main_loop.close()
        except:
            pass
    # reset everything now that new session starting!
    main_loop = None
    microphone_task = None
    byte_strs = Stream.new()  #
    filler_word_count = 0 
    all_emotion_scores_overalls = {} 
    positive_emotion_scores = {"sympathy": [], "calmness": [], "interest": []}
    negative_emotion_scores = {"doubt": [], "anxiety": [], "distress": []}

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
    return render_template('index.html')  # Conversation page

# --- Main ---
if __name__ == '__main__':
    socketio.run(app, debug=True)
