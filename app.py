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
from keras.saving import load_model
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, patches
import string
import re

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # change later
socketio = SocketIO(app, async_mode='threading')
matplotlib.use('Agg')  # use agg so that charts aren't rendered immediately (instead saved)

# Load the model and encoder
lexical_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
try:
    lexical_model = load_model('models/lexical_model_scaled_outputs.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    lexical_model = None  # Handle model loading failure

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
user_responses = []
total_interactions = 0

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

def flatten_embeddings(df):
    embeddings_df = pd.DataFrame(df['embedding'].tolist())
    embeddings_df.columns = [f'embedding_{i}' for i in range(embeddings_df.shape[1])]
    df = df.drop(columns=['embedding']).join(embeddings_df)
    return df

def get_lexical_score(responses, encoder, model):
    """Calculates the lexical score for a list of responses."""
    if model is None:
        print("Model not loaded, cannot calculate lexical score.")
        return None

    df = pd.DataFrame(responses, columns=["text"])
    df["embedding"] = df["text"].apply(lambda x: encoder.encode(x))
    numerical_df = flatten_embeddings(df).drop(columns=["text"])
    try:
        scores = model.predict(numerical_df)
        total_score = 0
        for score in scores:
            total_score += score[0]
        return np.round((total_score / len(scores))*100, 1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

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
        plt.figure(figsize=(8, 4))  # previous size
        bars = plt.bar(emotions, scores, color='#84d8b7')  # green color
        plt.xlabel('Emotions', fontproperties=font_prop, fontsize=12, fontweight='bold')
        plt.ylabel('Scores', fontproperties=font_prop, fontsize=12, fontweight='bold')
        plt.title('Current Top 3 Emotions', fontproperties=font_prop, fontsize=14, fontweight='bold')
        plt.ylim(0, max(scores) + 0.1)  # Adjust y-axis limit to make space for value labels
        plt.yticks(np.arange(0, max(scores) + 0.1, 0.1), fontproperties=font_prop, fontsize=8, fontweight='bold')  # Y-axis increments of 0.05
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop, fontsize=8, fontweight='bold')  # Rotate emotion labels for readability
        plt.tight_layout() # Add value labels above the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', 
                     fontproperties=font_prop, fontsize=8, fontweight='bold')

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.savefig(img, format='png', dpi=200)
        img.seek(0)
        plt.close()  # Close the plot to free memory

        # Encode as base64
        return base64.b64encode(img.read()).decode()
    except Exception as e:
        print(f"Error creating emotion chart: {e}")
        return None
    
def create_line_chart(emotion_data: dict, title: str, ylabel: str, color_map: dict):
    """
    Generates a line chart for emotion trends over interactions.... Args:
        emotion_data (dict): Dictionary of emotions and their scores over interactions.
        title (str): Title of the chart.
        ylabel (str): Title of the chart.
        color_map (dict): Dictionary mapping emotion names to colors.

    Returns:
        str: Base64 encoded PNG image of the chart.
    """
    try:
        font_path = os.path.abspath("static/fonts/Roboto-Regular.ttf")  
        font_prop = font_manager.FontProperties(fname=font_path)
        
        # Check if there are any emotions with scores
        if not any(emotion_data.values()):  # This checks if all lists are empty
            plt.figure()  # Create a new figure to avoid errors
            plt.text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=12)
            plt.axis('off')  # Turn off axis lines and ticks
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()
            print(f"No data for line chart '{title}'. Returning empty chart.")
            return base64.b64encode(img.read()).decode()
            
        # Determine maximum score for y-axis
        max_score = 0
        for scores in emotion_data.values():
            if scores:
                max_score = max(max_score, max(scores))
        
        y_max = max_score + 0.1  # Add buffer
        y_ticks = np.arange(0, y_max + 0.05, 0.05)  # Generate ticks

        plt.figure(figsize=(6, 3))  # Reduced figure size
        
        # Ensure there are interactions to plot
        num_interactions = len(next(iter(emotion_data.values()), []))
        if num_interactions == 0:
            plt.text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=12)
            plt.axis('off')  # Turn off axis lines and ticks
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()
            return base64.b64encode(img.read()).decode()

        x_values = range(1, num_interactions + 1)  # Interactions start from 1

        for emotion, scores in emotion_data.items():
            if scores:
                plt.plot(x_values, scores, marker='o', linestyle='-', color=color_map[emotion], label=emotion)
        plt.title(title, fontproperties=font_prop, fontsize=15, fontweight='bold', color = "#1d2f4b")
        plt.xlabel('Conversation Progression', fontproperties=font_prop, fontsize=8, fontweight='bold')
        plt.ylabel(ylabel, fontproperties=font_prop, fontsize=8, fontweight='bold')
        plt.xticks(x_values, fontproperties=font_prop, fontsize=6, fontweight='bold')
        plt.yticks(y_ticks, fontproperties=font_prop, fontsize=6, fontweight='bold')
        plt.ylim(0, y_max)  # set upper bond for max
        plt.grid(True)
        plt.legend(prop=font_prop, fontsize=6)  # Use font_prop for the legend
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=200)
        img.seek(0)
        plt.close()

        return base64.b64encode(img.read()).decode()
    except Exception as e:
        print(f"Error creating line chart: {e}")
        return None

def create_top_emotions_bar_chart(emotion_scores: dict, total_interactions: int):
    """
    Generates a bar chart of the top 5 average emotions from the overall scores.

    Args:
        emotion_scores (dict): Dictionary of overall emotion scores.
        total_interactions (int): Total number of interactions.... Returns:
        str: Base64 encoded PNG image of the chart.
    """
    try:
        font_path = os.path.abspath("static/fonts/Roboto-Regular.ttf")  
        font_prop = font_manager.FontProperties(fname=font_path)
        
        # Calculate average scores
        average_emotion_scores = {k: v / total_interactions for k, v in emotion_scores.items()} # Extract top 5 emotions
        top_5_emotions = extract_top_n_emotions(average_emotion_scores, 5)
        emotions = list(top_5_emotions.keys())
        scores = list(top_5_emotions.values())
            
        max_score = max(scores) if scores else 0  # Determine max score for Y axis
        y_max = max_score + 0.1  # Add buffer for chart
        y_ticks = np.arange(0, y_max + 0.05, 0.05)  # Generate ticks

        plt.figure(figsize=(6, 3))  # Increased figure size
        bars = plt.bar(emotions, scores, color='#84d8b7')
        plt.xlabel('Emotions', fontproperties=font_prop, fontsize=8, fontweight='bold')
        plt.ylabel('Average Score', fontproperties=font_prop, fontsize=8, fontweight='bold')
        plt.title('Top 5 Average Emotions', fontproperties=font_prop, fontsize=15, fontweight='bold', color = "#1d2f4b")
        plt.ylim(0, y_max) # add upper bond
        plt.yticks(y_ticks, fontproperties=font_prop, fontsize=6, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontproperties=font_prop, fontsize=6, fontweight='bold')
        plt.tight_layout()

        # Add value labels above the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', 
                     fontproperties=font_prop, fontsize=6, fontweight='bold')

        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=200)
        img.seek(0)
        plt.close()

        # revert to previous size to maintain appearance of main convo page
        #plt.figure(figsize=(8, 4))  # previous size

        return base64.b64encode(img.read()).decode()
    except Exception as e:
        print(f"Error creating top emotions chart: {e}")
        return None

def create_circular_progress_bar(score):
    try:
        font_path = os.path.abspath("static/fonts/Roboto-Regular.ttf")
        font_prop = font_manager.FontProperties(fname=font_path)

        fig, ax = plt.subplots(figsize=(4, 4))  # Keep size consistent

        # Create the circular progress bar
        wedgeprops = {'width': 0.3, 'edgecolor': 'white'}
        ax.pie([score, 100 - score], colors=['#84d8b7', 'white'],
               startangle=90, counterclock=False, wedgeprops=wedgeprops)

        # Add the percentage text in the center
        ax.text(0, 0, f"{score}%", ha='center', va='center',
                fontsize=30, fontweight='bold', fontproperties=font_prop, color="#1d2f4b")

        # Ensure it's a circle
        ax.axis('equal')

        # Hide the axes
        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Title outside the plot area
        ax.set_title('Lexical Quality Score', fontproperties=font_prop, fontsize=18,
                     fontweight='bold', color="#1d2f4b", pad=20)

        fig.canvas.draw()
        img = io.BytesIO()
        fig.savefig(img, format='png', dpi=200, bbox_inches='tight', transparent=True)
        img.seek(0)
        plt.close(fig)

        return base64.b64encode(img.read()).decode()

    except Exception as e:
        print(f"Error creating circular progress bar: {e}")
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
    global total_interactions, user_responses, lexical_encoder, lexical_model
    print('Conversation stopped by client')
    stop_hume_connection()
    lexical_score = None
    if lexical_model:
        #print(f"User Responses:\n{user_responses}")  
        scores = get_lexical_score(user_responses, lexical_encoder, lexical_model)
        if scores is not None:
            lexical_score = scores
            #print(f"Lexical Score: {lexical_score}")
        else:
            print("Could not generate lexical score")
    else:
        print("Lexical Model did not load")
    emit('redirect', {'url': url_for('evaluation', total_interactions=total_interactions, lexical_score=lexical_score)})

# --- Hume API Integration ---
async def hume_websocket_handler(message):
    global filler_word_count, total_interactions, user_responses
    scores = {}
    filler_words = ["like", "well", "so", "um", "ah", "er", "you know", "i mean"]

    if message.type in ["user_message", "assistant_message"]:
        role = "E.A.S.E" if message.message.role.upper() == "ASSISTANT" else "You"
        message_text = message.message.content
        
        # Collect User Responses
        if role == "You":
            user_responses.append(message_text)

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
            total_interactions += 1

    elif message.type == "audio_output":
        message_str: str = message.data
        message_bytes = base64.b64decode(message_str.encode("utf-8"))
        await byte_strs.put(message_bytes)
        return

    elif message.type == "error":
        error_message: str = message.message
        print(f"Error from Hume API: {error}")
        socketio.emit('hume_error', {'message': f"Hume API Error: {error}"})
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
        
        # Wait for the loop to actually stop
        try:
            main_loop.run_until_complete(asyncio.sleep(0.1))
        except RuntimeError:
            pass  # The loop was already closed

    if hume_socket:
        hume_socket = None

    main_loop = None


def reset_global_variables():
    global hume_socket, main_loop, microphone_task, byte_strs, hume_connected, filler_word_count, all_emotion_scores_overalls, positive_emotion_scores, negative_emotion_scores, total_interactions, user_responses, lexical_model, lexical_encoder
    hume_socket = None
    hume_connected = False
    if main_loop:
        try:
            for task in asyncio.all_tasks(main_loop):
                task.cancel()
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
    user_responses = []
    total_interactions = 0
    try:
        lexical_model = load_model('models/lexical_model_scaled_outputs.keras')
        lexical_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Model load error:\n{e}")

def run_hume_client():
    global main_loop, hume_socket
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_loop = loop
    try:
        loop.run_until_complete(hume_client_task())
    except RuntimeError as e:
        if str(e) == 'Event loop stopped before Future completed.':
            print("Hume client task stopped prematurely. This is expected during shutdown or when starting a new conversation.")
        else:
            print(f"Unexpected error in run_hume_client: {e}")
    finally:
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
            try:
                await asyncio.Future()  # Keep the connection open indefinitely
            except asyncio.CancelledError:
                print("Hume client task cancelled.")
            except Exception as e:
                print(f"Exception in hume_client_task: {e}")
    except Exception as e:
        hume_connected = False
        print(f"Error connecting to Hume: {e}")
        socketio.emit('hume_error', {'message': f"Hume Connection Error: {e}"})

# --- Flask Routes ---
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/evaluation/<int:total_interactions>')
def evaluation(total_interactions):
    positive_emotion_chart = create_line_chart(
        positive_emotion_scores,
        "Shifts in Progress of Desired Tones",
        "Scores",
        {"sympathy": "#2ecc71", "calmness": "#3498db", "interest": "#9b59b6"}
    )
    negative_emotion_chart = create_line_chart(
        negative_emotion_scores,
        "Shifts in Progress of Undesired Tones",
        "Scores",
        {"doubt": "#e74c3c", "anxiety": "#f39c12", "distress": "#95a5a6"}
    )
    top_emotions_chart = create_top_emotions_bar_chart(all_emotion_scores_overalls, total_interactions)
    
    # Calculate lexical score
    lexical_score = None
    if lexical_model and user_responses:
        lexical_score = get_lexical_score(user_responses, lexical_encoder, lexical_model)
    
    # Create circular progress bar chart
    lexical_score_chart = create_circular_progress_bar(lexical_score) if lexical_score is not None else None

    return render_template(
        'evaluation.html',
        positive_emotion_chart=positive_emotion_chart,
        negative_emotion_chart=negative_emotion_chart,
        top_emotions_chart=top_emotions_chart,
        lexical_score_chart=lexical_score_chart
    )

@app.route('/new_conversation')
def new_conversation():
    reset_global_variables() # reset everything
    session.clear()  # Clears flask session variables
    stop_hume_connection()  # Stop the current Hume connection
    start_hume_connection()  # Start a new Hume connection
    return redirect(url_for('chat'))  # Redirect to the chat page

@app.route('/chat')
def chat():
    return render_template('index.html')  # Conversation page

# --- Main ---
if __name__ == '__main__':
    socketio.run(app, debug=True)
