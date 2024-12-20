from flask import Flask, request, send_file, jsonify
import subprocess
import logging
import requests


import os
from werkzeug.utils import secure_filename


import chess
import chess.pgn
import random

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.info("Starting Flask TTS API...")

# ---- Begin Chess Code ----

# Initialize a game board
board = chess.Board()
conversation_context = []

import os

# Function to load the base prompt from a file
def load_base_prompt():
    base_prompt_path = "/var/www/chess.saphraxinos.com/base_prompt.txt"
    try:
        with open(base_prompt_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Base prompt file not found. Please ensure base_prompt.txt exists."

@app.route('/start_game', methods=['POST'])
def start_game():
    """Start a new game of chess between user and GPT."""
    global board, conversation_context
    
    # Reset the board and conversation context
    board.reset()
    conversation_context = []

    # Randomly assign colors
    user_color = random.choice(['white', 'black'])
    gpt_color = 'white' if user_color == 'black' else 'black'

    # Load the base prompt
    base_prompt = load_base_prompt()
    if "not found" in base_prompt.lower():
        return jsonify({"error": "Base prompt file missing"}), 500

    # Initialize GPT conversation context
    initial_prompt = (
        f"{base_prompt}\n\n"
        f"You are playing chess as {gpt_color}. "
        f"The user is {user_color}. If you are white, make the first move."
    )
    conversation_context.append({"role": "system", "content": initial_prompt})

     # If GPT is white, make the first move
    if gpt_color == 'white':
        legal_moves = [board.san(move) for move in board.legal_moves]
        conversation_context.append({"role": "user", "content": f"Your legal moves are: {', '.join(legal_moves)}"})#TODO why join?
        
        # Query GPT for its first move
        gpt_move_response = make_gpt_move(conversation_context, api_key=os.getenv("GPT_API_KEY"))#TODO different than current
        if "error" in gpt_move_response:
            return jsonify(gpt_move_response), 500

        gpt_message = gpt_move_response["gpt_message"]
        return jsonify({"message": "Game started", "user_color": user_color, "gpt_color": gpt_color, "gpt_message": gpt_message})

    return jsonify({"message": "Game started", "user_color": user_color, "gpt_color": gpt_color})

    # # Initial prompt for GPT
    # initial_prompt = (
    #     f"{base_prompt}\n\n"
    #     f"You are playing chess as {gpt_color}. "
    #     f"{user_color.capitalize()} moves first. "
    #     f"Remember, provide responses in the specified format."
    # )

    # # If GPT is white, make its first move
    # if gpt_color == 'white':
    #     legal_moves = [move.uci() for move in board.legal_moves]
    #     gpt_prompt = (
    #         f"{initial_prompt}\n\n"
    #         f"The board is empty. Your legal moves are: {', '.join(legal_moves)}. "
    #         f"Make your first move."
    #     )

    #     headers = {"Authorization": f"Bearer {os.environ.get('GPT_API_KEY')}"}
    #     gpt_response = requests.post(
    #         "https://api.openai.com/v1/chat/completions",
    #         headers=headers,
    #         json={
    #             "model": "gpt-3.5-turbo",
    #             "messages": [
    #                 {"role": "system", "content": "You are a chess-playing assistant."},
    #                 {"role": "user", "content": gpt_prompt},
    #             ],
    #         },
    #     )

    #     if gpt_response.status_code == 200:
    #         gpt_data = gpt_response.json()
    #         gpt_message = gpt_data['choices'][0]['message']['content']

    #         try:
    #             gpt_move = gpt_message.split('---chess-program---')[1].strip()
    #             chess_move = chess.Move.from_uci(gpt_move)
    #             if chess_move in board.legal_moves:
    #                 board.push(chess_move)
    #                 gpt_user_message = gpt_message.split('---tell-the-user---')[1].strip()
    #                 return jsonify({"message": "Game started", "user_color": user_color, "gpt_color": gpt_color, "gpt_move": gpt_move, "gpt_message": gpt_user_message})
    #         except Exception as e:
    #             return jsonify({"error": f"Failed to parse GPT move: {e}"}), 500
    #     else:
    #         return jsonify({"error": "Failed to get GPT response"}), 500

    # return jsonify({"message": "Game started", "user_color": user_color, "gpt_color": gpt_color})

@app.route('/make_move', methods=['POST'])
def make_move():
    """Process a move from the user or GPT."""
    # global board
    global board, conversation_context

    data = request.json
    user_input = data.get('move')
    api_key = data.get('api_key')

    if not user_input or not api_key:
        return jsonify({"error": "Move and API key are required"}), 400

    # Get legal moves for GPT
    legal_moves = [board.san(move) for move in board.legal_moves]
    conversation_context.append({"role": "user", "content": f"Legal moves: {', '.join(legal_moves)}"})

    # Query GPT for its move
    gpt_move_response = make_gpt_move(conversation_context, api_key)
    if "error" in gpt_move_response:
        return jsonify(gpt_move_response), 500

    gpt_message = gpt_move_response["gpt_message"]

    # Parse GPT's move
    try:
        gpt_move = gpt_message.split('---chess_program---')[1].strip()
        board.push_san(gpt_move)
    except (ValueError, IndexError):
        return jsonify({"error": "Failed to parse GPT's move."}), 400

    # Extract user message for transcription and playback
    try:
        user_message = gpt_message.split('---tell_the_user---')[1].strip()
    except IndexError:
        user_message = "GPT did not provide a valid user message."

    return jsonify({"board": board.fen(), "gpt_message": user_message})


    # # Add user's input to conversation context
    # conversation_context.append({"role": "user", "content": user_input})


    # # try:
    # #     # Validate and apply user's move
    # #     chess_move = chess.Move.from_uci(move)
    # #     if chess_move not in board.legal_moves:
    # #         return jsonify({"error": "Invalid move"}), 400

    # #     board.push(chess_move)
    # # except Exception as e:
    # #     return jsonify({"error": f"Invalid move format: {e}"}), 400

    # # # Check if game is over
    # # if board.is_game_over():
    # #     return jsonify({"result": board.result(), "message": "Game over!"})

    # # # Prepare prompt for GPT
    # # legal_moves = [move.uci() for move in board.legal_moves]
    # # moves_so_far = ", ".join(board.move_stack)
    # # gpt_prompt = (
    # #     f"{load_base_prompt()}\n\n"
    # #     f"Here is the game so far: {moves_so_far}. "
    # #     f"Opponent played {move}. Your legal moves are: {', '.join(legal_moves)}. "
    # #     "Choose your move or respond appropriately as instructed."
    # # )

    # # Query GPT with the conversation context
    # headers = {"Authorization": f"Bearer {api_key}"}
    # gpt_response = requests.post(
    #     "https://api.openai.com/v1/chat/completions",
    #     headers=headers,
    #     json={
    #         "model": "gpt-3.5-turbo",
    #         "messages": conversation_context,
    #         # "messages": [
    #         #     {"role": "system", "content": "You are a chess-playing assistant."},
    #         #     {"role": "user", "content": gpt_prompt},
    #         # ],
    #     },
    # )

    # if gpt_response.status_code != 200:
    #     return jsonify({"error": "Failed to get GPT response"}), 500

    # gpt_data = gpt_response.json()
    # gpt_message = gpt_data['choices'][0]['message']['content']

    # # Add GPT's reply to conversation context
    # conversation_context.append({"role": "assistant", "content": gpt_message})

    # return jsonify({"gpt_message": gpt_message})

    # # Handle GPT response
    # gpt_move = None
    # user_message = None

    # try:
    #     if '---chess-program---' in gpt_message:
    #         gpt_move = gpt_message.split('---chess-program---')[1].strip()

    #         chess_move = chess.Move.from_uci(gpt_move)
    #         if chess_move not in board.legal_moves:
    #             return jsonify({"error": "GPT made an invalid move"}), 400

    #         board.push(chess_move)

    #     if '---tell-the-user---' in gpt_message:
    #         user_message = gpt_message.split('---tell-the-user---')[1].strip()
    #     else:
    #         user_message = "GPT did not provide a user message."
    # except Exception as e:
    #     return jsonify({"error": f"GPT response parsing failed: {e}"}), 400

    # return jsonify({
    #     "board": board.fen(),
    #     "gpt_move": gpt_move,
    #     "user_message": user_message,
    # })
# ---- End Chess Code ----

def make_gpt_move(conversation_context, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        gpt_response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={"model": "gpt-3.5-turbo", "messages": conversation_context},
        )
        if gpt_response.status_code != 200:
            logging.error(f"GPT API error: {gpt_response.text}")
            return {"error": "Failed to get GPT response"}
        gpt_data = gpt_response.json()
        return {"gpt_message": gpt_data['choices'][0]['message']['content']}
    except Exception as e:
        logging.error(f"Error in make_gpt_move: {e}")
        return {"error": "An unexpected error occurred"}



# Error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"An error occurred: {str(e)}")
    return jsonify(error=str(e)), 500


@app.route('/tts', methods=['GET', 'POST'])
def tts():
    if request.method == 'GET':
        return "Use POST to send text for TTS.", 200

    if request.method == 'POST':
        # Extract the text first
        logging.info(f"POST data received: {request.form}")
        text = request.form.get('text')  
        if not text:
            logging.error("No text provided in POST request.")
            return "Error: No text provided", 400

        # Prepare output file
        output_file = "output.wav"

        try:
            # Run espeak-ng subprocess
            logging.info(f"Running TTS subprocess with text: {text}")
            result = subprocess.run(['espeak-ng', text, '-w', output_file], check=True, capture_output=True, text=True)
            logging.info(f"Subprocess output: {result.stdout}")
            logging.info(f"Subprocess errors: {result.stderr}")
            logging.info(f"TTS file generated: {output_file}")
            return send_file(output_file, as_attachment=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess failed: {e.stderr}")
            return "Error during TTS processing.", 500
        except Exception as e:
            logging.error(f"Unexpected error during TTS processing: {str(e)}")
            return "Error during TTS processing.", 500

@app.route('/gpt', methods=['POST'])
def gpt():
    data = request.json
    api_key = data.get('api_key')
    user_transcription = data.get('prompt')

    if not api_key or not user_transcription:
        return jsonify({"error": "API key and transcription are required"}), 400

    # Define the base prompt
    base_prompt = "You are now going to give a very short reply to the following some what vague prompt:"

    # Combine the base prompt with the user's transcription
    full_prompt = f"{base_prompt}\n\nUser: {user_transcription}"

    headers = {"Authorization": f"Bearer {api_key}"}
    json_data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": full_prompt}]
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data
        )
        response.raise_for_status()
        result = response.json()
        return jsonify({"response": result['choices'][0]['message']['content']})
    except requests.exceptions.RequestException as e:
        logging.error(f"GPT API error: {e}")
        return jsonify({"error": "Could not retrieve response from GPT"}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return "No audio file uploaded", 400
    
    audio = request.files['audio']

    # Define consistent file paths
    audio_path = "/var/www/chess.saphraxinos.com/uploaded.wav"
    converted_audio_path = "/var/www/chess.saphraxinos.com/uploaded_converted.wav"

    # Save the uploaded file, overwriting the old one
    audio.save(audio_path)

    # Use full path to deepspeech executable in myenv39
    deepspeech_path = "/var/www/chess.saphraxinos.com/myenv39/bin/deepspeech"


    # # Save uploaded file with a unique name
    # audio_filename = secure_filename(audio.filename)
    # audio_path = os.path.join("/var/www/chess.saphraxinos.com", audio_filename)
    # converted_audio_path = os.path.splitext(audio_path)[0] + "_converted.wav"
    # audio.save(audio_path)  # Save the uploaded file

    # Convert audio to 16kHz, mono, PCM s16le
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-filter:a", "volume=2.0", converted_audio_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        return f"Audio conversion failed: {e}", 500

    try:
        result = subprocess.run(
            [deepspeech_path, "--model", "/root/deepspeech-models/deepspeech-0.9.3-models.pbmm",
             "--scorer", "/root/deepspeech-models/deepspeech-0.9.3-models.scorer",
             "--audio", converted_audio_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return jsonify(error=result.stderr), 500

        # Return the transcription
        return jsonify(transcription=result.stdout.strip())
    except Exception as e:
        return jsonify(error=str(e)), 500
    
    try:
        os.remove(audio_path)
        os.remove(converted_audio_path)
    except Exception as e:
        logging.warning(f"Failed to clean up files: {e}")

@app.route('/save_audio', methods=['POST'])
def save_audio():
    """Save uploaded audio and convert it for Whisper compatibility."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    try:
        # Save the uploaded file
        audio = request.files['audio']
        audio_path = "/var/www/chess.saphraxinos.com/whisper.wav"
        audio.save(audio_path)

        # Convert the audio for Whisper compatibility
        converted_audio_path = "/var/www/chess.saphraxinos.com/converted_whisper.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", converted_audio_path],
            check=True,
            stderr=subprocess.PIPE
        )

        return jsonify({"message": "Audio file converted successfully"}), 200
    except subprocess.CalledProcessError as e:
        logging.error(f"Audio conversion failed: {e}")
        return jsonify({"error": "Audio conversion failed"}), 500
    except Exception as e:
        logging.error(f"Error saving or converting audio: {e}")
        return jsonify({"error": "Failed to save audio"}), 500


@app.route('/transcribe_whisper', methods=['POST'])
def transcribe_whisper():
    """Send the converted audio to the Whisper API for transcription."""
    try:
        logging.debug("Received /transcribe_whisper request.")

        # Extract the user's API key from the request
        api_key = request.headers.get("Authorization")

        if not api_key or not api_key.startswith("Bearer "):
            return jsonify({"error": "Invalid or missing API key"}), 400
        api_key = api_key.replace("Bearer ", "").strip()

        # Check if the converted audio file exists
        converted_audio_path = "/var/www/chess.saphraxinos.com/converted_whisper.wav"
        if not os.path.exists(converted_audio_path):
            return jsonify({"error": "Converted audio file not found"}), 400
        
        logging.debug("Sending file to Whisper API...")

        # Send the converted file to the Whisper API
        with open(converted_audio_path, "rb") as audio_file:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": audio_file},
                data={"model": "whisper-1"}
            )

        # Handle Whisper API response
        if response.status_code == 200:
            logging.debug("Whisper API returned success.")
            return jsonify(response.json())
        else:
            logging.error(f"Whisper API error: {response.text}")
            return jsonify({"error": "Failed to transcribe audio"}), response.status_code
    except requests.exceptions.RequestException as req_err:
        logging.error(f"RequestException in Whisper transcription: {req_err}")
        return jsonify({"error": "An error occurred while communicating with the Whisper API"}), 500  
    except Exception as e:
        logging.error(f"Unexpected error in Whisper transcription: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500




if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000, debug=False)