# Google Gemini Flash-2.0 with history 
### Uses Google Generative AI
import yaml
import json
import os
import google.generativeai as genai

# Find the correct path to character_config.yaml
config_path = None
for potential_path in [
    'character_config.yaml', 
    '../character_config.yaml',
    '../../character_config.yaml', 
    '../../../character_config.yaml',
    '../../../../character_config.yaml'
]:
    try:
        with open(potential_path, 'r') as f:
            char_config = yaml.safe_load(f)
            config_path = potential_path
            break
    except FileNotFoundError:
        continue

if config_path is None:
    raise FileNotFoundError("Could not find character_config.yaml")

# Configure Google Gemini API
genai.configure(api_key=char_config['GEMINI_API_KEY'])
model = genai.GenerativeModel(char_config['model'])

# Constants
HISTORY_FILE = char_config['history_file']
SYSTEM_PROMPT = char_config['presets']['default']['system_prompt']

# Load/save chat history for Gemini (simplified format)
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def get_riko_response_gemini(chat_history, user_input):
    """
    Get response from Google Gemini Flash-2.0
    """
    try:
        # Create a chat session with history
        chat = model.start_chat(history=chat_history)
        
        # Send message and get response
        response = chat.send_message(
            f"{SYSTEM_PROMPT}\n\nUser: {user_input}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                top_p=1.0,
                max_output_tokens=2048,
            )
        )
        
        return response.text
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return "Sorry senpai, I'm having trouble thinking right now. Try again?"

def llm_response(user_input):
    """
    Main function to get LLM response and manage conversation history
    """
    # Load conversation history
    history = load_history()
    
    # Get response from Gemini
    riko_response = get_riko_response_gemini(history, user_input)
    
    # Add user message to history
    history.append({"role": "user", "parts": [user_input]})
    
    # Add assistant response to history
    history.append({"role": "model", "parts": [riko_response]})
    
    # Save updated history
    save_history(history)
    
    return riko_response


if __name__ == "__main__":
    print('running main')