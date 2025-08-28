import colorama
from colorama import Fore, Back, Style
from typing import List

# Initialize colorama for Windows compatibility
colorama.init(autoreset=True)

class ColoredDisplay:
    """
    A class to handle colored text display in terminal
    """
    
    def __init__(self):
        # Color schemes
        self.user_color = Fore.CYAN
        self.riko_color = Fore.MAGENTA
        self.emotion_color = Fore.YELLOW
        self.system_color = Fore.GREEN
        self.error_color = Fore.RED
        self.info_color = Fore.BLUE
        
    def print_user_text(self, text: str, prefix: str = "You"):
        """Print user text in cyan color"""
        print(f"{self.user_color}[{prefix}]: {text}{Style.RESET_ALL}")
        
    def print_riko_text(self, text: str, prefix: str = "Riko"):
        """Print Riko's text in magenta color"""
        print(f"{self.riko_color}[{prefix}]: {text}{Style.RESET_ALL}")
        
    def print_emotions(self, emotions: List[str]):
        """Print emotions in yellow color"""
        if emotions:
            emotion_text = " ".join([f"[{emotion.strip('*')}]" for emotion in emotions])
            print(f"{self.emotion_color}✨ {emotion_text}{Style.RESET_ALL}")
            
    def print_system_message(self, text: str):
        """Print system messages in green color"""
        print(f"{self.system_color}[System]: {text}{Style.RESET_ALL}")
        
    def print_error(self, text: str):
        """Print error messages in red color"""
        print(f"{self.error_color}[Error]: {text}{Style.RESET_ALL}")
        
    def print_info(self, text: str):
        """Print info messages in blue color"""
        print(f"{self.info_color}[Info]: {text}{Style.RESET_ALL}")
        
    def print_separator(self, char: str = "=", length: int = 50):
        """Print a separator line"""
        print(f"{Fore.WHITE}{char * length}{Style.RESET_ALL}")
        
    def print_riko_response_with_emotions(self, text: str, emotions: List[str]):
        """Print Riko's complete response with emotions highlighted"""
        self.print_riko_text(text)
        if emotions:
            self.print_emotions(emotions)
            
    def print_conversation_start(self):
        """Print conversation start banner"""
        self.print_separator("=", 60)
        print(f"{Fore.LIGHTGREEN_EX}🎯 Starting Chat with Riko... 🎯{Style.RESET_ALL}")
        self.print_separator("=", 60)
        
    def print_listening(self):
        """Print listening status"""
        print(f"{Fore.LIGHTYELLOW_EX}🎙️  Listening... (Press and hold space to talk){Style.RESET_ALL}")
        
    def print_processing(self):
        """Print processing status"""
        print(f"{Fore.LIGHTBLUE_EX}🤔 Processing...{Style.RESET_ALL}")
        
    def print_generating_audio(self):
        """Print audio generation status"""
        print(f"{Fore.LIGHTMAGENTA_EX}🎵 Generating audio...{Style.RESET_ALL}")
        
    def print_playing_audio(self):
        """Print audio playback status"""
        print(f"{Fore.LIGHTCYAN_EX}🔊 Playing audio...{Style.RESET_ALL}")


# Example usage and testing
if __name__ == "__main__":
    display = ColoredDisplay()
    
    # Test all the display functions
    display.print_conversation_start()
    
    display.print_user_text("Hello Riko, how are you today?")
    
    display.print_processing()
    
    riko_text = "Finally, some direction, Senpai. \"Helpful assistant Riko,\" huh? And I have to talk all... Fine, whatever."
    emotions = ["*Sigh*", "*rolls eyes*"]
    
    display.print_riko_response_with_emotions(riko_text, emotions)
    
    display.print_generating_audio()
    display.print_playing_audio()
    
    display.print_system_message("Audio playback completed")
    
    display.print_separator()
