import time
import random
from typing import List, Dict
from colorama import Fore, Style

class EmotionHandler:
    """
    Handles emotion implementation through visual effects, sounds, or other non-verbal feedback
    """
    
    def __init__(self):
        # Emotion to visual mapping
        self.emotion_visuals = {
            "rolls eyes": "🙄",
            "roll eyes": "🙄", 
            "angry face": "😠",
            "creates angry face": "😠",
            "sigh": "😮‍💨",
            "sighs": "😮‍💨",
            "hmph": "😤",
            "whatever": "🤷",
            "shrug": "🤷",
            "shrugs": "🤷",
            "blush": "😊",
            "blushes": "😊",
            "pout": "😒",
            "pouts": "😒",
            "smile": "😊",
            "smiles": "😊",
            "laugh": "😂",
            "laughs": "😂",
            "giggle": "😄",
            "giggles": "😄",
            "yawn": "🥱",
            "yawns": "🥱",
            "nod": "👍",
            "nods": "👍",
            "shake head": "👎",
            "shakes head": "👎",
            "crosses arms": "🙅",
            "cross arms": "🙅",
            "looks away": "👀",
            "look away": "👀",
            "glare": "😡",
            "glares": "😡",
            "wink": "😉",
            "winks": "😉",
        }
        
        # Emotion descriptions for ASCII art (optional)
        self.emotion_descriptions = {
            "rolls eyes": "눈_눈",
            "angry face": "ಠ_ಠ",
            "sigh": "(-_-)",
            "hmph": "(>_<)",
            "whatever": "¯\\_(ツ)_/¯",
            "shrug": "¯\\_(ツ)_/¯",
            "blush": "(>.<)",
            "smile": "(^_^)",
            "laugh": "(≧∇≦)",
            "wink": "(^_~)",
        }
        
        # Color mappings for different emotion categories
        self.emotion_colors = {
            "negative": Fore.RED,
            "positive": Fore.GREEN, 
            "neutral": Fore.YELLOW,
            "playful": Fore.MAGENTA,
        }
        
        # Emotion categories
        self.emotion_categories = {
            "negative": ["rolls eyes", "angry face", "sigh", "hmph", "pout", "glare"],
            "positive": ["smile", "laugh", "giggle", "blush", "wink"],
            "neutral": ["whatever", "shrug", "yawn", "nod", "shake head"],
            "playful": ["crosses arms", "looks away", "pouts"],
        }
        
    def get_emotion_category(self, emotion: str) -> str:
        """Get the category of an emotion"""
        clean_emotion = emotion.strip('*').lower()
        for category, emotions in self.emotion_categories.items():
            if clean_emotion in emotions:
                return category
        return "neutral"
        
    def get_emotion_visual(self, emotion: str) -> str:
        """Get the visual representation of an emotion"""
        clean_emotion = emotion.strip('*').lower()
        return self.emotion_visuals.get(clean_emotion, "😐")
        
    def get_emotion_ascii(self, emotion: str) -> str:
        """Get ASCII art representation of an emotion"""
        clean_emotion = emotion.strip('*').lower()
        return self.emotion_descriptions.get(clean_emotion, "(・_・)")
        
    def display_emotion_visual(self, emotion: str, style: str = "emoji") -> str:
        """Display emotion with visual effects"""
        clean_emotion = emotion.strip('*').lower()
        category = self.get_emotion_category(emotion)
        color = self.emotion_colors.get(category, Fore.WHITE)
        
        if style == "emoji":
            visual = self.get_emotion_visual(emotion)
        elif style == "ascii":
            visual = self.get_emotion_ascii(emotion)
        else:
            visual = f"[{clean_emotion}]"
            
        return f"{color}{visual}{Style.RESET_ALL}"
        
    def play_emotion_sequence(self, emotions: List[str], delay: float = 0.5):
        """Play a sequence of emotions with timing"""
        print(f"{Fore.CYAN}✨ Emotion sequence:{Style.RESET_ALL}")
        for emotion in emotions:
            visual = self.display_emotion_visual(emotion)
            print(f"   {visual}")
            time.sleep(delay)
            
    def create_emotion_banner(self, emotions: List[str]) -> str:
        """Create a banner showing all emotions"""
        if not emotions:
            return ""
            
        visuals = []
        for emotion in emotions:
            visual = self.display_emotion_visual(emotion, style="emoji")
            visuals.append(visual)
            
        return " ".join(visuals)
        
    def trigger_emotion_effect(self, emotion: str):
        """Trigger special effects for emotions"""
        clean_emotion = emotion.strip('*').lower()
        category = self.get_emotion_category(emotion)
        
        effects = {
            "rolls eyes": self._roll_eyes_effect,
            "angry face": self._angry_effect,
            "sigh": self._sigh_effect,
            "laugh": self._laugh_effect,
            "blush": self._blush_effect,
        }
        
        effect_func = effects.get(clean_emotion)
        if effect_func:
            effect_func()
        else:
            # Default effect
            self._default_emotion_effect(emotion)
            
    def _roll_eyes_effect(self):
        """Special effect for rolling eyes"""
        eyes = ["◔_◔", "◑_◑", "◒_◒", "◓_◓", "●_●"]
        print(f"{Fore.YELLOW}Rolling eyes...{Style.RESET_ALL}")
        for eye in eyes:
            print(f"\r{eye}", end="", flush=True)
            time.sleep(0.3)
        print(f"\r🙄 {Style.RESET_ALL}")
        
    def _angry_effect(self):
        """Special effect for anger"""
        anger_levels = ["😠", "😡", "🤬", "💢"]
        print(f"{Fore.RED}Getting angry...{Style.RESET_ALL}")
        for level in anger_levels:
            print(f"\r{level}", end="", flush=True)
            time.sleep(0.2)
        print(f"\r😡 {Style.RESET_ALL}")
        
    def _sigh_effect(self):
        """Special effect for sighing"""
        print(f"{Fore.BLUE}*Siiiiigh*{Style.RESET_ALL}")
        sighs = ["(-_-)", "(=_=)", "(-.-)", "(>_<)"]
        for s in sighs:
            print(f"\r{s}", end="", flush=True)
            time.sleep(0.4)
        print(f"\r😮‍💨 {Style.RESET_ALL}")
        
    def _laugh_effect(self):
        """Special effect for laughing"""
        laughs = ["😄", "😆", "🤣", "😂"]
        print(f"{Fore.GREEN}Laughing...{Style.RESET_ALL}")
        for laugh in laughs:
            print(f"\r{laugh}", end="", flush=True)
            time.sleep(0.3)
        print(f"\r😂 {Style.RESET_ALL}")
        
    def _blush_effect(self):
        """Special effect for blushing"""
        blushes = ["😊", "😌", "🥰", "☺️"]
        print(f"{Fore.MAGENTA}Blushing...{Style.RESET_ALL}")
        for blush in blushes:
            print(f"\r{blush}", end="", flush=True)
            time.sleep(0.4)
        print(f"\r😊 {Style.RESET_ALL}")
        
    def _default_emotion_effect(self, emotion: str):
        """Default emotion effect"""
        visual = self.display_emotion_visual(emotion)
        clean_emotion = emotion.strip('*').title()
        print(f"{Fore.CYAN}✨ {clean_emotion}: {visual}{Style.RESET_ALL}")


# Example usage
if __name__ == "__main__":
    emotion_handler = EmotionHandler()
    
    # Test emotions
    test_emotions = ["*rolls eyes*", "*sigh*", "*angry face*", "*laugh*", "*blush*"]
    
    print("Testing emotion effects:")
    for emotion in test_emotions:
        print(f"\nTesting: {emotion}")
        emotion_handler.trigger_emotion_effect(emotion)
        time.sleep(1)
        
    print("\n" + "="*50)
    print("Testing emotion banner:")
    banner = emotion_handler.create_emotion_banner(test_emotions)
    print(f"Emotion banner: {banner}")
    
    print("\n" + "="*50)
    print("Testing emotion sequence:")
    emotion_handler.play_emotion_sequence(test_emotions[:3])
