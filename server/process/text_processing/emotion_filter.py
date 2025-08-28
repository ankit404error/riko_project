import re
from typing import Tuple, List

class EmotionFilter:
    """
    A class to filter out emotion descriptions from text before TTS
    and extract emotions for separate processing
    """
    
    def __init__(self):
        # Emotion patterns that should be filtered out from TTS
        self.emotion_patterns = [
            r'\*rolls eyes\*',
            r'\*roll eyes\*', 
            r'\*angry face\*',
            r'\*creates angry face\*',
            r'\*sigh\*',
            r'\*sighs\*',
            r'\*hmph\*',
            r'\*whatever\*',
            r'\*shrug\*',
            r'\*shrugs\*',
            r'\*blush\*',
            r'\*blushes\*',
            r'\*pout\*',
            r'\*pouts\*',
            r'\*smile\*',
            r'\*smiles\*',
            r'\*laugh\*',
            r'\*laughs\*',
            r'\*giggle\*',
            r'\*giggles\*',
            r'\*yawn\*',
            r'\*yawns\*',
            r'\*nod\*',
            r'\*nods\*',
            r'\*shake head\*',
            r'\*shakes head\*',
            r'\*crosses arms\*',
            r'\*cross arms\*',
            r'\*looks away\*',
            r'\*look away\*',
            r'\*glare\*',
            r'\*glares\*',
            r'\*wink\*',
            r'\*winks\*',
            # Generic patterns for other emotions in asterisks
            r'\*[^*]+\*',
        ]
        
        # Compile patterns for better performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.emotion_patterns]
        
    def extract_emotions(self, text: str) -> List[str]:
        """
        Extract all emotion markers from the text
        
        Args:
            text: Input text containing emotions
            
        Returns:
            List of emotion markers found in the text
        """
        emotions = []
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            emotions.extend(matches)
        return emotions
    
    def filter_emotions_for_tts(self, text: str) -> str:
        """
        Remove emotion markers from text to prepare for TTS
        
        Args:
            text: Input text with emotions
            
        Returns:
            Text with emotion markers removed
        """
        filtered_text = text
        for pattern in self.compiled_patterns:
            filtered_text = pattern.sub('', filtered_text)
            
        # Clean up extra spaces and newlines
        filtered_text = re.sub(r'\s+', ' ', filtered_text)
        filtered_text = filtered_text.strip()
        
        return filtered_text
    
    def process_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Process text to separate clean text for TTS and emotions for display
        
        Args:
            text: Input text with emotions
            
        Returns:
            Tuple of (filtered_text_for_tts, list_of_emotions)
        """
        emotions = self.extract_emotions(text)
        filtered_text = self.filter_emotions_for_tts(text)
        
        return filtered_text, emotions
    
    def get_emotion_display(self, emotions: List[str]) -> str:
        """
        Convert emotion markers to display format
        
        Args:
            emotions: List of emotion markers
            
        Returns:
            Formatted string for emotion display
        """
        if not emotions:
            return ""
            
        # Remove asterisks and format for display
        display_emotions = []
        for emotion in emotions:
            clean_emotion = emotion.strip('*').strip()
            display_emotions.append(f"[{clean_emotion}]")
            
        return " ".join(display_emotions)


# Example usage
if __name__ == "__main__":
    emotion_filter = EmotionFilter()
    
    test_text = "*Sigh* Finally, some direction, Senpai. \"Helpful assistant Riko,\" huh? And I have to talk all... *this* way? Fine, whatever. *rolls eyes*"
    
    filtered_text, emotions = emotion_filter.process_text(test_text)
    emotion_display = emotion_filter.get_emotion_display(emotions)
    
    print(f"Original text: {test_text}")
    print(f"Filtered for TTS: {filtered_text}")
    print(f"Emotions found: {emotions}")
    print(f"Emotion display: {emotion_display}")
