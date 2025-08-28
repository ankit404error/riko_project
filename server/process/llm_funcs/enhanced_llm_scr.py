#!/usr/bin/env python3
"""
Enhanced LLM Integration with Contextual Memory and Optimized TTS
================================================================

This module provides an enhanced version of the LLM response system that integrates:
- Contextual memory management
- User preference learning
- Optimized TTS generation
- Emotional continuity
- Conversation summarization

Features:
- Maintains conversation context across sessions
- Learns user preferences and adapts responses
- Provides fast, cached TTS generation
- Tracks emotional states and topics
- Intelligent memory management
"""

import os
import sys
import time
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import google.generativeai as genai

# Add project paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))
sys.path.append(str(current_dir.parent.parent / "memory"))
sys.path.append(str(current_dir.parent.parent / "audio"))

try:
    from memory.context_manager import get_context_manager
    from audio.tts_optimizer import get_tts_optimizer
    from text_processing.emotion_filter import EmotionFilter
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLLMProcessor:
    """
    Enhanced LLM processor with memory, optimization, and emotion tracking
    """
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        
        # Initialize Google Gemini
        genai.configure(api_key=self.config['GEMINI_API_KEY'])
        self.model = genai.GenerativeModel(self.config['model'])
        
        # Enhanced features (if available)
        if ENHANCED_FEATURES_AVAILABLE:
            self.context_manager = get_context_manager()
            self.tts_optimizer = get_tts_optimizer()
            self.emotion_filter = EmotionFilter()
        else:
            self.context_manager = None
            self.tts_optimizer = None
            self.emotion_filter = None
            logger.warning("Enhanced features disabled - using basic mode")
        
        # Conversation state
        self.current_conversation_id = None
        self.conversation_started = time.time()
        
        # Load/save basic history for fallback
        self.history_file = self.config.get('history_file', 'chat_history.json')
        
        # Statistics
        self.stats = {
            'total_responses': 0,
            'memory_enhanced_responses': 0,
            'cache_hits': 0,
            'average_response_time': 0.0,
            'total_response_time': 0.0
        }
        
        logger.info("Enhanced LLM Processor initialized")
    
    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from multiple possible locations"""
        
        # Try multiple config paths
        potential_paths = [
            config_path,
            'character_config.yaml',
            '../character_config.yaml', 
            '../../character_config.yaml',
            '../../../character_config.yaml',
            '../../../../character_config.yaml'
        ]
        
        for path in potential_paths:
            if path and Path(path).exists():
                try:
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                        logger.info(f"Loaded config from: {path}")
                        return config
                except Exception as e:
                    logger.error(f"Error loading config from {path}: {e}")
        
        raise FileNotFoundError("Could not find character_config.yaml")
    
    def extract_user_emotions(self, user_input: str) -> List[str]:
        """Extract emotional indicators from user input"""
        emotion_patterns = {
            'happy': ['happy', 'glad', 'joy', 'excited', 'great', 'awesome', 'wonderful', 'amazing'],
            'sad': ['sad', 'down', 'depressed', 'unhappy', 'miserable', 'crying', 'upset'],
            'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'],
            'confused': ['confused', "don't understand", 'unclear', 'what', 'huh', '?'],
            'curious': ['how', 'why', 'what', 'when', 'where', 'tell me', 'explain'],
            'tired': ['tired', 'exhausted', 'sleepy', 'weary', 'drained'],
            'worried': ['worried', 'concerned', 'anxious', 'nervous', 'scared', 'afraid'],
            'grateful': ['thanks', 'thank you', 'grateful', 'appreciate']
        }
        
        user_input_lower = user_input.lower()
        detected_emotions = []
        
        for emotion, keywords in emotion_patterns.items():
            if any(keyword in user_input_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions or ['neutral']
    
    def get_enhanced_response(self, user_input: str, voice_mode: bool = False) -> Tuple[str, List[str]]:
        """Get enhanced response with memory and optimization"""
        start_time = time.time()
        
        try:
            # Extract user emotions
            user_emotions = self.extract_user_emotions(user_input)
            
            # Get memory context if available
            if self.context_manager:
                memory_context = self.context_manager.get_memory_context_for_llm(user_input)
                self.stats['memory_enhanced_responses'] += 1
            else:
                memory_context = ""
            
            # Build enhanced prompt
            enhanced_prompt = self.build_enhanced_prompt(
                user_input, 
                user_emotions, 
                memory_context, 
                voice_mode
            )
            
            # Get response from Gemini
            riko_response = self.get_gemini_response(enhanced_prompt)
            
            # Process response for emotions
            if self.emotion_filter:
                clean_response, riko_emotions = self.emotion_filter.process_text(riko_response)
            else:
                clean_response = riko_response
                riko_emotions = self.extract_basic_emotions(riko_response)
            
            # Add to memory if available
            if self.context_manager:
                self.context_manager.add_conversation_turn(
                    user_input=user_input,
                    user_emotions=user_emotions,
                    riko_response=clean_response,
                    riko_emotions=riko_emotions
                )
            else:
                # Fallback to basic history
                self.add_to_basic_history(user_input, clean_response)
            
            # Update statistics
            response_time = time.time() - start_time
            self.stats['total_responses'] += 1
            self.stats['total_response_time'] += response_time
            self.stats['average_response_time'] = (
                self.stats['total_response_time'] / self.stats['total_responses']
            )
            
            logger.info(f"Enhanced response generated in {response_time:.2f}s")
            
            return clean_response, riko_emotions
            
        except Exception as e:
            logger.error(f"Error in enhanced response: {e}")
            # Fallback to basic response
            return self.get_basic_response(user_input), ['neutral']
    
    def build_enhanced_prompt(self, user_input: str, user_emotions: List[str], 
                            memory_context: str, voice_mode: bool) -> str:
        """Build enhanced prompt with memory context and emotional awareness"""
        
        # Base system prompt
        system_prompt = self.config['presets']['default']['system_prompt']
        
        # Add memory context
        if memory_context:
            system_prompt += f"\n\n{memory_context}\n"
            system_prompt += "\n[INSTRUCTIONS]: Use the above context to maintain conversation continuity. Reference previous topics and emotional states when appropriate. Adapt your responses based on learned user preferences."
        
        # Add emotional awareness
        if user_emotions and user_emotions != ['neutral']:
            emotion_instruction = f"\nUser's current emotional state: {', '.join(user_emotions)}. Respond with appropriate emotional tone and empathy."
            system_prompt += emotion_instruction
        
        # Voice mode optimization
        if voice_mode:
            system_prompt += "\n[VOICE MODE]: Keep responses concise (1-2 sentences) for better speech delivery."
        
        # Add user input
        full_prompt = f"{system_prompt}\n\nUser: {user_input}"
        
        return full_prompt
    
    def get_gemini_response(self, prompt: str) -> str:
        """Get response from Google Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.9,
                    top_p=1.0,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Sorry senpai, I'm having trouble thinking right now. Try again?"
    
    def extract_basic_emotions(self, text: str) -> List[str]:
        """Extract basic emotions from response text (fallback method)"""
        emotion_indicators = {
            'tsundere': ['hmph', 'whatever', 'not like', "don't think", 'baka'],
            'annoyed': ['ugh', 'seriously', 'jeez', 'sheesh'],
            'helpful': ['help', 'assist', 'explain', 'show you'],
            'sarcastic': ['obviously', 'of course', 'really?'],
            'caring': ['make sure', 'be careful', 'take care'],
            'embarrassed': ['blush', '*blushes*', 'n-not', 's-shut up']
        }
        
        text_lower = text.lower()
        detected_emotions = []
        
        for emotion, indicators in emotion_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                detected_emotions.append(emotion)
        
        return detected_emotions or ['neutral']
    
    def get_basic_response(self, user_input: str) -> str:
        """Basic response without enhanced features (fallback)"""
        try:
            # Load basic history
            history = self.load_basic_history()
            
            # Create chat with history
            chat = self.model.start_chat(history=history)
            
            system_prompt = self.config['presets']['default']['system_prompt']
            response = chat.send_message(
                f"{system_prompt}\n\nUser: {user_input}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.9,
                    top_p=1.0,
                    max_output_tokens=2048,
                )
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Basic response error: {e}")
            return "Sorry senpai, I'm having trouble thinking right now. Try again?"
    
    def load_basic_history(self) -> List[Dict]:
        """Load basic conversation history"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def add_to_basic_history(self, user_input: str, riko_response: str):
        """Add to basic conversation history"""
        history = self.load_basic_history()
        
        history.append({"role": "user", "parts": [user_input]})
        history.append({"role": "model", "parts": [riko_response]})
        
        # Keep only recent history (last 50 turns)
        if len(history) > 100:
            history = history[-100:]
        
        try:
            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving basic history: {e}")
    
    def generate_optimized_audio(self, text: str, output_path: str) -> Optional[str]:
        """Generate audio with optimization if available"""
        if self.tts_optimizer:
            try:
                result = self.tts_optimizer.generate_audio_sync(text, output_path)
                if result:
                    self.stats['cache_hits'] += 1
                return result
            except Exception as e:
                logger.error(f"Optimized TTS error: {e}")
        
        # Fallback to original TTS method
        try:
            # Import original method
            from tts_func.sovits_ping import sovits_gen
            return sovits_gen(text, output_path)
        except Exception as e:
            logger.error(f"Fallback TTS error: {e}")
            return None
    
    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive conversation statistics"""
        stats = self.stats.copy()
        
        # Add context manager stats if available
        if self.context_manager:
            try:
                memory_context = self.context_manager.get_conversation_context(max_turns=1)
                stats['memory_system_active'] = True
                stats['recent_conversations'] = len(self.context_manager.recent_conversations)
                stats['user_preferences'] = len(self.context_manager.user_preferences)
                stats['active_topics'] = len(self.context_manager.active_topics)
            except:
                stats['memory_system_active'] = False
        else:
            stats['memory_system_active'] = False
        
        # Add TTS optimizer stats if available
        if self.tts_optimizer:
            try:
                tts_stats = self.tts_optimizer.get_optimization_stats()
                stats['tts_optimization_active'] = True
                stats['tts_cache_hit_rate'] = tts_stats['performance']['cache_hit_rate_percent']
                stats['tts_average_generation_time'] = tts_stats['performance']['average_generation_time']
            except:
                stats['tts_optimization_active'] = False
        else:
            stats['tts_optimization_active'] = False
        
        return stats
    
    def cleanup_old_data(self, days: int = 7):
        """Clean up old data from enhanced systems"""
        if self.context_manager:
            try:
                self.context_manager.cleanup_old_data(days)
                logger.info(f"Cleaned up context data older than {days} days")
            except Exception as e:
                logger.error(f"Error cleaning context data: {e}")
        
        if self.tts_optimizer:
            try:
                self.tts_optimizer.cleanup_old_cache(days)
                logger.info(f"Cleaned up TTS cache older than {days} days")
            except Exception as e:
                logger.error(f"Error cleaning TTS cache: {e}")


# Singleton instance for enhanced LLM processor
_enhanced_llm_instance = None

def get_enhanced_llm_processor(**kwargs) -> EnhancedLLMProcessor:
    """Get singleton instance of EnhancedLLMProcessor"""
    global _enhanced_llm_instance
    
    if _enhanced_llm_instance is None:
        _enhanced_llm_instance = EnhancedLLMProcessor(**kwargs)
    return _enhanced_llm_instance


def enhanced_llm_response(user_input: str, voice_mode: bool = False) -> Tuple[str, List[str]]:
    """
    Enhanced LLM response function with memory and optimization
    
    Args:
        user_input: User's input text
        voice_mode: Whether this is for voice output (shorter responses)
    
    Returns:
        Tuple of (response_text, emotions_list)
    """
    try:
        processor = get_enhanced_llm_processor()
        return processor.get_enhanced_response(user_input, voice_mode)
    except Exception as e:
        logger.error(f"Enhanced LLM response error: {e}")
        # Return basic fallback
        return "Sorry senpai, something went wrong. Try again?", ['neutral']


def enhanced_audio_generation(text: str, output_path: str) -> Optional[str]:
    """
    Enhanced audio generation with optimization and caching
    
    Args:
        text: Text to convert to speech
        output_path: Path where audio file should be saved
    
    Returns:
        Path to generated audio file or None if failed
    """
    try:
        processor = get_enhanced_llm_processor()
        return processor.generate_optimized_audio(text, output_path)
    except Exception as e:
        logger.error(f"Enhanced audio generation error: {e}")
        return None


if __name__ == "__main__":
    # Test the enhanced system
    print("Testing Enhanced LLM System...")
    
    # Test conversation with memory
    test_conversations = [
        "Hi Riko, how are you today?",
        "Can you help me with math?", 
        "What's 7 minus 8?",
        "Thanks, that was helpful!",
        "Do you remember what we talked about before?"
    ]
    
    processor = get_enhanced_llm_processor()
    
    print("\n=== Enhanced Conversation Test ===")
    for i, user_input in enumerate(test_conversations):
        print(f"\n👤 You: {user_input}")
        
        response, emotions = enhanced_llm_response(user_input, voice_mode=True)
        
        print(f"🤖 Riko: {response}")
        if emotions:
            print(f"   Emotions: {', '.join(emotions)}")
    
    # Test audio generation
    print(f"\n=== Audio Generation Test ===")
    test_audio_text = "Hello Senpai, this is a test of the enhanced audio system!"
    audio_result = enhanced_audio_generation(test_audio_text, "test_enhanced_audio.wav")
    
    if audio_result:
        print(f"✅ Audio generated: {audio_result}")
    else:
        print("❌ Audio generation failed")
    
    # Show statistics
    print(f"\n=== System Statistics ===")
    stats = processor.get_conversation_statistics()
    
    print(f"Total responses: {stats['total_responses']}")
    print(f"Memory system active: {stats['memory_system_active']}")
    print(f"TTS optimization active: {stats['tts_optimization_active']}")
    print(f"Average response time: {stats['average_response_time']:.2f}s")
    
    if stats['memory_system_active']:
        print(f"Recent conversations: {stats.get('recent_conversations', 0)}")
        print(f"User preferences learned: {stats.get('user_preferences', 0)}")
        print(f"Active topics: {stats.get('active_topics', 0)}")
    
    if stats['tts_optimization_active']:
        print(f"TTS cache hit rate: {stats.get('tts_cache_hit_rate', 0):.1f}%")
        print(f"TTS avg generation time: {stats.get('tts_average_generation_time', 0):.2f}s")
    
    # Cleanup test file
    try:
        Path("test_enhanced_audio.wav").unlink(missing_ok=True)
    except:
        pass
    
    print("\n✨ Enhanced LLM system test completed!")
