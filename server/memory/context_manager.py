#!/usr/bin/env python3
"""
Enhanced Contextual Memory System for Riko
==========================================

This module provides intelligent memory management for maintaining conversation
context, user preferences, emotional states, and long-term memory across sessions.

Features:
- Conversation context tracking
- User preference learning
- Emotional state continuity
- Topic awareness and transitions
- Long-term memory persistence
- Automatic memory summarization
"""

import os
import json
import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import deque, defaultdict
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in conversation"""
    timestamp: str
    user_input: str
    user_emotions: List[str]
    riko_response: str
    riko_emotions: List[str]
    topics: List[str]
    user_satisfaction: Optional[float] = None

@dataclass
class UserPreference:
    """User preference data"""
    preference_type: str  # conversation_style, topics_of_interest, response_length, etc.
    value: Any
    confidence: float  # 0.0 to 1.0
    last_updated: str
    frequency: int = 1

@dataclass
class MemoryEntry:
    """Long-term memory entry"""
    id: str
    content: str
    importance: float  # 0.0 to 1.0
    topics: List[str]
    timestamp: str
    access_count: int = 0
    last_accessed: str = ""

class ContextualMemoryManager:
    """
    Advanced memory management system for Riko
    """
    
    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Initialize databases
        self.db_path = self.memory_dir / "memory.db"
        self.init_database()
        
        # Memory configuration
        self.config = self.load_config()
        
        # In-memory caches
        self.recent_conversations = deque(maxlen=self.config.get('max_recent_turns', 50))
        self.current_session_context = {}
        self.user_preferences = {}
        self.active_topics = defaultdict(float)  # topic -> relevance score
        self.emotional_state_history = deque(maxlen=20)
        
        # Thread locks for concurrent access
        self.db_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        
        # Load existing data
        self.load_user_preferences()
        self.load_recent_conversations()
        
        logger.info("Contextual Memory Manager initialized")
    
    def load_config(self) -> Dict[str, Any]:
        """Load memory configuration"""
        config_file = self.memory_dir / "memory_config.yaml"
        
        default_config = {
            'max_recent_turns': 50,
            'max_long_term_memories': 1000,
            'conversation_summary_threshold': 100,
            'topic_decay_rate': 0.95,
            'importance_threshold': 0.3,
            'auto_summarize_enabled': True,
            'user_preference_learning': True,
            'emotional_continuity': True
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
        
        # Save default config if it doesn't exist
        if not config_file.exists():
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def init_database(self):
        """Initialize SQLite database for persistent memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    user_emotions TEXT,
                    riko_response TEXT NOT NULL,
                    riko_emotions TEXT,
                    topics TEXT,
                    user_satisfaction REAL,
                    session_id TEXT
                )
            """)
            
            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preference_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    last_updated TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    UNIQUE(preference_type)
                )
            """)
            
            # Long-term memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    topics TEXT,
                    timestamp TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
            """)
            
            # Conversation summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_timestamp TEXT NOT NULL,
                    end_timestamp TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_topics TEXT,
                    emotional_tone TEXT,
                    turn_count INTEGER
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_topics ON long_term_memory(topics)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_importance ON long_term_memory(importance)")
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def add_conversation_turn(self, user_input: str, user_emotions: List[str], 
                            riko_response: str, riko_emotions: List[str],
                            topics: List[str] = None, user_satisfaction: float = None):
        """Add a new conversation turn to memory"""
        
        # Extract topics if not provided
        if topics is None:
            topics = self.extract_topics(user_input + " " + riko_response)
        
        # Create conversation turn
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            user_emotions=user_emotions,
            riko_response=riko_response,
            riko_emotions=riko_emotions,
            topics=topics,
            user_satisfaction=user_satisfaction
        )
        
        # Add to recent conversations
        with self.memory_lock:
            self.recent_conversations.append(turn)
            
            # Update active topics
            for topic in topics:
                self.active_topics[topic] = min(1.0, self.active_topics[topic] + 0.1)
            
            # Track emotional state
            self.emotional_state_history.append({
                'timestamp': turn.timestamp,
                'user_emotions': user_emotions,
                'riko_emotions': riko_emotions
            })
        
        # Store in database
        self.store_conversation_turn(turn)
        
        # Update user preferences based on conversation
        if self.config.get('user_preference_learning', True):
            self.learn_user_preferences(turn)
        
        # Check if we need to summarize old conversations
        if (len(self.recent_conversations) >= self.config.get('conversation_summary_threshold', 100) 
            and self.config.get('auto_summarize_enabled', True)):
            self.auto_summarize_old_conversations()
        
        logger.info(f"Added conversation turn with topics: {topics}")
    
    def get_conversation_context(self, max_turns: int = 10) -> str:
        """Get recent conversation context for the LLM"""
        with self.memory_lock:
            recent_turns = list(self.recent_conversations)[-max_turns:]
        
        if not recent_turns:
            return ""
        
        context_parts = ["[RECENT CONVERSATION CONTEXT]"]
        
        for turn in recent_turns:
            # Format: "User (emotions): message" -> "Riko (emotions): response"
            user_emotions_str = f"({', '.join(turn.user_emotions)})" if turn.user_emotions else ""
            riko_emotions_str = f"({', '.join(turn.riko_emotions)})" if turn.riko_emotions else ""
            
            context_parts.append(f"User {user_emotions_str}: {turn.user_input}")
            context_parts.append(f"Riko {riko_emotions_str}: {turn.riko_response}")
            context_parts.append("---")
        
        # Add current topics
        active_topics_str = self.get_active_topics_summary()
        if active_topics_str:
            context_parts.append(f"[CURRENT TOPICS]: {active_topics_str}")
        
        # Add user preferences
        preferences_str = self.get_user_preferences_summary()
        if preferences_str:
            context_parts.append(f"[USER PREFERENCES]: {preferences_str}")
        
        # Add emotional continuity
        emotional_context = self.get_emotional_continuity_context()
        if emotional_context:
            context_parts.append(f"[EMOTIONAL CONTEXT]: {emotional_context}")
        
        return "\n".join(context_parts)
    
    def get_active_topics_summary(self) -> str:
        """Get summary of currently active topics"""
        with self.memory_lock:
            # Decay topic scores
            for topic in list(self.active_topics.keys()):
                self.active_topics[topic] *= self.config.get('topic_decay_rate', 0.95)
                if self.active_topics[topic] < 0.1:
                    del self.active_topics[topic]
            
            # Get top topics
            top_topics = sorted(self.active_topics.items(), key=lambda x: x[1], reverse=True)[:5]
            
        return ", ".join([f"{topic} ({score:.2f})" for topic, score in top_topics])
    
    def get_user_preferences_summary(self) -> str:
        """Get summary of learned user preferences"""
        with self.memory_lock:
            prefs = list(self.user_preferences.values())
        
        # Sort by confidence and frequency
        prefs.sort(key=lambda x: x.confidence * x.frequency, reverse=True)
        
        summary_parts = []
        for pref in prefs[:3]:  # Top 3 preferences
            summary_parts.append(f"{pref.preference_type}: {pref.value} (confidence: {pref.confidence:.2f})")
        
        return "; ".join(summary_parts)
    
    def get_emotional_continuity_context(self) -> str:
        """Get emotional context from recent interactions"""
        with self.memory_lock:
            recent_emotions = list(self.emotional_state_history)[-5:]
        
        if not recent_emotions:
            return ""
        
        # Analyze emotional patterns
        user_emotion_counts = defaultdict(int)
        riko_emotion_counts = defaultdict(int)
        
        for entry in recent_emotions:
            for emotion in entry.get('user_emotions', []):
                user_emotion_counts[emotion] += 1
            for emotion in entry.get('riko_emotions', []):
                riko_emotion_counts[emotion] += 1
        
        context_parts = []
        
        # User's predominant emotions
        if user_emotion_counts:
            top_user_emotion = max(user_emotion_counts.items(), key=lambda x: x[1])
            context_parts.append(f"User has been mostly {top_user_emotion[0]}")
        
        # Riko's recent emotional responses
        if riko_emotion_counts:
            top_riko_emotion = max(riko_emotion_counts.items(), key=lambda x: x[1])
            context_parts.append(f"Riko has been responding with {top_riko_emotion[0]} emotions")
        
        return "; ".join(context_parts)
    
    def learn_user_preferences(self, turn: ConversationTurn):
        """Learn user preferences from conversation turns"""
        
        # Analyze response length preference
        response_length = len(turn.riko_response.split())
        if turn.user_satisfaction is not None and turn.user_satisfaction > 0.7:
            if response_length < 20:
                self.update_preference("response_length", "short", 0.1)
            elif response_length > 50:
                self.update_preference("response_length", "long", 0.1)
            else:
                self.update_preference("response_length", "medium", 0.1)
        
        # Analyze conversation style preferences
        if "question" in turn.user_input.lower():
            self.update_preference("conversation_style", "question_asker", 0.05)
        if any(word in turn.user_input.lower() for word in ["tell me about", "explain", "what is"]):
            self.update_preference("conversation_style", "information_seeker", 0.05)
        
        # Learn topic preferences
        for topic in turn.topics:
            if turn.user_satisfaction is None or turn.user_satisfaction > 0.6:
                self.update_preference(f"topic_interest_{topic}", "high", 0.05)
        
        # Learn emotional response preferences
        if turn.user_emotions and turn.riko_emotions:
            for user_emotion in turn.user_emotions:
                for riko_emotion in turn.riko_emotions:
                    self.update_preference(f"emotional_response_{user_emotion}", riko_emotion, 0.03)
    
    def update_preference(self, pref_type: str, value: str, confidence_delta: float):
        """Update user preference with new information"""
        
        current_time = datetime.now().isoformat()
        
        with self.memory_lock:
            if pref_type in self.user_preferences:
                pref = self.user_preferences[pref_type]
                if pref.value == value:
                    # Reinforce existing preference
                    pref.confidence = min(1.0, pref.confidence + confidence_delta)
                    pref.frequency += 1
                else:
                    # Conflicting preference, adjust confidence
                    pref.confidence = max(0.0, pref.confidence - confidence_delta * 0.5)
                    if pref.confidence < confidence_delta * 2:
                        # Replace with new preference
                        pref.value = value
                        pref.confidence = confidence_delta
                        pref.frequency = 1
                
                pref.last_updated = current_time
                
            else:
                # New preference
                self.user_preferences[pref_type] = UserPreference(
                    preference_type=pref_type,
                    value=value,
                    confidence=confidence_delta,
                    last_updated=current_time,
                    frequency=1
                )
        
        # Store in database
        self.store_user_preference(self.user_preferences[pref_type])
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using simple keyword matching"""
        
        topic_keywords = {
            "anime": ["anime", "manga", "otaku", "japanese animation"],
            "technology": ["computer", "tech", "programming", "AI", "robot", "software"],
            "gaming": ["game", "gaming", "video game", "play", "player"],
            "math": ["math", "mathematics", "calculation", "number", "equation"],
            "school": ["school", "study", "homework", "test", "exam", "class"],
            "food": ["food", "eat", "cooking", "recipe", "restaurant"],
            "music": ["music", "song", "listen", "artist", "band"],
            "movies": ["movie", "film", "cinema", "watch", "director"],
            "personal": ["feel", "emotion", "sad", "happy", "tired", "excited"],
            "help": ["help", "assist", "support", "advice", "guidance"]
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics or ["general"]
    
    def store_conversation_turn(self, turn: ConversationTurn):
        """Store conversation turn in database"""
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO conversations 
                        (timestamp, user_input, user_emotions, riko_response, riko_emotions, topics, user_satisfaction)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        turn.timestamp,
                        turn.user_input,
                        json.dumps(turn.user_emotions),
                        turn.riko_response,
                        json.dumps(turn.riko_emotions),
                        json.dumps(turn.topics),
                        turn.user_satisfaction
                    ))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error storing conversation turn: {e}")
    
    def store_user_preference(self, pref: UserPreference):
        """Store user preference in database"""
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO user_preferences 
                        (preference_type, value, confidence, last_updated, frequency)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        pref.preference_type,
                        json.dumps(pref.value) if isinstance(pref.value, (dict, list)) else str(pref.value),
                        pref.confidence,
                        pref.last_updated,
                        pref.frequency
                    ))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error storing user preference: {e}")
    
    def load_user_preferences(self):
        """Load user preferences from database"""
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM user_preferences")
                    rows = cursor.fetchall()
                    
                    for row in rows:
                        _, pref_type, value_str, confidence, last_updated, frequency = row
                        
                        # Try to parse JSON value, fallback to string
                        try:
                            value = json.loads(value_str)
                        except:
                            value = value_str
                        
                        self.user_preferences[pref_type] = UserPreference(
                            preference_type=pref_type,
                            value=value,
                            confidence=confidence,
                            last_updated=last_updated,
                            frequency=frequency
                        )
                
                logger.info(f"Loaded {len(self.user_preferences)} user preferences")
                
            except Exception as e:
                logger.error(f"Error loading user preferences: {e}")
    
    def load_recent_conversations(self, days: int = 7):
        """Load recent conversations from database"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM conversations 
                        WHERE timestamp > ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (cutoff_date, self.config.get('max_recent_turns', 50)))
                    
                    rows = cursor.fetchall()
                    
                    for row in reversed(rows):  # Reverse to get chronological order
                        _, timestamp, user_input, user_emotions_str, riko_response, riko_emotions_str, topics_str, user_satisfaction, _ = row
                        
                        turn = ConversationTurn(
                            timestamp=timestamp,
                            user_input=user_input,
                            user_emotions=json.loads(user_emotions_str) if user_emotions_str else [],
                            riko_response=riko_response,
                            riko_emotions=json.loads(riko_emotions_str) if riko_emotions_str else [],
                            topics=json.loads(topics_str) if topics_str else [],
                            user_satisfaction=user_satisfaction
                        )
                        
                        self.recent_conversations.append(turn)
                
                logger.info(f"Loaded {len(self.recent_conversations)} recent conversations")
                
            except Exception as e:
                logger.error(f"Error loading recent conversations: {e}")
    
    def auto_summarize_old_conversations(self):
        """Automatically summarize and archive old conversations"""
        # This is a placeholder for conversation summarization
        # In a full implementation, you might use an LLM to create summaries
        
        logger.info("Auto-summarizing old conversations (placeholder)")
        
        # For now, just remove oldest conversations when we exceed the limit
        with self.memory_lock:
            while len(self.recent_conversations) > self.config.get('max_recent_turns', 50):
                old_turn = self.recent_conversations.popleft()
                
                # Create a simple summary entry
                summary_content = f"User asked about {', '.join(old_turn.topics)}, Riko responded with {', '.join(old_turn.riko_emotions)} emotions"
                
                # Store as long-term memory if important enough
                importance = 0.5  # Default importance
                if old_turn.user_satisfaction and old_turn.user_satisfaction > 0.8:
                    importance = 0.8
                elif old_turn.topics and any(topic in ["personal", "help"] for topic in old_turn.topics):
                    importance = 0.7
                
                if importance > self.config.get('importance_threshold', 0.3):
                    memory_id = hashlib.md5(f"{old_turn.timestamp}_{summary_content}".encode()).hexdigest()
                    memory_entry = MemoryEntry(
                        id=memory_id,
                        content=summary_content,
                        importance=importance,
                        topics=old_turn.topics,
                        timestamp=old_turn.timestamp
                    )
                    self.store_long_term_memory(memory_entry)
    
    def store_long_term_memory(self, memory: MemoryEntry):
        """Store a long-term memory entry"""
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO long_term_memory
                        (id, content, importance, topics, timestamp, access_count, last_accessed)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        memory.id,
                        memory.content,
                        memory.importance,
                        json.dumps(memory.topics),
                        memory.timestamp,
                        memory.access_count,
                        memory.last_accessed
                    ))
                    conn.commit()
                    logger.info(f"Stored long-term memory: {memory.id}")
            except Exception as e:
                logger.error(f"Error storing long-term memory: {e}")
    
    def search_memories(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Search long-term memories for relevant content"""
        query_topics = self.extract_topics(query)
        
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Search by topic relevance and importance
                    memories = []
                    
                    for topic in query_topics:
                        cursor.execute("""
                            SELECT * FROM long_term_memory 
                            WHERE topics LIKE ? 
                            ORDER BY importance DESC, access_count DESC
                            LIMIT ?
                        """, (f'%{topic}%', limit))
                        
                        rows = cursor.fetchall()
                        for row in rows:
                            memory_id, content, importance, topics_str, timestamp, access_count, last_accessed = row
                            
                            memory = MemoryEntry(
                                id=memory_id,
                                content=content,
                                importance=importance,
                                topics=json.loads(topics_str) if topics_str else [],
                                timestamp=timestamp,
                                access_count=access_count,
                                last_accessed=last_accessed or ""
                            )
                            memories.append(memory)
                    
                    # Remove duplicates and sort by relevance
                    unique_memories = {m.id: m for m in memories}.values()
                    sorted_memories = sorted(unique_memories, key=lambda x: x.importance, reverse=True)
                    
                    # Update access count for retrieved memories
                    current_time = datetime.now().isoformat()
                    for memory in sorted_memories[:limit]:
                        cursor.execute("""
                            UPDATE long_term_memory 
                            SET access_count = access_count + 1, last_accessed = ?
                            WHERE id = ?
                        """, (current_time, memory.id))
                    
                    conn.commit()
                    return list(sorted_memories[:limit])
                    
            except Exception as e:
                logger.error(f"Error searching memories: {e}")
                return []
    
    def get_memory_context_for_llm(self, user_input: str) -> str:
        """Get relevant memory context for the current user input"""
        
        # Get conversation context
        conversation_context = self.get_conversation_context()
        
        # Search for relevant long-term memories
        relevant_memories = self.search_memories(user_input)
        
        context_parts = [conversation_context]
        
        if relevant_memories:
            context_parts.append("[RELEVANT MEMORIES]")
            for memory in relevant_memories:
                context_parts.append(f"Memory: {memory.content} (importance: {memory.importance:.2f})")
        
        return "\n".join(context_parts)
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old conversation data"""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        with self.db_lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Remove old conversations
                    cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (cutoff_date,))
                    deleted_conversations = cursor.rowcount
                    
                    # Remove low-importance old memories
                    cursor.execute("""
                        DELETE FROM long_term_memory 
                        WHERE timestamp < ? AND importance < ?
                    """, (cutoff_date, 0.5))
                    deleted_memories = cursor.rowcount
                    
                    conn.commit()
                    
                    logger.info(f"Cleaned up {deleted_conversations} old conversations and {deleted_memories} low-importance memories")
                    
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


# Singleton instance
_context_manager_instance = None
_instance_lock = threading.Lock()

def get_context_manager(**kwargs) -> ContextualMemoryManager:
    """Get singleton instance of ContextualMemoryManager"""
    global _context_manager_instance
    
    with _instance_lock:
        if _context_manager_instance is None:
            _context_manager_instance = ContextualMemoryManager(**kwargs)
        return _context_manager_instance


if __name__ == "__main__":
    # Test the memory system
    manager = get_context_manager()
    
    # Test adding conversation turns
    manager.add_conversation_turn(
        user_input="Hi Riko, how are you today?",
        user_emotions=["happy", "curious"],
        riko_response="Ugh, fine I guess. What do you want, Senpai?",
        riko_emotions=["annoyed", "tsundere"],
        topics=["greeting", "personal"]
    )
    
    manager.add_conversation_turn(
        user_input="Can you help me with my math homework?",
        user_emotions=["confused", "hopeful"],
        riko_response="Math? Really? Fine, I'll help you, but don't expect me to do everything for you!",
        riko_emotions=["helpful", "tsundere"],
        topics=["math", "school", "help"],
        user_satisfaction=0.8
    )
    
    # Test getting context
    context = manager.get_memory_context_for_llm("I need more help with math")
    print("Memory Context:")
    print(context)
    
    print("\nMemory system test completed!")
