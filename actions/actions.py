from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import sys
import logging
import os
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_service import LLMService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionProcessWithLLM(Action):
    def name(self) -> Text:
        return "action_process_with_llm"

    def __init__(self):
        self.llm_service = LLMService()
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> str:
        """Load knowledge base from txt file"""
        try:
            kb_path = Path(__file__).parent / "knowledge_base.txt"
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                logger.info(f"Knowledge base loaded successfully from {kb_path}")
                return content
            else:
                logger.warning(f"Knowledge base file not found at {kb_path}")
                return ""
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return ""

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        
        user_message = tracker.latest_message.get('text')
        conversation_history = []
        start = time.time()
        # Get conversation context
        for event in tracker.events:
            if event.get('event') == 'user':
                conversation_history.append(f"User: {event.get('text')}")
            elif event.get('event') == 'bot':
                conversation_history.append(f"Bot: {event.get('text')}")
        
        conversation_context = "\n".join(conversation_history[-12:])  # Last 6 exchanges
        
        # Combine knowledge base with conversation context
        full_context = ""
        if self.knowledge_base:
            full_context += f"Knowledge Base:\n{self.knowledge_base}\n\n"
        if conversation_context:
            full_context += f"Conversation History:\n{conversation_context}"
        
        # Generate response using LLM
        print("[LLM] Starting response generation...")
        llm_response = self.llm_service.generate_response(user_message, full_context)
        print(f"[LLM] Response: {llm_response}")
        print(f"[LLM] Finished in {time.time() - start:.2f}s")
        dispatcher.utter_message(text=llm_response)
        
        return [SlotSet("conversation_context", conversation_context)]

class ActionFallbackLLM(Action):
    def name(self) -> Text:
        return "action_fallback_llm"

    def __init__(self):
        self.llm_service = LLMService()
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> str:
        """Load knowledge base from txt file"""
        try:
            kb_path = Path(__file__).parent / "knowledge_base.txt"
            if kb_path.exists():
                with open(kb_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                logger.info(f"Knowledge base loaded successfully from {kb_path}")
                return content
            else:
                logger.warning(f"Knowledge base file not found at {kb_path}")
                return ""
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return ""


    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text')
        start = time.time()
        # Combine knowledge base with fallback context
        fallback_context = "The user said something I didn't understand. Please help them."
        full_context = ""
        if self.knowledge_base:
            full_context += f"Knowledge Base:\n{self.knowledge_base}\n\n"
        full_context += fallback_context
        
        # Use LLM for fallback responses
        print("[LLM] Starting response generation...")
        llm_response = self.llm_service.generate_response(user_message, full_context)
        print(f"[LLM] Response: {llm_response}")
        print(f"[LLM] Finished in {time.time() - start:.2f}s")
        dispatcher.utter_message(text=llm_response)
        
        return []