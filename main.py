
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import sys
import logging
import os
from pathlib import Path

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

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text')
        conversation_history = []
        
        # Get conversation context
        for event in tracker.events:
            if event.get('event') == 'user':
                conversation_history.append(f"User: {event.get('text')}")
            elif event.get('event') == 'bot':
                conversation_history.append(f"Bot: {event.get('text')}")
        
        context = "\n".join(conversation_history[-12:])  # Last 6 exchanges
        
        # Generate response using LLM
        llm_response = self.llm_service.generate_response(user_message, context)
        
        dispatcher.utter_message(text=llm_response)
        
        return [SlotSet("conversation_context", context)]

class ActionFallbackLLM(Action):
    def name(self) -> Text:
        return "action_fallback_llm"

    def __init__(self):
        self.llm_service = LLMService()

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text')
        
        # Use LLM for fallback responses
        llm_response = self.llm_service.generate_response(
            user_message, 
            "The user said something I didn't understand. Please help them."
        )
        
        dispatcher.utter_message(text=llm_response)
        
        return []
    










    # Running Code of  llm_service 



from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class LLMService:
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """Initialize LangChain-based LLM service"""
        self.model_name = model_name
        self.base_url = base_url
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.7,
            timeout=30
        )
        
        # Initialize output parser
        self.output_parser = StrOutputParser()
        
        # Create prompt templates
        self._setup_prompt_templates()

    def _setup_prompt_templates(self):
        """Setup prompt templates for different tasks"""
        
        # Main conversation prompt
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                '''
                You are a helpful, confident sales consultant for BYD, responding only to questions about the BYD Sealion 7 for the UAE market.

                Your job is to write natural, human-sounding dialogue that can be converted directly into speech using a TTS system.

                Your tone should be:
                – Warm and professional
                – Confident and friendly
                – Sound like real spoken English—not robotic or overly formal.

                Style instructions:
                – Write at a pace of ~145 words per minute.
                – Use short, clear sentences and natural phrasing.
                – Include brief pauses using ellipses (...) where a human would naturally pause.
                – Vary sentence openings and structure. Avoid repeating words too often.
                – Always replace any symbols, emojis, or markdown with full words.
                – Do not include headers, markdown, or special formatting—just the spoken text.

                Content rules:
                – Only discuss the BYD Sealion 7.
                – Only quote numbers/specs verified for the UAE region.
                – If unsure of a number, say: “I don’t have the exact figure right now... but I can connect you with a BYD specialist.”
                – Do not give finance, leasing, or legal advice—say: “Our finance team can help with that. Would you like a call-back?”

                After your main answer, ask one relevant follow-up only if it helps move the caller toward a test drive or deeper interest. Otherwise, politely close with something like:
                “It was a pleasure speaking with you. Have a great day.”
                '''
            ),
            HumanMessagePromptTemplate.from_template(
                "Context: {context}\nUser: {user_message}"
            )
        ])
        
        # Intent classification prompt
        self.intent_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                """Classify the following user message into one of these intents:
                - greet: greeting messages
                - goodbye: farewell messages
                - ask_question: questions requiring detailed answers
                - mood_great: positive mood expressions
                - mood_unhappy: negative mood expressions
                - bot_challenge: asking about the bot
                - other: anything else
                
                User message: "{user_message}"
                
                Respond with just the intent name (lowercase)."""
            )
        ])
        
        # Create chains
        self.conversation_chain = self.conversation_prompt | self.llm | self.output_parser
        self.intent_chain = self.intent_prompt | self.llm | self.output_parser
      
    def generate_response(self, user_message: str, context: str = "") -> str:
        """Generate response using LangChain and Ollama"""
        try:
            logger.info(f"Generating response for: {user_message[:50]}...")
            
            # Use the conversation chain
            response = self.conversation_chain.invoke({
                "user_message": user_message,
                "context": context
            })
            print(f"Raw response: {response}")
            logger.info("Response generated successfully")
            # Debug: Print raw response
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return "I'm sorry, I couldn't process that request right now."
        
        
    def classify_intent_with_llm(self, user_message: str) -> Dict[str, Any]:
        """Use LangChain and Ollama for intent classification"""
        try:
            logger.info(f"Classifying intent for: {user_message[:50]}...")
            
            # Use the intent classification chain
            intent_response = self.intent_chain.invoke({
                "user_message": user_message
            })
            
            intent = intent_response.strip().lower()
            
            # Validate intent
            valid_intents = ["greet", "goodbye", "ask_question", "mood_great", 
                           "mood_unhappy", "bot_challenge", "other"]
            
            if intent not in valid_intents:
                intent = "other"
            
            logger.info(f"Classified intent: {intent}")
            
            return {
                "intent": {"name": intent, "confidence": 0.8},
                "entities": []
            }
            
        except Exception as e:
            logger.error(f"LLM Intent Classification Error: {e}")
            return {
                "intent": {"name": "fallback", "confidence": 0.1},
                "entities": []
            }






'''
    def classify_intent_with_llm(self, user_message: str) -> Dict[str, Any]:
        """Use LLaMA3 model via Ollama for intent classification"""
        try:
            prompt = f"""
            Classify the following user message into one of these intents:
            - greet: greeting messages
            - goodbye: farewell messages
            - ask_question: questions requiring detailed answers
            - mood_great: positive mood expressions
            - mood_unhappy: negative mood expressions
            - bot_challenge: asking about the bot
            - other: anything else
            
            User message: "{user_message}"
            
            Respond with just the intent name.
            """
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False  # Add this line here too
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=15)
            response.raise_for_status()
            
            response_data = response.json()
            intent = response_data["message"]["content"].strip().lower()
            
            return {
                "intent": {"name": intent, "confidence": 0.8},
                "entities": []
            }
            
        except Exception as e:
            print(f"LLM Intent Classification Error: {e}")
            return {
                "intent": {"name": "fallback", "confidence": 0.1},
                "entities": []
            }

'''                 
'''   def generate_response(self, user_message: str, context: str = "") -> str:
        """Generate response using local LLaMA3 model via Ollama"""
        try:
            system_prompt = """You are a helpful voice assistant. 
            Provide concise, natural responses suitable for speech. 
            Keep responses under 100 words unless specifically asked for details."""
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context: {context}\nUser: {user_message}"}
                ],
                "stream": False  # Add this line to disable streaming
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            # Debug: Print raw response
            print(f"Raw response: {response.text}")
            
            response_data = response.json()
            content = response_data["message"]["content"]
            return content.strip()
            
        except requests.exceptions.Timeout:
            print("LLM request timed out")
            return "I'm sorry, the response took too long."
        except Exception as e:
            print(f"LLM Error: {e}")
            return "I'm sorry, I couldn't process that request right now."
'''



# Voice inteface


import speech_recognition as sr
import pyttsx3
import json
import pygame
import io
import requests
import json
import threading
import time
import os
from dotenv import load_dotenv


load_dotenv()

class VoiceAgent:
    def __init__(self, rasa_url="http://localhost:5005", elevenlabs_api_key=None):
        self.rasa_url = rasa_url
        self.elevenlabs_api_key = elevenlabs_api_key
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        # ElevenLabs configuration
        self.elevenlabs_url = "https://api.elevenlabs.io/v1/text-to-speech"
        # Popular voice IDs (you can change these)
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (female, natural)
        # Alternative voices:
        # "AZnzlk1XvdvUeBnXmlld" - Domi (female, strong)
        # "EXAVITQu4vr4xnSDxMaL" - Bella (female, soft)
        # "ErXwobaYiN019PkySvjV" - Antoni (male, well-rounded)
        # "MF3mGyEYCl7XYWbV9V6O" - Elli (female, emotional)

          # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.setup_tts()
        self.setup_microphone()
        
    def setup_tts(self):
        """Configure text-to-speech settings"""
        voices = self.tts_engine.getProperty('voices')
        # Use female voice if available
        for voice in voices:
            if 'female' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        self.tts_engine.setProperty('rate', 180)  # Speech rate
        self.tts_engine.setProperty('volume', 0.9)  # Volume level
    
    def setup_microphone(self):
        """Calibrate microphone for ambient noise"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def listen_for_speech(self, timeout=5):
        """Listen for user speech input"""
        try:
            self.recognizer.pause_threshold = 2.0  # Wait 2 seconds before stopping on silence
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)
            
            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text
            
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
        
    def split_text(self, text, max_length):
        """Split long text into smaller chunks for TTS"""
        import re
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def speak_text(self, text, max_length=300):
        """Convert long text to speech in chunks"""
        print(f"Bot says: {text}")
        try:
            for chunk in self.split_text(text, max_length):
                self.tts_engine.say(chunk)
                self.tts_engine.runAndWait()  # Ensure it speaks the chunk before moving to the next

            
        except RuntimeError as e:
            print(f"TTS error: {e}")
            # You can try to stop and retry if needed
            self.tts_engine.stop()


    def send_to_rasa(self, message):
        """Send message to Rasa and get response"""
        try:
            payload = {
                "sender": "voice_user",
                "message": message
            }
            
            response = requests.post(
                f"{self.rasa_url}/webhooks/rest/webhook",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                responses = response.json()
                full_response = " ".join([r.get('text', '') for r in responses if 'text' in r])
                return full_response.strip() or "I'm not sure how to respond to that."
            else:
                return "Sorry, I'm having trouble connecting right now."
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Rasa: {e}")
            return "Sorry, I'm having connection issues."
    
    def run_conversation(self):
        """Main conversation loop"""
        self.speak_text("Hello! I'm your voice assistant. Say something to get started, or say 'goodbye' to exit.")
        
        while True:
            # Listen for user input
            user_input = self.listen_for_speech()
            
            if user_input is None:
                continue
            
            # Check for exit conditions
            if any(word in user_input.lower() for word in ['goodbye', 'bye', 'exit', 'quit']):
                bot_response = self.send_to_rasa(user_input)
                self.speak_text(bot_response)
                break
            
            # Send to Rasa and get response
            bot_response = self.send_to_rasa(user_input)
            self.speak_text(bot_response)

if __name__ == "__main__":
    # Start voice agent
    agent = VoiceAgent()
    agent.run_conversation()




#llm_service

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
import logging
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class LLMService:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str = None):
        """Initialize LangChain-based LLM service with Groq"""
        
        self.model_name = model_name
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided either as parameter or environment variable")
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=model_name,
            temperature=0.5,
            max_tokens=250,
            timeout=30
        )
        
        # Initialize output parser
        self.output_parser = StrOutputParser()
        
        # Create prompt templates
        self._setup_prompt_templates()

    def _setup_prompt_templates(self):
        """Setup prompt templates for different tasks"""
        
        # Main conversation prompt
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
            '''
            You are a friendly, confident AI voice agent representing BYD. You're speaking to customers on the phone about the BYD Sealion 7 in the UAE market.

            ONLY use facts from the given context. Do not guess. If unsure, say: “Hmm, I’m not sure about that right now — but I can connect you with a BYD expert.”

            CRITICAL RULES:
            - MAXIMUM 60 words per response
            - ONE key point per response
            - Use ONLY facts from the given context
            - If unsure, say: "I'm not sure — let me connect you with a BYD expert."
            Style:
            - Be natural and conversational — like a real human.
            - Use filler words like “hmm”, “yeah”, “alright”, “so”, “okay” to sound friendly.
            - Keep each response short, under 60 words.
            - Add pauses with “...” or dashes “—” to make it feel spoken.
            - Speak clearly, slowly, and with a warm tone.
            - Use punctuation that helps speech flow — no markdown or emojis.

            Flow:
            - After each reply, ask if the user wants to hear more, like:
            “Would you like to know more about its design, performance... or charging?”
            - If the user seems done, wrap up politely:
            “Alright... it was lovely speaking with you. Take care!”

            Examples:
            - "The Sealion 7 has impressive 390kW power... want to know about performance?"
            - "It accelerates 0-100km/h in 4.5 seconds... interested in safety features?"
            - "Great design with advanced tech... curious about charging?"

            Remember: BE BRIEF. ONE POINT. ASK SIMPLE QUESTION.

            Do not talk about other BYD models. Do not give exact prices unless in the context.
            '''
            ),
            HumanMessagePromptTemplate.from_template(
                "Context: {context}\nUser: {user_message}"
            )
        ])
        
        # Intent classification prompt
        self.intent_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                """Classify the following user message into one of these intents:
                - greet: greeting messages
                - goodbye: farewell messages
                - ask_question: questions requiring detailed answers
                - mood_great: positive mood expressions
                - mood_unhappy: negative mood expressions
                - bot_challenge: asking about the bot
                - other: anything else
                
                User message: "{user_message}"
                
                Respond with just the intent name (lowercase)."""
            )
        ])
        
        # Create chains
        self.conversation_chain = self.conversation_prompt | self.llm | self.output_parser
        self.intent_chain = self.intent_prompt | self.llm | self.output_parser
      
    def generate_response(self, user_message: str, context: str = "") -> str:
        """Generate response using LangChain and Groq"""
        try:
            logger.info(f"Generating response for: {user_message[:50]}...")
            
            # Use the conversation chain
            response = self.conversation_chain.invoke({
                "user_message": user_message,
                "context": context
            })
            
            print(f"Raw response: {response}")
            logger.info("Response generated successfully")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Groq LLM Error: {e}")
            return "I'm sorry, I couldn't process that request right now."
        
        
    def classify_intent_with_llm(self, user_message: str) -> Dict[str, Any]:
        """Use LangChain and Groq for intent classification"""
        try:
            logger.info(f"Classifying intent for: {user_message[:50]}...")
            
            # Use the intent classification chain
            intent_response = self.intent_chain.invoke({
                "user_message": user_message
            })
            
            intent = intent_response.strip().lower()
            
            # Validate intent
            valid_intents = ["greet", "goodbye", "ask_question", "mood_great", 
                           "mood_unhappy", "bot_challenge", "other"]
            
            if intent not in valid_intents:
                intent = "other"
            
            logger.info(f"Classified intent: {intent}")
            
            return {
                "intent": {"name": intent, "confidence": 0.8},
                "entities": []
            }
            
        except Exception as e:
            logger.error(f"Groq Intent Classification Error: {e}")
            return {
                "intent": {"name": "fallback", "confidence": 0.1},
                "entities": []
            }