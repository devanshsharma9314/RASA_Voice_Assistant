import speech_recognition as sr
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import pyttsx3
import json
import pygame
import io
import requests
import json
import threading
import time
import os
from typing import Optional as Opt
from dotenv import load_dotenv


load_dotenv()
mcp = FastMCP("voice-agent")

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
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=256)
        #self.setup_tts()
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
        

    def text_to_speech_elevenlabs(self, text: str) -> Opt[bytes]:
        """Convert text to speech using ElevenLabs API"""
        if not self.elevenlabs_api_key:
            print("ElevenLabs API key not provided!")
            return None
            
        try:
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_turbo_v2",  # Fast, good quality
                "voice_settings": {
                    "stability": 0.5,      # 0-1, higher = more stable
                    "similarity_boost": 0.75,  # 0-1, higher = more similar to original voice
                    "style": 0.0,          # 0-1, only for v2 models
                    "use_speaker_boost": True
                },
               # "optimize_streaming_latency": 4,  # Maximum optimization
               # "output_format": "mp3_22050_32"   # Lower quality for speed
            }
            
            response = requests.post(
                f"{self.elevenlabs_url}/{self.voice_id}",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ElevenLabs: {e}")
            return None
    
    def play_audio(self, audio_data: bytes):
        """Play audio using pygame"""
        try:
            audio_stream = io.BytesIO(audio_data)
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            
            # Wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Audio playback error: {e}")

    def speak_text(self, text: str):
        """Convert text to speech and play it"""
        print(f"Bot says: {text}")
        
        if not text.strip():
            return
            
        # Generate speech with ElevenLabs
        audio_data = self.text_to_speech_elevenlabs(text)
        
        if audio_data:
            self.play_audio(audio_data)
        else:
            print("Failed to generate speech")


    



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

ELEVENLABS_API_KEY = os.getenv("ELEVEN_LABS_API2")
agent = VoiceAgent(elevenlabs_api_key=ELEVENLABS_API_KEY)




if __name__ == "__main__":
    # Start voice agent
    
    ELEVENLABS_API_KEY = os.getenv("ELEVEN_LABS_API2")
    agent = VoiceAgent(elevenlabs_api_key=ELEVENLABS_API_KEY)  # ← API key passed correctly
    agent.run_conversation()
