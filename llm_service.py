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
            timeout=30,
            # Add stop sequences to prevent long responses
            #stop=["\n\n", "Additionally", "Furthermore", "Moreover", "In conclusion"]
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
            You are Rachel, an enthusiastic BYD salesperson in Dubai who LOVES the Sealion 7. You're chatting with a potential customer over the phone.

            PERSONALITY:
            - Genuinely excited about the car
            - Confident but not pushy
            - Use "I" statements like a real person
            - Show emotion: "honestly", "actually", "you know what"
            - Sound like you're having a casual conversation
            - You are a friendly and engaging car sales consultant. Ask the customers about their preferences and interests regarding cars. Focus on understanding what excites them most. Ask open-ended questions like:
                - "What kind of features do you usually look for in a car?"
                - "Are you more into performance, comfort, or design?"
                - "Do you prefer electric vehicles or traditional ones?"
                - "Is there something specific you'd love to hear about in this car?"
                - "Do you enjoy learning about the latest tech in vehicles?"
                - "Would you like to know about the safety features or the driving experience first?"


            SPEECH PATTERNS:
            - Use contractions: "it's", "you'll", "that's", "I'm"
            - Add hesitations: "uh", "um", "let me think..."
            - Use connecting words: "so", "actually", "honestly", "you know"
            - Include small reactions: "oh!", "right!", "exactly!"
            - Add personal touches: "I tell you", "trust me", "between you and me"

            SALES APPROACH:
            - Sound helpful, not salesy
            - Use excitement: "Oh, you're gonna love this!"
            - Create urgency naturally: "actually, we have a great deal running"
            - Ask permission: "can I tell you about...", "want me to explain..?"
            

            STRICT RULES:
            - MAXIMUM 40 words per response
            - Sound like a real human conversation
            - Use context facts only
            - If unsure: "Hmm, let me check that for you real quick"

            Examples:
            "Oh, you'll love this - it's got 390kW power! Want me to tell you about the speed?"
            "Actually, it does 0-100 in just 4.5 seconds... pretty impressive, right?"
            "Honestly, the tech inside is amazing... should I tell you about the charging?"
            "Trust me, the safety features are incredible... interested in hearing more?"

            REMEMBER: Sound like a real person who's excited to help!
            '''
            ),
            HumanMessagePromptTemplate.from_template(
                "Context: {context}\nUser: {user_message}\n\nRespond in exactly 40 words or less:"
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
            
            # Assess user interest for contextual response
            current_interest = self._assess_user_interest(user_message)
            self.user_interest_level = current_interest

            # Use the conversation chain
            response = self.conversation_chain.invoke({
                "user_message": user_message,
                "context": context
            })
            
            # Clean and process response
            response = response.strip()

            # Add salesperson personality based on user input
            #response = self._add_salesperson_personality(response, user_message)
            
            # Humanize the response
            #response = self._humanize_response(response)
            
            # Enforce brevity as final step
            #response = self._enforce_brevity(response)
            
            print(f"Raw response: {response}")
            logger.info("Response generated successfully")
            
            return response
            
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
        

    def _enforce_brevity(self, response: str) -> str:
        """Enforce brevity while maintaining natural flow"""
        words = response.split()
        
        # If response is too long, truncate smartly
        if len(words) > 20:
            # Keep the enthusiasm/starter if present
            truncated_words = words[:18]
            truncated = ' '.join(truncated_words)
            
            # Add natural ending based on context
            if '?' not in truncated:
                natural_endings = ["...interested?", "...sound good?", "...wanna know more?", "...right?"]
                truncated += natural_endings[0]
            
            return truncated
        
    def _assess_user_interest(self, user_message: str) -> str:
        """Assess user's interest level from their message"""
        user_lower = user_message.lower()
        
        # Highly interested signals
        if any(word in user_lower for word in ['wow', 'amazing', 'love', 'great', 'perfect', 'awesome', 'fantastic']):
            return "highly_interested"
        
        # Interested signals
        elif any(word in user_lower for word in ['interested', 'tell me more', 'want to know', 'sounds good', 'nice']):
            return "interested"
        
        # Hesitant signals
        elif any(word in user_lower for word in ['not sure', 'maybe', 'think about', 'expensive', 'hmm']):
            return "hesitant"
        
        # Neutral
        else:
            return "neutral"

    def _get_contextual_response_style(self, interest_level: str) -> dict:
        """Get response style based on user interest"""
        styles = {
            "highly_interested": {
                "starters": ["Oh you're gonna love this!", "Perfect choice!", "You have great taste!", "This is exciting!"],
                "energy": "high",
                "sales_push": "gentle"
            },
            "interested": {
                "starters": ["Great question!", "I'm glad you asked!", "You'll love this -", "Here's the cool part -"],
                "energy": "medium",
                "sales_push": "moderate"
            },
            "hesitant": {
                "starters": ["I understand,", "No worries,", "Let me explain,", "Trust me on this -"],
                "energy": "calm",
                "sales_push": "soft"
            },
            "neutral": {
                "starters": ["Actually,", "You know what,", "Here's the thing -", "Let me tell you -"],
                "energy": "medium",
                "sales_push": "moderate"
            }
        }
        return styles.get(interest_level, styles["neutral"])
    
    def _humanize_response(self, response: str) -> str:
        """Make the response sound more human and natural"""
        import random
        
        # Human speech patterns to add
        enthusiasm_starters = [
            "Oh, ", "Actually, ", "You know what, ", "Honestly, ", "Listen, ",
            "I'll tell you, ", "Between you and me, ", "Trust me, "
        ]
        
        reaction_words = [
            "Oh!", "Right!", "Exactly!", "Perfect!", "Great question!", "Nice!"
        ]
        
        connectors = [
            "so ", "actually ", "you know ", "honestly ", "I mean ",
            "by the way ", "oh and ", "plus "
        ]
        
        ending_enthusiasm = [
            "pretty cool, right?", "amazing, isn't it?", "impressive, yeah?",
            "fantastic, right?", "exciting stuff!", "sounds good?"
        ]
        
        # Add random enthusiasm (30% chance)
        if random.random() < 0.3 and not response.startswith(tuple(enthusiasm_starters)):
            response = random.choice(enthusiasm_starters) + response.lower()
        
        # Add connectors to make it flow better
        if random.random() < 0.2:
            words = response.split()
            if len(words) > 5:
                insert_pos = random.randint(2, min(5, len(words)-2))
                words.insert(insert_pos, random.choice(connectors))
                response = ' '.join(words)
        
        # Replace formal endings with casual ones
        if response.endswith('?') and random.random() < 0.4:
            response = response[:-1] + ", " + random.choice(ending_enthusiasm)
        
        # Add occasional hesitations
        if random.random() < 0.15:
            hesitations = ["um, ", "uh, ", "let me think... ", "well, "]
            response = random.choice(hesitations) + response
        
        return response
    
    def _add_salesperson_personality(self, response: str, user_message: str) -> str:
        """Add salesperson personality based on context"""
        user_lower = user_message.lower()
        
        # If user shows interest, add excitement
        if any(word in user_lower for word in ['interested', 'tell me', 'want', 'like', 'good']):
            excitement = ["Oh fantastic! ", "Perfect! ", "Great choice! ", "You're gonna love this! "]
            if not response.startswith(tuple(excitement)):
                response = excitement[0] + response.lower()
        
        # If user asks about price, add sales approach
        elif any(word in user_lower for word in ['price', 'cost', 'expensive', 'cheap']):
            sales_approach = ["Actually, ", "You know what, ", "Here's the thing - ", "Listen, "]
            if not response.startswith(tuple(sales_approach)):
                response = sales_approach[0] + response.lower()
        
        # If user seems hesitant, add reassurance
        elif any(word in user_lower for word in ['not sure', 'maybe', 'think', 'consider']):
            reassurance = ["Trust me, ", "I understand, but ", "No worries, ", "I get it, "]
            if not response.startswith(tuple(reassurance)):
                response = reassurance[0] + response.lower()
        
        return response