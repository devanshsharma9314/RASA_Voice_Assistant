# RASA Voice Assistant

A sophisticated voice-enabled conversational AI assistant built with RASA, designed to serve as a sales consultant for the BYD Sealion 7 in the UAE market. This assistant combines natural language processing, speech recognition, and text-to-speech capabilities to provide an interactive voice experience.

## 🚀 Features

### Core Capabilities
- **Voice Interface**: Real-time speech recognition and text-to-speech conversion
- **Conversational AI**: Powered by RASA framework with advanced NLU processing
- **LLM Integration**: Enhanced responses using Groq LLM for natural conversations
- **BYD Sealion 7 Expertise**: Specialized knowledge about the BYD Sealion 7 vehicle
- **Multi-modal Communication**: Supports both text and voice interactions

### Technical Features
- **Speech Recognition**: Google Speech Recognition API integration
- **Text-to-Speech**: ElevenLabs API for high-quality voice synthesis
- **Fallback Handling**: Intelligent fallback mechanisms for unrecognized inputs
- **Context Awareness**: Maintains conversation context for coherent dialogues
- **Intent Classification**: Advanced intent recognition with confidence scoring

## 🏗️ Architecture

```
RASA Voice Assistant
├── RASA Core (Conversation Management)
├── NLU Pipeline (Intent Recognition)
├── LLM Service (Groq Integration)
├── Voice Interface (Speech I/O)
└── Custom Actions (Business Logic)
```

## 📋 Prerequisites

- Python 3.8+
- RASA 3.x
- Microphone and speakers
- Internet connection for speech recognition and TTS

### API Keys Required
- **ElevenLabs API Key**: For high-quality text-to-speech
- **Groq API Key**: For LLM-powered responses

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RASA_Voice_Assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv rasa_env310
   source rasa_env310/bin/activate  # On Windows: rasa_env310\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   ELEVEN_LABS_API2=your_elevenlabs_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 🚀 Quick Start

### 1. Train the RASA Model
```bash
rasa train
```

### 2. Start RASA Server
```bash
rasa run --enable-api --cors "*" --debug
```

### 3. Start Voice Interface
```bash
python voice_interface.py
```


## 📁 Project Structure

```
RASA_Voice_Assistant/
├── actions/                 # Custom RASA actions
│   ├── __init__.py
│   ├── actions.py          # LLM integration actions
│   └── knowledge_base.txt  # BYD Sealion 7 information
├── data/                   # Training data
│   ├── nlu.yml            # Intent and entity training
│   ├── stories.yml        # Conversation flows
│   └── rules.yml          # Business rules
├── models/                 # Trained RASA models
├── config.yml             # RASA configuration
├── domain.yml             # Domain definition
├── credentials.yml        # API credentials
├── endpoints.yml          # Service endpoints
├── voice_interface.py    # Voice interface implementation
├── llm_service.py        # LLM service integration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎯 Usage Examples

### Voice Interaction
1. **Start the assistant**: "Hello" or "Hi"
2. **Ask about BYD Sealion 7**: "Tell me about the Sealion 7"
3. **Performance questions**: "What's the acceleration like?"
4. **Exit**: "Goodbye" or "Bye"

### Supported Intents
- `greet`: Greeting messages
- `goodbye`: Farewell messages
- `ask_question`: General questions
- `ask_weather`: Weather-related queries
- `bot_challenge`: Questions about the bot
- `nlu_fallback`: Unrecognized inputs

## 🔧 Configuration

### RASA Configuration (`config.yml`)
- **Pipeline**: DIET classifier with 100 epochs
- **Policies**: Memoization, Rule, UnexpecTEDIntent, and TED policies
- **Fallback**: 0.3 threshold with ambiguity detection

### Voice Settings
- **Speech Rate**: 180 words per minute
- **Voice**: Female voice (configurable)
- **Audio Quality**: High-quality ElevenLabs synthesis

### LLM Configuration
- **Model**: Groq Llama-3.1-8b-instant
- **Temperature**: 0.5 for balanced creativity
- **Max Tokens**: 250 for concise responses

## 🧪 Testing

### Test Stories
```bash
rasa test --stories tests/test_stories.yml
```

### Interactive Testing
```bash
rasa shell
```

### Voice Testing
```bash
python voice_interface.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Speech Recognition Not Working**
   - Check microphone permissions
   - Ensure internet connection for Google Speech API
   - Verify microphone is properly connected

2. **TTS Not Working**
   - Verify ElevenLabs API key in `.env` file
   - Check internet connection
   - Ensure audio output is working

3. **RASA Server Issues**
   - Check if port 5005 is available
   - Verify model training completed successfully
   - Check logs for specific error messages

4. **LLM Service Errors**
   - Verify Groq API key
   - Check internet connection
   - Ensure API quota is not exceeded

### Debug Mode
Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance

### Model Performance
- **Training Time**: ~5-10 minutes (depending on hardware)
- **Response Time**: <2 seconds for voice interactions
- **Accuracy**: High intent recognition with fallback handling

### Voice Quality
- **ElevenLabs**: High-quality, natural-sounding speech
- **Latency**: Optimized for real-time conversation
- **Customization**: Configurable voice settings

## 🔒 Security

- API keys stored in environment variables
- No sensitive data in code repository
- Secure HTTPS communication for APIs
- Local speech processing where possible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## 🙏 Acknowledgments

- **RASA**: Open-source conversational AI framework
- **ElevenLabs**: High-quality text-to-speech API
- **Groq**: Fast LLM inference service
- **Google Speech Recognition**: Speech-to-text capabilities

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review RASA documentation

---
