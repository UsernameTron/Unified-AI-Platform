# API Keys Setup Guide

## How to get your API keys for the additional services:

### 🎤 ElevenLabs (Text-to-Speech)
1. Go to [ElevenLabs.io](https://elevenlabs.io/)
2. Sign up for an account (free tier available)
3. Navigate to your Profile → API Keys
4. Create a new API key
5. Copy the API key and replace `your_elevenlabs_api_key_here` in your .env file
6. For voice ID:
   - Go to Voice Library in your ElevenLabs dashboard
   - Choose a voice you like
   - Copy the Voice ID and replace `your_preferred_voice_id_here`

### 🤗 Hugging Face (Open Source Models)
1. Go to [HuggingFace.co](https://huggingface.co/)
2. Sign up for an account (free)
3. Go to Settings → Access Tokens
4. Create a new token with "Read" permissions
5. Copy the token and replace `your_huggingface_token_here` in your .env file
   - Variable name: `HUGGINGFACE_API_KEY=your_actual_token_here`

### 📝 Current .env Status:
- ✅ OpenAI: Configured
- ✅ Anthropic: Configured  
- ✅ Google Gemini: Configured
- ✅ Ollama: Configured (local)
- ⏳ ElevenLabs: Ready for API key
- ⏳ Hugging Face: Ready for API token

### 🚀 Once you add the keys:
Your system will support:
- **Text Generation**: OpenAI, Anthropic, Gemini, Ollama, Hugging Face
- **Text-to-Speech**: ElevenLabs, OpenAI TTS
- **Local Processing**: Ollama
- **Open Source Models**: Hugging Face

### 🧪 Testing:
After adding the keys, run:
```bash
python test_config_status.py
```

This will validate all your configured providers!
