"""
Advanced Sentiment Analysis Agent for Unified AI Platform

This module provides comprehensive emotion analysis, personality trait inference,
and communication style assessment using multiple Hugging Face models with
intelligent fallback to LLM providers.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import time

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Will use LLM fallback only.")

from ..core.agent_factory import AgentBase, AgentResponse
from ..config.multi_provider_config import AIProviderManager


@dataclass
class EmotionScores:
    """Detailed emotion analysis scores"""
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0
    trust: float = 0.0
    anticipation: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @property
    def dominant_emotion(self) -> str:
        """Return the emotion with the highest score"""
        emotions = self.to_dict()
        return max(emotions.items(), key=lambda x: x[1])[0]


@dataclass
class PersonalityTraits:
    """Big Five personality trait scores"""
    openness: float = 0.0
    conscientiousness: float = 0.0
    extraversion: float = 0.0
    agreeableness: float = 0.0
    neuroticism: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class CommunicationStyle:
    """Communication style characteristics"""
    formality: float = 0.0  # 0=informal, 1=formal
    assertiveness: float = 0.0  # 0=passive, 1=assertive
    analytical: float = 0.0  # 0=emotional, 1=analytical
    directness: float = 0.0  # 0=indirect, 1=direct
    enthusiasm: float = 0.0  # 0=reserved, 1=enthusiastic
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SentimentAnalysisResult:
    """Comprehensive sentiment analysis result"""
    # Basic sentiment
    sentiment: str  # positive, negative, neutral
    confidence: float
    
    # Advanced emotion analysis
    emotions: EmotionScores
    
    # Personality traits
    personality: PersonalityTraits
    
    # Communication style
    communication_style: CommunicationStyle
    
    # Metadata
    text_length: int
    processing_time: float
    timestamp: str
    models_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'emotions': self.emotions.to_dict(),
            'personality': self.personality.to_dict(),
            'communication_style': self.communication_style.to_dict(),
            'text_length': self.text_length,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp,
            'models_used': self.models_used
        }


class AdvancedSentimentAgent(AgentBase):
    """
    Advanced sentiment analysis agent with multi-layered emotion analysis,
    personality trait inference, and communication style assessment.
    
    Features:
    - Multi-model emotion detection using Hugging Face transformers
    - Big Five personality trait inference
    - Communication style analysis
    - Intelligent fallback to LLM providers
    - Caching and optimization for real-time analysis
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Advanced Sentiment Agent.
        
        Args:
            config_path: Optional path to configuration file (not used with AIProviderManager)
        """
        # Initialize provider manager
        self.provider_manager = AIProviderManager()
        
        # Initialize parent with required parameters
        super().__init__(
            name="Advanced Sentiment Agent",
            agent_type="advanced_sentiment_analysis",
            config=self.provider_manager.get_config_summary()
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.emotion_models = [
            "j-hartmann/emotion-english-distilroberta-base",
            "SamLowe/roberta-base-go_emotions",
            "cardiffnlp/twitter-roberta-base-emotion"
        ]
        
        self.sentiment_models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment"
        ]
        
        # Loaded models cache
        self._loaded_models = {}
        self._model_load_times = {}
        
        # Initialize models if transformers is available
        if TRANSFORMERS_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize and cache Hugging Face models"""
        try:
            # Load primary emotion model
            primary_emotion_model = self.emotion_models[0]
            self.logger.info(f"Loading primary emotion model: {primary_emotion_model}")
            
            emotion_pipeline = pipeline(
                "text-classification",
                model=primary_emotion_model,
                tokenizer=primary_emotion_model,
                device=0 if torch.cuda.is_available() else -1
            )
            self._loaded_models['emotion'] = emotion_pipeline
            
            # Load primary sentiment model
            primary_sentiment_model = self.sentiment_models[0]
            self.logger.info(f"Loading primary sentiment model: {primary_sentiment_model}")
            
            sentiment_pipeline = pipeline(
                "text-classification",
                model=primary_sentiment_model,
                tokenizer=primary_sentiment_model,
                device=0 if torch.cuda.is_available() else -1
            )
            self._loaded_models['sentiment'] = sentiment_pipeline
            
            self.logger.info("Successfully initialized Hugging Face models")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            self.logger.info("Will use LLM fallback for all analysis")
    
    async def analyze_sentiment(self, text: str, include_personality: bool = True, 
                               include_communication_style: bool = True) -> SentimentAnalysisResult:
        """
        Perform comprehensive sentiment analysis on the given text.
        
        Args:
            text: Text to analyze
            include_personality: Whether to include personality trait analysis
            include_communication_style: Whether to include communication style analysis
            
        Returns:
            SentimentAnalysisResult with comprehensive analysis
        """
        start_time = time.time()
        models_used = []
        
        try:
            # Basic sentiment analysis
            sentiment, confidence = await self._analyze_basic_sentiment(text)
            models_used.append("sentiment_model")
            
            # Advanced emotion analysis
            emotions = await self._analyze_emotions(text)
            models_used.append("emotion_model")
            
            # Personality trait analysis (if requested)
            personality = PersonalityTraits()
            if include_personality:
                personality = await self._analyze_personality(text)
                models_used.append("personality_analysis")
            
            # Communication style analysis (if requested)
            communication_style = CommunicationStyle()
            if include_communication_style:
                communication_style = await self._analyze_communication_style(text)
                models_used.append("communication_analysis")
            
            processing_time = time.time() - start_time
            
            return SentimentAnalysisResult(
                sentiment=sentiment,
                confidence=confidence,
                emotions=emotions,
                personality=personality,
                communication_style=communication_style,
                text_length=len(text),
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
                models_used=models_used
            )
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            # Return basic fallback result
            return await self._fallback_analysis(text, start_time)
    
    async def _analyze_basic_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze basic sentiment (positive/negative/neutral)"""
        if TRANSFORMERS_AVAILABLE and 'sentiment' in self._loaded_models:
            try:
                result = self._loaded_models['sentiment'](text)[0]
                label = result['label'].lower()
                confidence = result['score']
                
                # Map model labels to standard sentiment
                if 'pos' in label or label == 'label_2':
                    sentiment = 'positive'
                elif 'neg' in label or label == 'label_0':
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return sentiment, confidence
                
            except Exception as e:
                self.logger.error(f"Error in HF sentiment analysis: {e}")
        
        # Fallback to LLM analysis
        return await self._llm_sentiment_analysis(text)
    
    async def _analyze_emotions(self, text: str) -> EmotionScores:
        """Analyze detailed emotions using Hugging Face models"""
        if TRANSFORMERS_AVAILABLE and 'emotion' in self._loaded_models:
            try:
                results = self._loaded_models['emotion'](text)
                
                # Initialize emotion scores
                emotions = EmotionScores()
                
                # Map model outputs to our emotion categories
                emotion_mapping = {
                    'joy': ['joy', 'happiness', 'excitement'],
                    'sadness': ['sadness', 'disappointment', 'grief'],
                    'anger': ['anger', 'annoyance', 'rage'],
                    'fear': ['fear', 'nervousness', 'anxiety'],
                    'surprise': ['surprise', 'curiosity', 'realization'],
                    'disgust': ['disgust', 'disapproval'],
                    'trust': ['approval', 'admiration', 'gratitude'],
                    'anticipation': ['optimism', 'desire', 'excitement']
                }
                
                # Process results and map to our categories
                for result in results:
                    label = result['label'].lower()
                    score = result['score']
                    
                    for emotion, keywords in emotion_mapping.items():
                        if any(keyword in label for keyword in keywords):
                            current_score = getattr(emotions, emotion)
                            setattr(emotions, emotion, max(current_score, score))
                            break
                
                return emotions
                
            except Exception as e:
                self.logger.error(f"Error in HF emotion analysis: {e}")
        
        # Fallback to LLM analysis
        return await self._llm_emotion_analysis(text)
    
    async def _analyze_personality(self, text: str) -> PersonalityTraits:
        """Analyze Big Five personality traits from text"""
        # This is primarily done through LLM analysis as there are fewer
        # specialized models for personality trait detection
        return await self._llm_personality_analysis(text)
    
    async def _analyze_communication_style(self, text: str) -> CommunicationStyle:
        """Analyze communication style characteristics"""
        return await self._llm_communication_analysis(text)
    
    async def _llm_sentiment_analysis(self, text: str) -> Tuple[str, float]:
        """Fallback sentiment analysis using LLM"""
        try:
            provider = self.provider_manager.get_provider_for_task('analysis')
            
            prompt = f"""
            Analyze the sentiment of the following text and respond with ONLY a JSON object:
            
            Text: "{text}"
            
            Response format:
            {{
                "sentiment": "positive|negative|neutral",
                "confidence": 0.85
            }}
            """
            
            response = await self._call_provider_api(provider, prompt)
            result = json.loads(response.strip())
            
            return result['sentiment'], result['confidence']
            
        except Exception as e:
            self.logger.error(f"Error in LLM sentiment analysis: {e}")
            return 'neutral', 0.5
    
    async def _llm_emotion_analysis(self, text: str) -> EmotionScores:
        """Fallback emotion analysis using LLM"""
        try:
            provider = self.provider_manager.get_provider_for_task('analysis')
            
            prompt = f"""
            Analyze the emotional content of the following text using Plutchik's 8 basic emotions.
            Rate each emotion from 0.0 to 1.0 and respond with ONLY a JSON object:
            
            Text: "{text}"
            
            Response format:
            {{
                "joy": 0.3,
                "sadness": 0.1,
                "anger": 0.0,
                "fear": 0.2,
                "surprise": 0.4,
                "disgust": 0.0,
                "trust": 0.6,
                "anticipation": 0.5
            }}
            """
            
            response = await self._call_provider_api(provider, prompt)
            result = json.loads(response.strip())
            
            return EmotionScores(**result)
            
        except Exception as e:
            self.logger.error(f"Error in LLM emotion analysis: {e}")
            return EmotionScores()
    
    async def _llm_personality_analysis(self, text: str) -> PersonalityTraits:
        """Analyze personality traits using LLM"""
        try:
            provider = self.provider_manager.get_provider_for_task('analysis')
            
            prompt = f"""
            Analyze the Big Five personality traits evident in the following text.
            Rate each trait from 0.0 to 1.0 and respond with ONLY a JSON object:
            
            Text: "{text}"
            
            Traits:
            - Openness: creativity, curiosity, openness to experience
            - Conscientiousness: organization, responsibility, discipline
            - Extraversion: sociability, assertiveness, energy
            - Agreeableness: cooperation, trust, empathy
            - Neuroticism: emotional instability, anxiety, moodiness
            
            Response format:
            {{
                "openness": 0.6,
                "conscientiousness": 0.7,
                "extraversion": 0.4,
                "agreeableness": 0.8,
                "neuroticism": 0.2
            }}
            """
            
            response = await self._call_provider_api(provider, prompt)
            result = json.loads(response.strip())
            
            return PersonalityTraits(**result)
            
        except Exception as e:
            self.logger.error(f"Error in LLM personality analysis: {e}")
            return PersonalityTraits()
    
    async def _llm_communication_analysis(self, text: str) -> CommunicationStyle:
        """Analyze communication style using LLM"""
        try:
            provider = self.provider_manager.get_provider_for_task('analysis')
            
            prompt = f"""
            Analyze the communication style of the following text.
            Rate each characteristic from 0.0 to 1.0 and respond with ONLY a JSON object:
            
            Text: "{text}"
            
            Characteristics:
            - Formality: 0=informal/casual, 1=formal/professional
            - Assertiveness: 0=passive/tentative, 1=assertive/confident
            - Analytical: 0=emotional/intuitive, 1=logical/analytical
            - Directness: 0=indirect/subtle, 1=direct/straightforward
            - Enthusiasm: 0=reserved/calm, 1=enthusiastic/energetic
            
            Response format:
            {{
                "formality": 0.7,
                "assertiveness": 0.6,
                "analytical": 0.8,
                "directness": 0.5,
                "enthusiasm": 0.4
            }}
            """
            
            response = await self._call_provider_api(provider, prompt)
            result = json.loads(response.strip())
            
            return CommunicationStyle(**result)
            
        except Exception as e:
            self.logger.error(f"Error in LLM communication analysis: {e}")
            return CommunicationStyle()
    
    async def _fallback_analysis(self, text: str, start_time: float) -> SentimentAnalysisResult:
        """Provide basic fallback analysis when all methods fail"""
        processing_time = time.time() - start_time
        
        return SentimentAnalysisResult(
            sentiment='neutral',
            confidence=0.5,
            emotions=EmotionScores(),
            personality=PersonalityTraits(),
            communication_style=CommunicationStyle(),
            text_length=len(text),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            models_used=['fallback']
        )
    
    async def batch_analyze(self, texts: List[str], **kwargs) -> List[SentimentAnalysisResult]:
        """
        Analyze multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to analyze
            **kwargs: Additional arguments passed to analyze_sentiment
            
        Returns:
            List of SentimentAnalysisResult objects
        """
        tasks = [self.analyze_sentiment(text, **kwargs) for text in texts]
        return await asyncio.gather(*tasks)
    
    async def execute(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Execute sentiment analysis on provided text.
        
        Args:
            input_data: Dictionary containing 'text' and optional parameters
            
        Returns:
            AgentResponse with sentiment analysis results
        """
        start_time = time.time()
        try:
            # Validate input
            if 'text' not in input_data:
                raise ValueError("Missing required 'text' field in input data")
            
            text = input_data['text']
            include_personality = input_data.get('include_personality', True)
            include_communication_style = input_data.get('include_communication_style', True)
            
            # Perform analysis
            result = await self.analyze_sentiment(
                text=text,
                include_personality=include_personality,
                include_communication_style=include_communication_style
            )
            
            execution_time = time.time() - start_time
            return AgentResponse(
                success=True,
                result=result.to_dict(),
                agent_type=self.agent_type,
                timestamp=datetime.now().isoformat(),
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error executing sentiment analysis: {e}")
            execution_time = time.time() - start_time
            return AgentResponse(
                success=False,
                result=None,
                agent_type=self.agent_type,
                timestamp=datetime.now().isoformat(),
                execution_time=execution_time,
                error=str(e)
            )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models and capabilities"""
        return {
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'loaded_models': list(self._loaded_models.keys()),
            'available_emotion_models': self.emotion_models,
            'available_sentiment_models': self.sentiment_models,
            'cuda_available': torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False,
            'providers_available': self.provider_manager.get_enabled_providers()
        }
    
    async def _call_provider_api(self, provider_name: str, prompt: str) -> str:
        """
        Make an API call to the specified provider.
        
        Args:
            provider_name: Name of the provider (openai, anthropic, etc.)
            prompt: Prompt to send to the provider
            
        Returns:
            Response text from the provider
        """
        try:
            config = self.provider_manager.get_provider_config(provider_name)
            if not config:
                raise ValueError(f"No configuration found for provider: {provider_name}")
            
            if provider_name == 'openai':
                import openai
                client = openai.OpenAI(api_key=config.api_key)
                response = client.chat.completions.create(
                    model=config.model or 'gpt-4',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.max_tokens or 1000,
                    temperature=0.1
                )
                return response.choices[0].message.content or ""
                
            elif provider_name == 'anthropic':
                # For now, use a simplified implementation
                # TODO: Implement proper Anthropic API integration
                raise NotImplementedError("Anthropic provider API needs proper implementation")
                
            elif provider_name == 'gemini':
                # Simplified approach for now - will need proper google ai implementation
                raise NotImplementedError("Gemini provider API not fully implemented")
                
            elif provider_name == 'ollama':
                import requests
                response = requests.post(
                    f"{config.base_url or 'http://localhost:11434'}/api/generate",
                    json={
                        "model": config.model or 'phi3.5',
                        "prompt": prompt,
                        "stream": False
                    }
                )
                return response.json()['response']
                
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")
                
        except Exception as e:
            self.logger.error(f"Error calling {provider_name} API: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    async def test_advanced_sentiment():
        agent = AdvancedSentimentAgent()
        
        test_texts = [
            "I'm absolutely thrilled about this new opportunity! It's going to be amazing.",
            "I'm feeling quite anxious about the upcoming presentation tomorrow.",
            "The analysis shows clear patterns in the data that support our hypothesis.",
            "Hey there! Hope you're having a fantastic day! 😊"
        ]
        
        print("Testing Advanced Sentiment Analysis:")
        print("=" * 50)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text}")
            print("-" * 30)
            
            result = await agent.analyze_sentiment(text)
            
            print(f"Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
            print(f"Dominant emotion: {result.emotions.dominant_emotion}")
            print(f"Processing time: {result.processing_time:.3f}s")
            
            # Print top emotions
            emotions = result.emotions.to_dict()
            top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Top emotions: {', '.join([f'{e}: {s:.2f}' for e, s in top_emotions])}")
        
        # Test model info
        print(f"\nModel Information:")
        print("-" * 30)
        model_info = agent.get_model_info()
        for key, value in model_info.items():
            print(f"{key}: {value}")
    
    # Run the test
    asyncio.run(test_advanced_sentiment())
