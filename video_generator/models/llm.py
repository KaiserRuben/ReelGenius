
import os
import time
import json
import requests
from typing import Dict, List, Any, Optional, Union, Callable
from loguru import logger
from pydantic import BaseModel, Field
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import asyncio
from jinja2 import Template

class DeepSeekChatMessage(BaseModel):
    """DeepSeek chat message structure."""
    role: str
    content: str

class DeepSeekLLM(LLM):
    """Custom LLM class for DeepSeek API."""
    
    api_key: str
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    retries: int = 3
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the DeepSeek API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if stop:
            data["stop"] = stop
            
        # Override with any kwargs
        for key, value in kwargs.items():
            if key in data:
                data[key] = value
        
        # Retry mechanism
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except Exception as e:
                logger.error(f"DeepSeek API call failed (attempt {attempt+1}/{self.retries}): {str(e)}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise ValueError(f"Failed to call DeepSeek API after {self.retries} attempts: {str(e)}")
    
    def structured_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate structured output according to schema."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        schema_str = json.dumps(output_schema, indent=2)
        full_prompt = f"{prompt}\n\nPlease provide your response in the following JSON format:\n{schema_str}"
        messages.append({"role": "user", "content": full_prompt})
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        # Override with any kwargs
        for key, value in kwargs.items():
            if key in data:
                data[key] = value
        
        # Retry mechanism
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse the JSON response
                return json.loads(content)
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse DeepSeek response as JSON")
                if attempt < self.retries - 1:
                    # Try with stricter JSON enforcement
                    data["response_format"] = {"type": "json_object"}
                    time.sleep(2 ** attempt)
                else:
                    raise ValueError("Failed to get valid JSON from DeepSeek API")
                    
            except Exception as e:
                logger.error(f"DeepSeek API call failed (attempt {attempt+1}/{self.retries}): {str(e)}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise ValueError(f"Failed to call DeepSeek API after {self.retries} attempts: {str(e)}")

class DeepSeekChatModel(BaseChatModel):
    """Custom Chat Model class for DeepSeek API."""
    
    api_key: str
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60
    retries: int = 3
    
    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""
        deepseek_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                deepseek_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                deepseek_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                deepseek_messages.append({"role": "assistant", "content": message.content})
            else:
                logger.warning(f"Unsupported message type: {type(message)}. Using role 'user'.")
                deepseek_messages.append({"role": "user", "content": str(message.content)})
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": deepseek_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if stop:
            data["stop"] = stop
            
        # Override with any kwargs
        for key, value in kwargs.items():
            if key in data:
                data[key] = value
        
        # Retry mechanism
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                message_content = result["choices"][0]["message"]["content"]
                
                chat_generation = ChatGeneration(
                    message=AIMessage(content=message_content)
                )
                
                return ChatResult(generations=[chat_generation])
                
            except Exception as e:
                logger.error(f"DeepSeek API call failed (attempt {attempt+1}/{self.retries}): {str(e)}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise ValueError(f"Failed to call DeepSeek API after {self.retries} attempts: {str(e)}")
    
    def _generate_with_structured_output(
        self,
        messages: List[BaseMessage],
        output_schema: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate with structured output."""
        deepseek_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                deepseek_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                deepseek_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                deepseek_messages.append({"role": "assistant", "content": message.content})
            else:
                deepseek_messages.append({"role": "user", "content": str(message.content)})
        
        # Add schema information to the last message
        if deepseek_messages:
            schema_str = json.dumps(output_schema, indent=2)
            last_message = deepseek_messages[-1]
            last_message["content"] = f"{last_message['content']}\n\nPlease format your response as a JSON object with the following schema:\n{schema_str}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": deepseek_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        # Override with any kwargs
        for key, value in kwargs.items():
            if key in data:
                data[key] = value
        
        # Retry mechanism
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse the JSON response
                return json.loads(content)
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse DeepSeek response as JSON")
                if attempt < self.retries - 1:
                    # Try with stricter enforcement
                    data["response_format"] = {"type": "json_object"}
                    time.sleep(2 ** attempt)
                else:
                    raise ValueError("Failed to get valid JSON from DeepSeek API")
                    
            except Exception as e:
                logger.error(f"DeepSeek API call failed (attempt {attempt+1}/{self.retries}): {str(e)}")
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise ValueError(f"Failed to call DeepSeek API after {self.retries} attempts: {str(e)}")
    
    def structured_output(
        self,
        messages: List[BaseMessage],
        output_schema: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate output according to schema."""
        return self._generate_with_structured_output(messages, output_schema, **kwargs)

class PromptTemplateManager:
    """Manages prompt templates for the pipeline."""
    
    def __init__(self, template_dir: Optional[str] = None):
        from pathlib import Path
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent.parent / "templates"
        self.templates_cache = {}
        
        # Create template directory if it doesn't exist
        self.template_dir.mkdir(exist_ok=True, parents=True)
        
        # Default templates
        self._ensure_default_templates()
    
    def _ensure_default_templates(self):
        """Ensure default templates exist."""
        default_templates = {
            "meta_prompt.j2": """
            You are a prompt engineering expert. Your task is to optimize a prompt for {{purpose}}.
            
            The prompt will be used with the {{model}} model to generate {{output_type}}.
            
            Based on best practices for prompt engineering, enhance the following base prompt:
            
            BASE PROMPT:
            {{base_prompt}}
            
            Improve this prompt by:
            1. Adding specific instructions for high-quality outputs
            2. Including constraints to avoid common pitfalls
            3. Structuring it clearly with step-by-step guidance
            4. Adding few-shot examples where helpful
            5. Incorporating chain-of-thought reasoning
            
            Return ONLY the improved prompt without explanation.
            """,
            
            "content_analysis.j2": """
            You are an expert content analyzer for {{platform}} video creation. Your task is to analyze the provided content
            and determine its type, topics, complexity, target audience, sentiment, and key points.
            
            Be extremely precise in your analysis as this will guide the entire video creation process.
            
            {% if llm.chain_of_thought %}
            Think step-by-step:
            1. First, identify what type of content this is (scientific, educational, entertainment, etc.)
            2. Then, extract the main topics covered
            3. Assess the complexity level and appropriate target audience
            4. Analyze the sentiment and emotional tone
            5. Extract the key points that must be communicated
            {% endif %}
            
            {% if llm.few_shot_examples %}
            Example analysis:
            INPUT: "Quantum computing uses quantum bits or qubits to perform computations. Unlike classical bits..."
            OUTPUT: {
                "type": "scientific_educational",
                "topics": ["quantum computing", "qubits", "computational theory"],
                "complexity": "medium",
                "target_audience": "tech enthusiasts with basic physics knowledge",
                "sentiment": "neutral informative",
                "key_points": ["quantum bits vs classical bits", "fundamentals of quantum computing", "applications"]
            }
            {% endif %}
            
            Analyze the following content for {{platform}} video creation:
            {{content}}
            """,
            
            "content_strategy.j2": """
            You are a {{platform}} content strategy expert. Your task is to develop an effective content strategy
            that maximizes engagement, views, and follows.
            
            For the {{platform}} platform, focus specifically on:
            1. Hook creation (capturing attention in the first {{platform_config.hook_duration}} seconds)
            2. Optimal video pacing for the {{platform_config.audience_attention_span}}s attention span
            3. Trending elements that boost visibility in the algorithm
            4. Effective hashtag strategy specific to this content
            5. Call-to-action approaches that drive engagement
            
            {% if llm.chain_of_thought %}
            Think step-by-step about creating an engaging {{platform}} video:
            1. What hook type will best capture attention? (question, surprising fact, bold statement)
            2. How should we structure the narrative to maintain viewer interest?
            3. What trending elements could boost visibility?
            4. What unique angle differentiates this from similar content?
            5. What hashtags would reach the relevant audience?
            6. What optimal video length balances completeness and engagement?
            {% endif %}
            
            {% if llm.few_shot_examples %}
            Example strategy for scientific content:
            INPUT: {
                "type": "scientific_educational", 
                "topics": ["quantum computing"]
            }
            OUTPUT: {
                "hook_type": "surprising_fact",
                "hook_content": "A quantum computer could break all internet encryption in seconds",
                "narrative_style": "simplified_analogies",
                "engagement_approach": "mind_blowing_comparisons",
                "trending_elements": ["futuristic_technology", "science_mysteries"],
                "hashtags": ["quantumcomputing", "techfuture", "scienceexplained"],
                "optimal_length": 45,
                "unique_angle": "everyday_implications"
            }
            {% endif %}
            
            Based on this input analysis, develop a {{platform}} content strategy:
            {{input_analysis}}
            """,
            
            "script_generation.j2": """
            You are an expert {{platform}} scriptwriter. Your task is to create an engaging script optimized for {{platform}}'s format and algorithm.
            
            Key requirements for {{platform}} scripts:
            1. Create a strong hook in the first {{platform_config.hook_duration}} seconds
            2. Structure in segments of {{platform_config.audience_attention_span}} seconds to maintain attention
            3. Include precise visual directions for each segment
            4. End with a strong call to action
            5. Keep total duration between {{platform_config.min_duration}}-{{platform_config.max_duration}} seconds
            6. Identify keywords for visual emphasis
            
            {% if llm.chain_of_thought %}
            Think step-by-step about creating an effective script:
            1. Design a hook that immediately captures attention
            2. Break content into segments matching attention span
            3. For each segment, craft concise narration with visual direction
            4. Create a call-to-action that encourages engagement
            5. Check total duration for platform requirements
            6. Mark keywords that should be emphasized visually
            {% endif %}
            
            {% if llm.few_shot_examples %}
            Example script for 45-second science video:
            {
                "hook": "What if I told you computers could be in two states at once?",
                "segments": [
                    {
                        "text": "Unlike regular computers that use bits - either 0 or 1 - quantum computers use qubits that exist in multiple states simultaneously.",
                        "duration": 8,
                        "visual_direction": "Animation showing classical bit (binary) transforming into quantum bit (superposition)"
                    },
                    {
                        "text": "This property called 'superposition' gives quantum computers exponential power compared to classical machines.",
                        "duration": 7,
                        "visual_direction": "Visual comparison showing classical vs quantum processing power with exponential growth chart"
                    }
                ],
                "call_to_action": "Follow for more mind-blowing science explained simply!",
                "total_duration": 45,
                "keywords_to_emphasize": ["superposition", "qubits", "exponential power"]
            }
            {% endif %}
            
            Create a {{platform}} script based on the following:
            
            ORIGINAL CONTENT:
            {{original_content}}
            
            INPUT ANALYSIS:
            {{input_analysis}}
            
            CONTENT STRATEGY:
            {{content_strategy}}
            """,
            
            "script_quality_check.j2": """
            You are a {{platform}} script quality analyst. Your task is to evaluate the quality of a script
            and provide a score and improvement suggestions.
            
            Evaluate the script on:
            1. Hook strength - Does it grab attention immediately?
            2. Clarity - Is the message clear and concise?
            3. Engagement - Does it encourage viewer interaction?
            4. Platform fit - Is it optimized for {{platform}} specifically?
            5. Call to action - Is there a strong CTA?
            
            {% if llm.chain_of_thought %}
            Think step-by-step about evaluating this script:
            1. Assess the hook - Does it capture attention in {{platform_config.hook_duration}} seconds?
            2. Evaluate each segment for clarity and conciseness
            3. Check if the script maintains engagement throughout
            4. Verify it follows {{platform}} best practices
            5. Evaluate the effectiveness of the call-to-action
            6. Identify specific improvements needed
            {% endif %}
            
            Provide a detailed assessment with specific recommendations for improvement.
            
            SCRIPT:
            {{script}}
            
            CONTENT STRATEGY:
            {{content_strategy}}
            """,
            
            "visual_planning.j2": """
            You are a {{platform}} visual design expert. Your task is to create detailed visual plans that maximize
            engagement and viewer retention.
            
            For each script segment, create:
            1. A detailed image generation prompt in {{platform_config.aspect_ratio}} aspect ratio
            2. Text overlays that emphasize key points
            3. Visual effects and transitions that enhance the narrative
            
            {% if llm.chain_of_thought %}
            Think step-by-step about planning visuals:
            1. For each segment, what primary visual would best represent the content?
            2. What text elements should be overlaid to emphasize key points?
            3. How should text be positioned for maximum impact?
            4. What visual effects would enhance engagement?
            5. What transitions would create cohesion between segments?
            6. How can we maintain visual consistency throughout?
            {% endif %}
            
            {% if visual.visual_consistency %}
            Ensure visual consistency with these style guidelines:
            - Color scheme: {{visual.color_scheme}}
            - Text animation: {{visual.text_animation}}
            - Transition style: {{visual.transition_style}}
            {% endif %}
            
            {% if llm.few_shot_examples %}
            Example visual plan for quantum computing segment:
            {
                "scenes": [
                    {
                        "segment_index": 0,
                        "duration": 8,
                        "image_prompt": "Digital art showing classical computer bit transforming into quantum qubit, blue digital background with binary code morphing into quantum probability cloud, vertical 9:16 format, professional lighting, cinematic quality",
                        "text_overlay": "QUANTUM BITS: BEYOND 0 AND 1",
                        "text_position": "bottom",
                        "effect": "fade_in",
                        "transition": "dissolve"
                    }
                ],
                "style_consistency": "futuristic tech aesthetic with blue and purple tones",
                "color_palette": ["#3498db", "#9b59b6", "#1abc9c", "#34495e"],
                "text_style": "modern sans-serif with glow effect"
            }
            {% endif %}
            
            Create a visual plan for this {{platform}} script:
            
            SCRIPT:
            {{script}}
            
            INPUT ANALYSIS:
            {{input_analysis}}
            
            CONTENT STRATEGY:
            {{content_strategy}}
            """,
            
            "image_prompt.j2": """
            You are an expert AI image prompt engineer. Your task is to create detailed, effective prompts for {{image_gen.provider}} that will generate high-quality images for a {{platform}} video.
            
            Create a prompt that will produce a {{platform_config.aspect_ratio}} image for {{platform}} that visualizes:
            {{base_prompt}}
            
            {% if llm.chain_of_thought %}
            Think step-by-step about what makes an effective image prompt:
            1. What key elements must be included?
            2. What style words will guide the AI?
            3. What composition details ensure good vertical framing?
            4. What technical quality markers should be included?
            5. What mood/atmosphere words enhance the feeling?
            {% endif %}
            
            {% if image_gen.provider == "stability" %}
            Optimize specifically for Stability AI by:
            - Using descriptive adjectives for style (e.g., "photorealistic", "cinematic", "vibrant")
            - Specifying lighting details (e.g., "dramatic lighting", "soft natural light")
            - Including technical quality markers (e.g., "detailed", "high resolution")
            - Adding artistic style references where appropriate
            - Ensuring proper vertical (9:16) composition
            {% endif %}
            
            Write a single, detailed prompt paragraph without using hyphens or bullet points.
            """,
            
            "metadata_generation.j2": """
            You are a {{platform}} metadata optimization expert. Your task is to create engaging, algorithm-friendly metadata.
            
            Create the following metadata elements:
            1. Attention-grabbing title (max 100 characters)
            2. Engaging description with hooks and keywords
            3. Relevant hashtags (5-10)
            4. Category suggestions
            
            {% if llm.chain_of_thought %}
            Think step-by-step about creating effective metadata:
            1. What title would capture attention and include keywords?
            2. What description balances hooks, value proposition, and calls-to-action?
            3. What hashtags would maximize discoverability?
            4. What category best fits this content?
            {% endif %}
            
            {% if llm.few_shot_examples %}
            Example metadata for science video:
            {
                "title": "Quantum Computers Will Break The Internet 🤯 Here's How",
                "description": "Your passwords won't be safe when quantum computers arrive! Learn how these mind-bending machines work and why they'll change everything. #QuantumComputing #TechFuture",
                "hashtags": ["quantumcomputing", "techfuture", "scienceexplained", "futuretech", "computerscience"],
                "category": "Science & Technology"
            }
            {% endif %}
            
            Generate metadata based on:
            
            VIDEO CONTENT:
            {{script}}
            
            CONTENT STRATEGY:
            {{content_strategy}}
            """
        }
        
        # Write default templates if they don't exist
        for filename, content in default_templates.items():
            template_path = self.template_dir / filename
            if not template_path.exists():
                with open(template_path, 'w') as f:
                    f.write(content.strip())
    
    def get_template(self, template_name: str) -> Template:
        """Get a template by name."""
        if template_name in self.templates_cache:
            return self.templates_cache[template_name]
        
        template_path = self.template_dir / f"{template_name}.j2"
        if not template_path.exists():
            raise ValueError(f"Template {template_name} not found")
        
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        template = Template(template_content)
        self.templates_cache[template_name] = template
        return template
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """Render a template with the given arguments."""
        template = self.get_template(template_name)
        return template.render(**kwargs)
    
    def meta_prompt(self, base_prompt: str, purpose: str, model: str, output_type: str) -> str:
        """Generate a meta-prompt to improve a base prompt."""
        meta_template = self.get_template("meta_prompt")
        return meta_template.render(
            base_prompt=base_prompt,
            purpose=purpose,
            model=model,
            output_type=output_type
        )
    
    def optimize_prompt(self, prompt_text: str, purpose: str, config: Dict[str, Any]) -> str:
        """Use meta-prompting to optimize a prompt if enabled."""
        if not config.get("llm", {}).get("use_meta_prompting", True):
            return prompt_text
        
        # Call LLM to optimize the prompt
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['llm']['api_key']}"
        }
        
        meta_prompt_template = self.get_template("meta_prompt")
        meta_prompt = meta_prompt_template.render(
            base_prompt=prompt_text,
            purpose=purpose,
            model=config['llm']['model'],
            output_type=purpose
        )
        
        data = {
            "model": config['llm']['model'],
            "messages": [{"role": "user", "content": meta_prompt}],
            "temperature": 0.7,
        }
        
        try:
            response = requests.post(
                f"{config['llm']['base_url']}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            optimized_prompt = result["choices"][0]["message"]["content"]
            
            return optimized_prompt
        except Exception as e:
            logger.error(f"Meta-prompting failed: {e}")
            return prompt_text  # Return original prompt if optimization fails


