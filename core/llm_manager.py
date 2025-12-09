# core/llm_manager.py
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
import time
import os

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    GROQ = "groq"
    CLAUDE = "claude"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"        # Basic queries, summaries
    MEDIUM = "medium"        # Creative writing, analysis
    COMPLEX = "complex"      # Deep reasoning, multi-step
    EXPERT = "expert"        # Critical decisions, research


@dataclass
class LLMConfig:
    """Configuration for a specific LLM backend"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Performance characteristics
    cost_per_1k_tokens: float = 0.0  # USD
    avg_latency_ms: int = 1000
    max_tokens: int = 4096
    
    # Capabilities
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False
    
    # Quality scores (0-10)
    reasoning_quality: int = 5
    creative_quality: int = 5
    code_quality: int = 5
    speed_score: int = 5
    
    # Availability
    is_local: bool = False
    requires_internet: bool = True
    max_concurrent: int = 10


class LLMManager:
    """
    Manages multiple LLM backends and intelligently routes requests
    """
    
    def __init__(self):
        self.backends: Dict[str, LLMConfig] = {}
        self.clients: Dict[str, Any] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
    
    
    def register_backend(
        self,
        name: str,
        config: LLMConfig,
        initialize: bool = True
    ) -> bool:
        """
        Register an LLM backend
        
        Args:
            name: Unique name for this backend
            config: Backend configuration
            initialize: Whether to initialize the client immediately
            
        Returns:
            True if successful
        """
        try:
            self.backends[name] = config
            
            if initialize:
                client = self._initialize_client(config)
                if client:
                    self.clients[name] = client
                    logger.info(f"Registered LLM backend: {name} ({config.provider.value})")
                    return True
                else:
                    logger.warning(f"Failed to initialize {name}, but registered")
                    return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register backend {name}: {e}")
            return False
    
    
    def _initialize_client(self, config: LLMConfig) -> Optional[Any]:
        """Initialize the appropriate client for a provider"""
        try:
            if config.provider == LLMProvider.OLLAMA:
                import ollama
                return ollama
            
            elif config.provider == LLMProvider.GROQ:
                from groq import Groq
                return Groq(api_key=config.api_key)
            
            elif config.provider == LLMProvider.CLAUDE:
                from anthropic import Anthropic
                return Anthropic(api_key=config.api_key)
            
            elif config.provider == LLMProvider.OPENAI:
                from openai import OpenAI
                return OpenAI(api_key=config.api_key)
            
            elif config.provider == LLMProvider.HUGGINGFACE:
                from huggingface_hub import InferenceClient
                return InferenceClient(token=config.api_key)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to initialize {config.provider.value}: {e}")
            return None
    
    
    def select_backend(
        self,
        task_type: str = "general",
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        prefer_local: bool = False,
        prefer_fast: bool = False,
        max_cost: Optional[float] = None,
        required_features: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Intelligently select the best backend for a task
        (This method is kept for backwards compatibility but complete() now uses _rank_backends)
        """
        candidates = []
        
        for name, config in self.backends.items():
            # Check if client is available
            if name not in self.clients:
                logger.debug(f"Skipping {name} - client not initialized")
                continue
            
            # Check circuit breaker
            if self._is_circuit_open(name):
                logger.debug(f"Skipping {name} - circuit breaker open")
                continue
            
            # Apply filters
            if prefer_local and not config.is_local:
                continue
            
            if max_cost and config.cost_per_1k_tokens > max_cost:
                continue
            
            if required_features:
                if "streaming" in required_features and not config.supports_streaming:
                    continue
                if "functions" in required_features and not config.supports_functions:
                    continue
                if "vision" in required_features and not config.supports_vision:
                    continue
            
            # Calculate score
            score = self._calculate_backend_score(
                config, task_type, complexity, prefer_fast
            )
            
            candidates.append((name, score, config))
        
        if not candidates:
            logger.warning("No suitable backends available")
            return None
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected = candidates[0][0]
        logger.info(f"Selected backend: {selected} (score: {candidates[0][1]:.2f})")
        
        return selected
    
    
    def _calculate_backend_score(
        self,
        config: LLMConfig,
        task_type: str,
        complexity: TaskComplexity,
        prefer_fast: bool
    ) -> float:
        """Calculate UNBIASED score for backend selection"""
        score = 0.0
        
        # 1. Base quality score (most important)
        if task_type == "creative":
            score += config.creative_quality * 3  # Increased weight
        elif task_type == "code":
            score += config.code_quality * 3
        elif task_type == "reasoning" or task_type == "analytical":
            score += config.reasoning_quality * 3
        else:
            score += (config.reasoning_quality + config.creative_quality) / 2 * 3
        
        # 2. Complexity matching
        if complexity == TaskComplexity.SIMPLE:
            score += config.speed_score * 0.5
        elif complexity == TaskComplexity.EXPERT:
            score += config.reasoning_quality * 1.5
        
        # 3. Speed consideration (ALWAYS apply if backend is fast)
        # Read from .env
        prefer_fast = prefer_fast or os.getenv("LLM_PREFER_FAST", "true").lower() == "true"
        
        if prefer_fast:
            # Reward fast backends
            if config.avg_latency_ms < 1000:  # Under 1 second
                score += 10
            elif config.avg_latency_ms < 3000:  # Under 3 seconds
                score += 5
            # Penalize slow backends
            elif config.avg_latency_ms > 8000:  # Over 8 seconds
                score -= 5
        
        # 4. Cost consideration (FAIR penalty)
        if config.cost_per_1k_tokens == 0:
            score += 8  # Bonus for free
        else:
            # Fair cost penalty (not too harsh)
            max_cost = float(os.getenv("LLM_MAX_COST_PER_REQUEST", "0.01"))
            if config.cost_per_1k_tokens <= max_cost:
                score -= config.cost_per_1k_tokens * 100  # Reasonable penalty
            else:
                score -= 50  # Too expensive, big penalty
        
        # 5. Local preference (ONLY if explicitly enabled)
        prefer_local = os.getenv("LLM_PREFER_LOCAL", "false").lower() == "true"
        if prefer_local and config.is_local:
            score += 10  # Only bonus if user wants local
        elif not prefer_local and not config.is_local:
            score += 2  # Small bonus for cloud (more reliable)
        
        # 6. Priority from .env (OVERRIDE scores)
        priority_list = os.getenv("LLM_BACKEND_PRIORITY", "").split(",")
        if priority_list and config.provider.value in priority_list:
            priority_index = priority_list.index(config.provider.value)
            # First in list gets +20, second +10, third +5, etc.
            priority_bonus = max(0, 20 - (priority_index * 10))
            score += priority_bonus
        
        return score
    
    
    async def complete(
        self,
        prompt: str,
        backend: Optional[str] = None,
        task_type: str = "general",
        complexity: TaskComplexity = TaskComplexity.MEDIUM,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion using selected or automatic backend with fallback
        
        Args:
            prompt: Input prompt
            backend: Specific backend to use (or auto-select)
            task_type: Type of task
            complexity: Complexity level
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific arguments
            
        Returns:
            {
                "content": str,
                "backend": str,
                "model": str,
                "provider": str,
                "tokens_used": int,
                "latency_ms": int,
                "cost": float,
                "metadata": dict
            }
        """
        # If specific backend requested, try only that one
        if backend:
            if backend not in self.clients:
                available = [name for name in self.backends.keys() if name in self.clients]
                raise Exception(
                    f"Backend '{backend}' not available. "
                    f"Available backends: {available if available else 'None'}"
                )
            
            # Try the specific backend
            return await self._attempt_completion(
                backend, prompt, max_tokens, temperature, **kwargs
            )
        
        # Auto-select: get ranked list of all available backends
        candidates = self._rank_backends(task_type, complexity, kwargs)
        
        if not candidates:
            available_backends = [name for name in self.backends.keys() if name in self.clients]
            circuit_open = [name for name in available_backends if self._is_circuit_open(name)]
            
            if not available_backends:
                raise Exception(
                    f"No available backends. "
                    f"Registered: {list(self.backends.keys())}, "
                    f"but none have initialized clients. "
                    f"Check your .env configuration and ensure services are running."
                )
            elif circuit_open:
                raise Exception(
                    f"No suitable backends available. "
                    f"Available backends: {available_backends}, "
                    f"but all have circuit breakers open: {circuit_open}. "
                    f"Services may be experiencing issues."
                )
            else:
                raise Exception(
                    f"No backends match criteria. "
                    f"Available: {available_backends}, "
                    f"task_type={task_type}, complexity={complexity}"
                )
        
        # Try each candidate backend until one succeeds
        last_error = None
        
        for backend_name, score, config in candidates:
            try:
                logger.info(f"Attempting backend: {backend_name} (score: {score:.2f})")
                
                result = await self._attempt_completion(
                    backend_name, prompt, max_tokens, temperature, **kwargs
                )
                
                logger.info(f"Successfully used backend: {backend_name}")
                return result
                
            except Exception as e:
                logger.warning(f"Backend {backend_name} failed: {e}")
                last_error = e
                
                # Continue to next backend
                continue
        
        # All backends failed
        tried_backends = [name for name, _, _ in candidates]
        raise Exception(
            f"All {len(tried_backends)} backends failed. "
            f"Tried: {tried_backends}. "
            f"Last error: {last_error}"
        )
    
    
    def _rank_backends(
        self,
        task_type: str,
        complexity: TaskComplexity,
        kwargs: dict
    ) -> List[tuple]:
        """
        Rank all available backends by suitability score
        
        Returns:
            List of (backend_name, score, config) tuples, sorted by score (highest first)
        """
        candidates = []
        
        prefer_local = kwargs.get("prefer_local", False)
        prefer_fast = kwargs.get("prefer_fast", False)
        max_cost = kwargs.get("max_cost", None)
        required_features = kwargs.get("required_features", None)
        
        for name, config in self.backends.items():
            # Check if client is available
            if name not in self.clients:
                logger.debug(f"Skipping {name} - client not initialized")
                continue
            
            # Check circuit breaker
            if self._is_circuit_open(name):
                logger.debug(f"Skipping {name} - circuit breaker open")
                continue
            
            # Apply filters
            if prefer_local and not config.is_local:
                continue
            
            if max_cost and config.cost_per_1k_tokens > max_cost:
                continue
            
            if required_features:
                if "streaming" in required_features and not config.supports_streaming:
                    continue
                if "functions" in required_features and not config.supports_functions:
                    continue
                if "vision" in required_features and not config.supports_vision:
                    continue
            
            # Calculate score
            score = self._calculate_backend_score(
                config, task_type, complexity, prefer_fast
            )
            
            candidates.append((name, score, config))
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
    
    
    async def _attempt_completion(
        self,
        backend: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Attempt completion with a specific backend
        
        Args:
            backend: Backend name to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Completion result dictionary
            
        Raises:
            Exception if the backend fails
        """
        config = self.backends[backend]
        client = self.clients[backend]
        
        # Record start time
        start_time = time.time()
        
        try:
            # Call appropriate backend
            if config.provider == LLMProvider.OLLAMA:
                result = await self._call_ollama(
                    client, config, prompt, max_tokens, temperature, **kwargs
                )
            
            elif config.provider == LLMProvider.GROQ:
                result = await self._call_groq(
                    client, config, prompt, max_tokens, temperature, **kwargs
                )
            
            elif config.provider == LLMProvider.CLAUDE:
                result = await self._call_claude(
                    client, config, prompt, max_tokens, temperature, **kwargs
                )
            
            elif config.provider == LLMProvider.OPENAI:
                result = await self._call_openai(
                    client, config, prompt, max_tokens, temperature, **kwargs
                )
            
            else:
                raise Exception(f"Provider {config.provider.value} not implemented")
            
            # Calculate metrics
            latency_ms = int((time.time() - start_time) * 1000)
            tokens_used = result.get("tokens_used", 0)
            cost = (tokens_used / 1000) * config.cost_per_1k_tokens
            
            # Record success
            self._record_success(backend, latency_ms, tokens_used, cost)
            
            return {
                "content": result["content"],
                "backend": backend,
                "model": config.model,
                "provider": config.provider.value,
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
                "cost": cost,
                "metadata": result.get("metadata", {})
            }
        
        except Exception as e:
            # Record failure
            self._record_failure(backend)
            
            # Re-raise to allow fallback to next backend
            raise Exception(f"Backend {backend} failed: {e}") from e
    
    
    async def _call_ollama(
        self, client, config, prompt, max_tokens, temperature, **kwargs
    ) -> Dict[str, Any]:
        """Call Ollama API"""
        response = await asyncio.to_thread(
            client.chat,
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        
        return {
            "content": response['message']['content'],
            "tokens_used": response.get('eval_count', 0) + response.get('prompt_eval_count', 0)
        }
    
    
    async def _call_groq(
        self, client, config, prompt, max_tokens, temperature, **kwargs
    ) -> Dict[str, Any]:
        """Call Groq API"""
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "content": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens
        }
    
    
    async def _call_claude(
        self, client, config, prompt, max_tokens, temperature, **kwargs
    ) -> Dict[str, Any]:
        """Call Claude API"""
        response = await asyncio.to_thread(
            client.messages.create,
            model=config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "content": response.content[0].text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens
        }
    
    
    async def _call_openai(
        self, client, config, prompt, max_tokens, temperature, **kwargs
    ) -> Dict[str, Any]:
        """Call OpenAI API"""
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "content": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens
        }
    
    
    def _is_circuit_open(self, backend: str) -> bool:
        """Check if circuit breaker is open for a backend"""
        if backend not in self.circuit_breakers:
            return False
        
        cb = self.circuit_breakers[backend]
        
        # Check if enough time has passed to retry
        if time.time() - cb["opened_at"] > cb["timeout"]:
            logger.info(f"Circuit breaker half-open for {backend}, allowing retry")
            return False
        
        return cb["is_open"]
    
    
    def _record_success(self, backend: str, latency_ms: int, tokens: int, cost: float):
        """Record successful request"""
        if backend not in self.usage_stats:
            self.usage_stats[backend] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0
            }
        
        stats = self.usage_stats[backend]
        stats["total_requests"] += 1
        stats["successful_requests"] += 1
        stats["total_tokens"] += tokens
        stats["total_cost"] += cost
        
        # Update rolling average latency
        n = stats["successful_requests"]
        stats["avg_latency_ms"] = (
            stats["avg_latency_ms"] * (n - 1) + latency_ms
        ) / n
        
        # Reset circuit breaker on success
        if backend in self.circuit_breakers:
            self.circuit_breakers[backend]["consecutive_failures"] = 0
            self.circuit_breakers[backend]["is_open"] = False
    
    
    def _record_failure(self, backend: str):
        """Record failed request and manage circuit breaker"""
        if backend not in self.usage_stats:
            self.usage_stats[backend] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0
            }
        
        stats = self.usage_stats[backend]
        stats["total_requests"] += 1
        stats["failed_requests"] += 1
        
        # Update circuit breaker
        if backend not in self.circuit_breakers:
            self.circuit_breakers[backend] = {
                "consecutive_failures": 0,
                "is_open": False,
                "opened_at": 0,
                "timeout": 60  # 60 seconds
            }
        
        cb = self.circuit_breakers[backend]
        cb["consecutive_failures"] += 1
        
        # Open circuit after 3 consecutive failures
        if cb["consecutive_failures"] >= 3 and not cb["is_open"]:
            cb["is_open"] = True
            cb["opened_at"] = time.time()
            logger.warning(f"🔴 Circuit breaker OPENED for {backend} (3 consecutive failures)")
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all backends"""
        return {
            "backends": {
                name: {
                    "config": {
                        "provider": config.provider.value,
                        "model": config.model,
                        "is_local": config.is_local,
                        "cost_per_1k_tokens": config.cost_per_1k_tokens
                    },
                    "available": name in self.clients,
                    "circuit_open": self._is_circuit_open(name),
                    "stats": self.usage_stats.get(name, {})
                }
                for name, config in self.backends.items()
            }
        }