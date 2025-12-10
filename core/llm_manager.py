import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, date
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
    
    cost_per_1k_tokens: float = 0.0  # USD
    avg_latency_ms: int = 1000
    max_tokens: int = 4096
    
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False
    
    reasoning_quality: int = 5
    creative_quality: int = 5
    code_quality: int = 5
    speed_score: int = 5
    
    is_local: bool = False
    requires_internet: bool = True
    max_concurrent: int = 10


class LLMManager:
    """
    Manages multiple LLM backends and intelligently routes requests with budget tracking
    """
    
    def __init__(self):
        self.backends: Dict[str, LLMConfig] = {}
        self.clients: Dict[str, Any] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        self.daily_spend: float = 0.0
        self.last_reset_date: date = datetime.now().date()
        self.request_count: int = 0
        self.budget_warnings_sent: int = 0
    
    
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
    
    
    def _reset_daily_budget_if_needed(self):
        """Reset daily spend counter if it's a new day"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            logger.info(f"New day detected. Previous day spend: ${self.daily_spend:.4f}")
            self.daily_spend = 0.0
            self.last_reset_date = today
            self.budget_warnings_sent = 0
    
    
    def _check_budget(self, estimated_cost: float, config: LLMConfig) -> tuple[bool, Optional[str]]:
        """
        Check if request is within budget limits
        
        Returns:
            (is_within_budget, warning_message)
        """
        self._reset_daily_budget_if_needed()
        
        if estimated_cost == 0:
            return True, None
        
        per_request_budget = float(os.getenv("LLM_MAX_COST_PER_REQUEST", "0.05"))
        if estimated_cost > per_request_budget:
            return False, f"Estimated cost ${estimated_cost:.4f} exceeds per-request limit ${per_request_budget:.2f}"
        
        daily_budget = float(os.getenv("LLM_DAILY_BUDGET", "5.00"))
        if self.daily_spend + estimated_cost > daily_budget:
            return False, f"Daily budget exceeded: ${self.daily_spend:.2f} + ${estimated_cost:.4f} > ${daily_budget:.2f}"
        
        warn_threshold = float(os.getenv("LLM_WARN_AT_PERCENT", "80")) / 100
        if (self.daily_spend + estimated_cost) / daily_budget >= warn_threshold and self.budget_warnings_sent == 0:
            self.budget_warnings_sent += 1
            warning = f"Budget warning: ${self.daily_spend + estimated_cost:.2f}/{daily_budget:.2f} ({((self.daily_spend + estimated_cost) / daily_budget * 100):.1f}%)"
            logger.warning(warning)
            return True, warning
        
        return True, None
    
    
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
            if name not in self.clients:
                logger.debug(f"Skipping {name} - client not initialized")
                continue
            
            if self._is_circuit_open(name):
                logger.debug(f"Skipping {name} - circuit breaker open")
                continue
            
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
            
            score = self._calculate_backend_score(
                config, task_type, complexity, prefer_fast
            )
            
            candidates.append((name, score, config))
        
        if not candidates:
            logger.warning("No suitable backends available")
            return None
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected = candidates[0][0]
        logger.info(f"Selected backend: {selected} (score: {candidates[0][1]:.2f})")
        
        return selected
    
    
    def _calculate_backend_score(
        self,
        config: LLMConfig,
        task_type: str,
        complexity: TaskComplexity,
        prefer_fast: bool,
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate intelligent score for backend selection with task metadata awareness
        
        Args:
            config: Backend configuration
            task_type: Type of task (creative, reasoning, code, analytical)
            complexity: Task complexity level
            prefer_fast: Whether to prefer fast backends
            task_metadata: Additional task context (high_stakes, needs_speed, etc.)
        """
        task_metadata = task_metadata or {}
        score = 0.0
        
        quality_weight = 5.0  # Increased from 3.0
        
        if task_type == "creative":
            base_quality = config.creative_quality * quality_weight
        elif task_type == "code":
            base_quality = config.code_quality * quality_weight
        elif task_type in ["reasoning", "analytical"]:
            base_quality = config.reasoning_quality * quality_weight
        else:
            avg_quality = (config.reasoning_quality + config.creative_quality) / 2
            base_quality = avg_quality * quality_weight
        
        score += base_quality
        
        if task_metadata.get("is_high_stakes", False):
            if config.reasoning_quality >= 9:  # Premium models (GPT-4, Claude)
                score += 20
                logger.debug(f"High-stakes bonus (+20) for {config.model}")
            elif config.reasoning_quality >= 7:  # Good models
                score += 8
            else:  # Lower quality models
                score -= 10  # Penalty for using weak models on important tasks
        
        needs_speed = task_metadata.get("needs_speed", False)
        prefer_fast_env = os.getenv("LLM_PREFER_FAST", "true").lower() == "true"
        
        if needs_speed or prefer_fast or prefer_fast_env:
            if config.avg_latency_ms < 1000:  # Ultra-fast (Groq)
                score += 15
            elif config.avg_latency_ms < 3000:  # Fast
                score += 8
            elif config.avg_latency_ms > 8000:  # Slow
                score -= 10
        
        if complexity == TaskComplexity.SIMPLE:
            if config.reasoning_quality >= 9:
                score -= 8  # Penalty for overkill (save budget)
            elif config.reasoning_quality >= 7:
                score += 5  # Good balance
            else:
                score += 10  # Reward cheap models for simple tasks
        
        elif complexity == TaskComplexity.MEDIUM:
            if config.reasoning_quality >= 7:
                score += 8
        
        elif complexity == TaskComplexity.COMPLEX:
            if config.reasoning_quality >= 8:
                score += 15
            elif config.reasoning_quality < 6:
                score -= 15  # Penalty for weak models
        
        elif complexity == TaskComplexity.EXPERT:
            if config.reasoning_quality >= 9:
                score += 25
            elif config.reasoning_quality >= 7:
                score += 5
            else:
                score -= 25  # Strong penalty for inadequate models
        
        if config.cost_per_1k_tokens == 0:
            score += 10  # Good bonus for free
        else:
            cost_multiplier = 100
            
            if task_metadata.get("is_high_stakes", False):
                cost_multiplier = 50  # Less harsh penalty
            
            max_cost = float(os.getenv("LLM_MAX_COST_PER_REQUEST", "0.05"))
            
            if config.cost_per_1k_tokens <= max_cost:
                score -= config.cost_per_1k_tokens * cost_multiplier
            else:
                if task_metadata.get("is_high_stakes", False):
                    score -= 40  # Moderate penalty for high-stakes
                else:
                    score -= 60  # Higher penalty for regular tasks
        
        word_estimate = task_metadata.get("word_count_estimate", 0)
        if word_estimate > 400:
            if config.reasoning_quality >= 8:
                score += 10
        
        prefer_local = os.getenv("LLM_PREFER_LOCAL", "false").lower() == "true"
        if prefer_local and config.is_local:
            score += 12  # Bonus for local
        elif not prefer_local and not config.is_local:
            score += 3  # Small bonus for cloud (more reliable)
        
        priority_list = os.getenv("LLM_BACKEND_PRIORITY", "").split(",")
        if priority_list[0]:  # Check if not empty
            for idx, provider in enumerate(priority_list):
                if config.provider.value == provider.strip():
                    priority_bonus = max(0, 20 - (idx * 10))
                    score += priority_bonus
                    logger.debug(f"Priority bonus (+{priority_bonus}) for {config.provider.value}")
                    break
        
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
        Generate completion using selected or automatic backend with fallback and budget tracking
        
        Args:
            prompt: Input prompt
            backend: Specific backend to use (or auto-select)
            task_type: Type of task
            complexity: Complexity level
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific arguments including task_metadata
            
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
        self.request_count += 1
        
        if backend:
            if backend not in self.clients:
                available = [name for name in self.backends.keys() if name in self.clients]
                raise Exception(
                    f"Backend '{backend}' not available. "
                    f"Available backends: {available if available else 'None'}"
                )
            
            config = self.backends[backend]
            estimated_cost = (max_tokens / 1000) * config.cost_per_1k_tokens
            within_budget, warning = self._check_budget(estimated_cost, config)
            
            if not within_budget:
                raise Exception(f"Budget limit: {warning}")
            
            if warning:
                logger.warning(warning)
            
            return await self._attempt_completion(
                backend, prompt, max_tokens, temperature, **kwargs
            )
        
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
        
        last_error = None
        
        for backend_name, score, config in candidates:
            try:
                estimated_cost = (max_tokens / 1000) * config.cost_per_1k_tokens
                within_budget, warning = self._check_budget(estimated_cost, config)
                
                if not within_budget:
                    logger.warning(f"Skipping {backend_name}: {warning}")
                    continue  # Try next backend
                
                if warning:
                    logger.warning(warning)
                
                logger.info(f"Attempting backend: {backend_name} (score: {score:.2f}, est. cost: ${estimated_cost:.4f})")
                
                result = await self._attempt_completion(
                    backend_name, prompt, max_tokens, temperature, **kwargs
                )
                
                logger.info(f"Successfully used backend: {backend_name} (cost: ${result['cost']:.4f})")
                return result
                
            except Exception as e:
                logger.warning(f"Backend {backend_name} failed: {e}")
                last_error = e
                
                continue
        
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
        task_metadata = kwargs.get("task_metadata", {})  # NEW
        
        for name, config in self.backends.items():
            if name not in self.clients:
                logger.debug(f"Skipping {name} - client not initialized")
                continue
            
            if self._is_circuit_open(name):
                logger.debug(f"Skipping {name} - circuit breaker open")
                continue
            
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
            
            score = self._calculate_backend_score(
                config, task_type, complexity, prefer_fast, task_metadata
            )
            
            candidates.append((name, score, config))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            logger.info(f"Top backend candidates:")
            for idx, (name, score, config) in enumerate(candidates[:3], 1):
                logger.info(f"   {idx}. {name}: score={score:.1f}, "
                          f"quality={config.reasoning_quality}, "
                          f"cost=${config.cost_per_1k_tokens:.4f}/1K")
        
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
        
        start_time = time.time()
        
        try:
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
            
            latency_ms = int((time.time() - start_time) * 1000)
            tokens_used = result.get("tokens_used", 0)
            cost = (tokens_used / 1000) * config.cost_per_1k_tokens
            
            self._record_success(backend, latency_ms, tokens_used, cost)
            self.daily_spend += cost
            
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
            self._record_failure(backend)
            
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
        
        n = stats["successful_requests"]
        stats["avg_latency_ms"] = (
            stats["avg_latency_ms"] * (n - 1) + latency_ms
        ) / n
        
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
        
        if backend not in self.circuit_breakers:
            self.circuit_breakers[backend] = {
                "consecutive_failures": 0,
                "is_open": False,
                "opened_at": 0,
                "timeout": 60  # 60 seconds
            }
        
        cb = self.circuit_breakers[backend]
        cb["consecutive_failures"] += 1
        
        if cb["consecutive_failures"] >= 3 and not cb["is_open"]:
            cb["is_open"] = True
            cb["opened_at"] = time.time()
            logger.warning(f"Circuit breaker OPENED for {backend} (3 consecutive failures)")
    
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status (NEW)"""
        self._reset_daily_budget_if_needed()
        
        daily_budget = float(os.getenv("LLM_DAILY_BUDGET", "5.00"))
        per_request_budget = float(os.getenv("LLM_MAX_COST_PER_REQUEST", "0.05"))
        
        return {
            "daily_spend": self.daily_spend,
            "daily_budget": daily_budget,
            "daily_remaining": daily_budget - self.daily_spend,
            "daily_percent_used": (self.daily_spend / daily_budget * 100) if daily_budget > 0 else 0,
            "per_request_limit": per_request_budget,
            "total_requests_today": self.request_count,
            "last_reset_date": self.last_reset_date.isoformat()
        }
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all backends"""
        return {
            "backends": {
                name: {
                    "config": {
                        "provider": config.provider.value,
                        "model": config.model,
                        "is_local": config.is_local,
                        "cost_per_1k_tokens": config.cost_per_1k_tokens,
                        "reasoning_quality": config.reasoning_quality,
                        "creative_quality": config.creative_quality,
                        "avg_latency_ms": config.avg_latency_ms
                    },
                    "available": name in self.clients,
                    "circuit_open": self._is_circuit_open(name),
                    "stats": self.usage_stats.get(name, {})
                }
                for name, config in self.backends.items()
            },
            "budget": self.get_budget_status()
        }