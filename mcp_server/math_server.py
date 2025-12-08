import asyncio
from typing import Dict, Any, Optional
import logging
import re

from core.agent_base import BaseAgent
from core.message_bus import MessageBus, Message

logger = logging.getLogger(__name__)


class MathMCPServer(BaseAgent):
    """
    Math MCP Server - Handles mathematical calculations and visualizations.
    Implements Model Context Protocol for math operations.
    """
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(
            agent_id="math_mcp_server",
            name="Math MCP Server",
            message_bus=message_bus,
            capabilities=[
                "fibonacci",
                "factorial",
                "basic_arithmetic",
                "sequence_generation",
                "prime_numbers",
                "equation_solving"
            ]
        )
        
        # Cache for expensive calculations
        self.calculation_cache: Dict[str, Any] = {}
    
    
    async def setup(self):
        """Subscribe to math requests"""
        self.message_bus.subscribe(
            topic="math_request",
            agent_id=self.agent_id,
            callback=self._process_message
        )
        logger.info("Math MCP Server subscribed to topic: math_request")
    
    
    async def handle_message(self, message: Message):
        """Route incoming messages to appropriate handlers"""
        if message.topic == "math_request":
            await self.handle_math_request(message)
        else:
            logger.warning(f"Math MCP Server received unknown topic: {message.topic}")
    
    
    async def handle_math_request(self, message: Message):
        """
        Handle mathematical calculation requests.
        
        Expected payload:
        {
            "query": "calculate fibonacci 10",
            "operation": "auto_detect" or specific operation
        }
        """
        query = message.payload.get("query", "").lower()
        operation = message.payload.get("operation", "auto_detect")
        
        logger.info(f"Math MCP processing query: '{query}'")
        
        try:
            # Check cache first
            cache_key = f"{operation}:{query}"
            if cache_key in self.calculation_cache:
                logger.info("Result found in cache")
                result = self.calculation_cache[cache_key]
            else:
                # Perform calculation
                if operation == "auto_detect":
                    operation = self._detect_operation(query)
                
                result = await self._execute_operation(operation, query)
                
                # Cache result
                self.calculation_cache[cache_key] = result
            
            # Send response back
            await self.send_response(
                original_message=message,
                payload={
                    "result": result,
                    "operation": operation,
                    "status": "completed"
                },
                topic="math_response"
            )
            
            logger.info(f"Math calculation completed: {operation}")
            
        except Exception as e:
            logger.error(f"Error in math calculation: {e}", exc_info=True)
            await self.send_error(
                original_message=message,
                error=str(e),
                details={"operation": operation, "query": query}
            )
    
    
    def _detect_operation(self, query: str) -> str:
        """
        Auto-detect the mathematical operation from query text.
        
        Args:
            query: User's query string
            
        Returns:
            Detected operation type
        """
        query_lower = query.lower()
        
        # Check for different operations
        if any(word in query_lower for word in ["fibonacci", "fib"]):
            return "fibonacci"
        
        elif any(word in query_lower for word in ["factorial"]):
            return "factorial"
        
        elif any(word in query_lower for word in ["prime", "primes"]):
            return "prime_numbers"
        
        elif any(word in query_lower for word in ["add", "sum", "plus", "+"]):
            return "addition"
        
        elif any(word in query_lower for word in ["multiply", "times", "*", "×"]):
            return "multiplication"
        
        else:
            return "unknown"
    
    
    async def _execute_operation(self, operation: str, query: str) -> Dict[str, Any]:
        """
        Execute the mathematical operation.
        
        Args:
            operation: Type of operation to perform
            query: User's query string
            
        Returns:
            Calculation results
        """
        if operation == "fibonacci":
            return self._calculate_fibonacci(query)
        
        elif operation == "factorial":
            return self._calculate_factorial(query)
        
        elif operation == "prime_numbers":
            return self._calculate_primes(query)
        
        elif operation == "addition":
            return self._perform_arithmetic(query, operation)
        
        elif operation == "multiplication":
            return self._perform_arithmetic(query, operation)
        
        else:
            return {
                "message": f"Operation '{operation}' not yet implemented",
                "suggestions": ["fibonacci", "factorial", "prime_numbers", "addition"]
            }
    
    
    def _calculate_fibonacci(self, query: str) -> Dict[str, Any]:
        """
        Calculate Fibonacci sequence.
        
        Args:
            query: Query string (may contain number)
            
        Returns:
            Fibonacci sequence data
        """
        # Extract number from query (default to 10)
        n = self._extract_number(query, default=10)
        n = min(n, 50)  # Limit to prevent excessive computation
        
        # Generate sequence
        sequence = [0, 1]
        for i in range(2, n):
            sequence.append(sequence[i-1] + sequence[i-2])
        
        return {
            "sequence": sequence,
            "count": n,
            "formula": "F(n) = F(n-1) + F(n-2)",
            "explanation": "Each number is the sum of the two preceding ones",
            "properties": {
                "golden_ratio": "The ratio of consecutive Fibonacci numbers approaches φ ≈ 1.618",
                "in_nature": "Found in spirals, flower petals, and tree branches"
            }
        }
    
    
    def _calculate_factorial(self, query: str) -> Dict[str, Any]:
        """
        Calculate factorial.
        
        Args:
            query: Query string containing number
            
        Returns:
            Factorial calculation result
        """
        n = self._extract_number(query, default=5)
        n = min(n, 20)  # Limit to prevent overflow
        
        # Calculate factorial
        result = 1
        for i in range(1, n + 1):
            result *= i
        
        return {
            "n": n,
            "result": result,
            "formula": f"{n}! = {' × '.join(map(str, range(1, n+1)))}",
            "explanation": f"The factorial of {n} is the product of all positive integers up to {n}",
            "applications": ["Permutations", "Combinations", "Probability calculations"]
        }
    
    
    def _calculate_primes(self, query: str) -> Dict[str, Any]:
        """
        Calculate prime numbers up to n.
        
        Args:
            query: Query string containing number
            
        Returns:
            List of prime numbers
        """
        n = self._extract_number(query, default=20)
        n = min(n, 1000)  # Limit range
        
        # Sieve of Eratosthenes
        primes = []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        
        primes = [i for i in range(n + 1) if is_prime[i]]
        
        return {
            "primes": primes,
            "count": len(primes),
            "range": f"up to {n}",
            "explanation": "Prime numbers are natural numbers greater than 1 that have no positive divisors other than 1 and themselves"
        }
    
    
    def _perform_arithmetic(self, query: str, operation: str) -> Dict[str, Any]:
        """
        Perform basic arithmetic operations.
        
        Args:
            query: Query string with numbers
            operation: Type of arithmetic operation
            
        Returns:
            Calculation result
        """
        numbers = self._extract_all_numbers(query)
        
        if len(numbers) < 2:
            return {
                "error": "Need at least 2 numbers for arithmetic",
                "found_numbers": numbers
            }
        
        if operation == "addition":
            result = sum(numbers)
            symbol = "+"
        elif operation == "multiplication":
            result = 1
            for num in numbers:
                result *= num
            symbol = "×"
        else:
            result = 0
            symbol = "?"
        
        return {
            "numbers": numbers,
            "operation": operation,
            "result": result,
            "expression": f" {symbol} ".join(map(str, numbers)) + f" = {result}"
        }
    
    
    def _extract_number(self, text: str, default: int = 10) -> int:
        """
        Extract a number from text.
        
        Args:
            text: Text containing number
            default: Default value if no number found
            
        Returns:
            Extracted number or default
        """
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else default
    
    
    def _extract_all_numbers(self, text: str) -> list:
        """
        Extract all numbers from text.
        
        Args:
            text: Text containing numbers
            
        Returns:
            List of numbers
        """
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(n) if '.' in n else int(n) for n in numbers]
    
    
    def clear_cache(self):
        """Clear the calculation cache"""
        self.calculation_cache.clear()
        logger.info("Math MCP Server cache cleared")