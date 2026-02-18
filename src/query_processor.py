"""Query intent analysis and processing."""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query type classification."""
    SIMPLE_EXTRACT = "simple_extract"      # 直接提取
    FORMAT_CONVERT = "format_convert"      # 格式转换
    COMPUTE = "compute"                    # 计算/过滤
    MULTI_STEP = "multi_step"              # 多步骤

@dataclass
class QueryIntent:
    """Analyzed query intent."""
    original_query: str
    query_type: QueryType
    target: Optional[str] = None           # e.g., "Table 1.2.1"
    output_format: str = "excel"           # excel, csv, json
    template: Optional[str] = None         # Custom format template
    confidence: float = 0.0

@dataclass
class ProcessingResult:
    """Result from query processing."""
    data: Any
    format: str
    metadata: Dict[str, Any]

class QueryProcessor:
    """Process user queries using LLM."""

    # System prompt for intent analysis
    INTENT_ANALYSIS_PROMPT = """Analyze the user's query and extract:
1. Query type (simple_extract, format_convert, compute, multi_step)
2. Target (what to extract - e.g., "Table 1.2.1")
3. Output format (excel, csv, json)

Respond in JSON format:
{
    "query_type": "...",
    "target": "...",
    "output_format": "...",
    "confidence": 0.95
}"""

    def __init__(self, model_manager):
        """Initialize query processor.

        Args:
            model_manager: ModelManager instance
        """
        self.model_manager = model_manager
        self._llm = None

    def _get_llm(self):
        """Get or load LLM for query understanding."""
        if self._llm is None:
            self._llm = self.model_manager.get_model("llama_3_1")
        return self._llm

    def analyze_intent(self, query: str) -> QueryIntent:
        """Analyze user query intent.

        Args:
            query: User's natural language query

        Returns:
            QueryIntent
        """
        # Try LLM for intent analysis if model_manager is available
        if self.model_manager is not None:
            try:
                llm = self._get_llm()

                prompt = f"{self.INTENT_ANALYSIS_PROMPT}\n\nUser query: {query}"

                response = llm(
                    prompt,
                    max_tokens=256,
                    temperature=0.1,
                    stop=["\n\n"]
                )

                # Parse JSON response
                import json
                response_text = response['choices'][0]['text'].strip()

                # Try to extract JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return QueryIntent(
                        original_query=query,
                        query_type=QueryType(data.get("query_type", "simple_extract")),
                        target=data.get("target"),
                        output_format=data.get("output_format", "excel"),
                        confidence=data.get("confidence", 0.8)
                    )

            except Exception as e:
                logger.warning(f"LLM intent analysis failed: {e}, using fallback")

        # Fallback: rule-based analysis
        return self._fallback_analysis(query)

    def _fallback_analysis(self, query: str) -> QueryIntent:
        """Fallback rule-based intent analysis."""
        query_lower = query.lower()

        # Detect target
        target = None
        table_match = re.search(r'[Tt]able\s+([\d.]+)', query)
        if table_match:
            target = f"Table {table_match.group(1)}"

        section_match = re.search(r'(\d+\.?\d*)', query)
        if section_match and not target:
            target = section_match.group(1)

        # Detect output format
        output_format = "excel"
        if "json" in query_lower:
            output_format = "json"
        elif "csv" in query_lower:
            output_format = "csv"

        # Detect query type (check multi-step first)
        query_type = QueryType.SIMPLE_EXTRACT
        if "，" in query or " then " in query_lower or "然后" in query:
            query_type = QueryType.MULTI_STEP
        elif any(word in query_lower for word in ["转换", "格式", "format", "convert"]):
            query_type = QueryType.FORMAT_CONVERT
        elif any(word in query_lower for word in ["计算", "筛选", "过滤", "filter", "compute"]):
            query_type = QueryType.COMPUTE

        return QueryIntent(
            original_query=query,
            query_type=query_type,
            target=target,
            output_format=output_format,
            confidence=0.7
        )

    def format_data(self, data: Any, intent: QueryIntent) -> str:
        """Format data according to query intent.

        Args:
            data: Data to format
            intent: Query intent with format specifications

        Returns:
            Formatted data as string
        """
        llm = self._get_llm()

        if intent.output_format == "json":
            if isinstance(data, list):
                import json
                return json.dumps(data, ensure_ascii=False, indent=2)

        # Use LLM for complex formatting
        if intent.template or intent.query_type == QueryType.FORMAT_CONVERT:
            prompt = f"""Format the following data according to the user's request.

Data: {str(data)[:1000]}

User request: {intent.original_query}

Respond only with the formatted output."""

            response = llm(
                prompt,
                max_tokens=2048,
                temperature=0.3
            )

            return response['choices'][0]['text'].strip()

        # Default formatting
        return str(data)
