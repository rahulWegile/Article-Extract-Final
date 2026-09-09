from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class BlockKnowledge:
    block_id: str
    text: str = ""
    confidence: float = 0.0
    first_sentence: str = ""
    last_sentence: str = ""
    first_word: str = ""
    last_word: str = ""
    starts_with_lowercase: bool = False
    ends_with_punctuation: bool = False
    language: str = ""
    entities: List[Any] = field(default_factory=list)
    heading_probability: float = 0.0
    caption_probability: float = 0.0
    continuation_probability: float = 0.0
    embedding: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
