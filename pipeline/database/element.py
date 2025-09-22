from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any

from pipeline.utils.agent_logger import log_agent_run
from .model import summarizer
from .prompt import Summary_input


@dataclass
class DataElement:
    """Data element model for experimental results - Updated to handle new API fields."""
    time: str
    name: str
    result: Dict[str, Any]  # Changed from Dict[str, str] to Dict[str, Any] for compatibility
    program: str
    motivation: str
    analysis: str
    cognition: str
    log: str
    index: int = 0  # Changed from Optional[int] to int with default for compatibility
    parent: Optional[int] = None
    summary: str = ""  # Changed from Optional[str] to str with default
    
    # Additional fields from util.py for full compatibility
    motivation_embedding: Optional[List[float]] = None
    score: Optional[float] = None
    
    # NEW FIELDS from API update
    name_new: str = ""  # New name for the element
    parameters: str = ""  # Model parameters information
    svg_picture: str = ""  # SVG representation of the element
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DataElement instance to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataElement':
        """Create DataElement instance from dictionary - handles both old and new fields."""
        return cls(
            time=data.get('time', ''),
            name=data.get('name', ''),
            result=data.get('result', {}),
            program=data.get('program', ''),
            motivation=data.get('motivation', ''),
            analysis=data.get('analysis', ''),
            cognition=data.get('cognition', ''),
            log=data.get('log', ''),
            index=data.get('index', 0),
            parent=data.get('parent', None),
            summary=data.get('summary', ''),
            motivation_embedding=data.get('motivation_embedding', None),
            score=data.get('score', None),
            # Handle new fields with defaults
            name_new=data.get('name_new', ''),
            parameters=data.get('parameters', ''),
            svg_picture=data.get('svg_picture', '')
        )

    def __post_init__(self):
        """Post-initialization processing - ensures compatibility."""
        # Ensure required fields are not None
        if self.time is None:
            self.time = ''
        if self.name is None:
            self.name = ''
        if self.result is None:
            self.result = {}
        if self.program is None:
            self.program = ''
        if self.motivation is None:
            self.motivation = ''
        if self.analysis is None:
            self.analysis = ''
        if self.cognition is None:
            self.cognition = ''
        if self.log is None:
            self.log = ''
        if self.index is None:
            self.index = 0
    
    async def get_context(self) -> str:
        """Generate enhanced context with structured experimental evidence presentation."""
        summary = await log_agent_run(
            "summarizer",
            summarizer,
            Summary_input(self.motivation, self.analysis, self.cognition)
        )
        summary_result = summary.final_output.experience

        return f"""## EXPERIMENTAL EVIDENCE PORTFOLIO

### Experiment: {self.name}
**Architecture Identifier**: {self.name}

#### Performance Metrics Summary
**Training Progression**: {self.result.get("train", "N/A")}
**Evaluation Results**: {self.result.get("test", "N/A")}

#### Implementation Analysis
```python
{self.program}
```

#### Synthesized Experimental Insights
{summary_result}

---"""
