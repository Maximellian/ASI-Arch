from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import inspect
from agents import Runner
import csv
import io

class AgentLogger:
    """Agent call logger"""

    def __init__(self, log_dir: str = "pipeline/logs/agent_calls"):
        """
        Initializes the logger

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.detailed_log_dir = self.log_dir / "detailed"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.detailed_log_dir.mkdir(parents=True, exist_ok=True)

        # Create the main log file
        self.main_log_file = self.log_dir / "agent_calls.log"

        # Pipeline related
        self.current_pipeline_id: Optional[str] = None
        self.current_pipeline_dir: Optional[Path] = None
        self.pipeline_log_file: Optional[Path] = None
        self.pipeline_full_log_file: Optional[Path] = None  # Full log file

    def start_pipeline(self, pipeline_name: str = "") -> str:
        """
        Starts a new pipeline process

        Args:
            pipeline_name: Pipeline name (optional)

        Returns:
            pipeline_id: Generated pipeline ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if pipeline_name:
            self.current_pipeline_id = f"pipeline_{timestamp}_{pipeline_name}"
        else:
            self.current_pipeline_id = f"pipeline_{timestamp}"

        # Creates pipeline-specific directory
        self.current_pipeline_dir = self.log_dir / self.current_pipeline_id
        self.current_pipeline_dir.mkdir(exist_ok=True)

        # Creates pipeline log file
        self.pipeline_log_file = self.current_pipeline_dir / "pipeline.log"
        self.pipeline_full_log_file = self.current_pipeline_dir / "full.log"

        # Logs the start of the pipeline
        pipeline_start_log = {
            "pipeline_id": self.current_pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "status": "started",
            "pipeline_name": pipeline_name
        }

        self._write_pipeline_log(pipeline_start_log)
        self.log_info(f"Started pipeline: {self.current_pipeline_id}")

        return self.current_pipeline_id

    def end_pipeline(self, success: bool = True, summary: str = "") -> None:
        """
        Ends the current pipeline process

        Args:
            success: Whether the pipeline completed successfully
            summary: Pipeline summary information
        """
        if not self.current_pipeline_id:
            print("Warning: No active pipeline to end")
            return

        # Get the pipeline's usage statistics
        pipeline_usage = self._get_pipeline_usage_internal()

        # Log that the pipeline has ended
        pipeline_end_log = {
            "pipeline_id": self.current_pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if success else "failed",
            "summary": summary,
            "usage_summary": pipeline_usage
        }

        self._write_pipeline_log(pipeline_end_log)

        # Log the end of the pipeline and the usage
        if pipeline_usage.get("total_tokens", 0) > 0:
            usage_info = f"Input: {pipeline_usage['input_tokens']}, Output: {pipeline_usage['output_tokens']}, Total: {pipeline_usage['total_tokens']} tokens"
            self.log_info(f"Ended pipeline: {self.current_pipeline_id} ({'success' if success else 'failed'}) - Usage: {usage_info}")
        else:
            self.log_info(f"Ended pipeline: {self.current_pipeline_id} ({'success' if success else 'failed'})")

        # Clear current pipeline state
        self.current_pipeline_id = None
        self.current_pipeline_dir = None
        self.pipeline_log_file = None
        self.pipeline_full_log_file = None

    async def log_agent_call(self, agent_name: str, agent, input_data: Any = None, **kwargs) -> Any:
        """
        Logs the complete agent call and executes the call

        Args:
            agent_name: Agent name
            agent: Agent object
            input_data: Input data
            **kwargs: Other parameters passed to Runner.run

        Returns:
            The result of the agent call
        """
        call_id, timestamp, start_log = self._generate_call_id()

        # Log the start of the call
        start_log = {
            "call_id": call_id,
            "timestamp": timestamp,
            "agent_name": agent_name,
            "status": "started",
            "input": self._serialize_data(input_data),
            "kwargs": self._serialize_data(kwargs),
            "pipeline_id": self.current_pipeline_id  # Add pipeline_id
        }

        self._write_log(start_log)
        if self.current_pipeline_id:
            self._write_pipeline_log(start_log)  # Also write to the pipeline log

        try:
            # Execute the agent call
            result = await Runner.run(agent, input=input_data, **kwargs)

            # Extract usage information from the agents library's return result
            usage_info = self._extract_usage_from_result(result)

            # Log successful outcome
            end_log = {
                "call_id": call_id,
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "status": "completed",
                "output": self._serialize_data(result.final_output if hasattr(result, 'final_output') else result),
                "full_result": self._serialize_data(result, max_depth=5),  # Limit depth to prevent excessive size
                "pipeline_id": self.current_pipeline_id,  # Add pipeline_id
                "usage": usage_info  # Add usage info extracted from result
            }

            self._write_log(end_log)
            if self.current_pipeline_id:
                self._write_pipeline_log(end_log)  # Also write to the pipeline log

            # Create Separate detailed log file
            self._create_detailed_log(call_id, agent_name, start_log, end_log)

            return result

        except Exception as e:
            # Log the error
            error_log = {
                "call_id": call_id,
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "pipeline_id": self.current_pipeline_id,  # Add pipeline_id
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}  # Usage set to 0 on error
            }

            self._write_log(error_log)
            if self.current_pipeline_id:
                self._write_pipeline_log(error_log)  # Also write to the pipeline log

            # Create detailed error log file
            self._create_detailed_log(call_id, agent_name, start_log, error_log)

            # Re-raise the exception
            raise

    def _generate_call_id(self) -> str:
        """Generates a unique call ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"call_{timestamp}", timestamp, {
            "call_id": f"call_{timestamp}",
            "timestamp": timestamp,
            "agent_name": "UnknownAgent",
            "status": "started",
            "input": "None",
            "kwargs": "None",
            "pipeline_id": "None"
        }

    def _extract_usage_from_result(self, result: Any) -> Dict[str, int]:
        """
        Extracts usage information from the results returned by agents.Runner.

        Gets the usage data from the raw_responses that contain the raw API responses.

        Args:
            result: The result of Runner.run

        Returns:
            A dictionary containing usage information
        """
        default_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # First, verify whether the object has the 'raw_responses' attribute
        if hasattr(result, 'raw_responses'):
            raw_responses = result.raw_responses
            if raw_responses:
                # Accumulate usage information across all responses
                total_input_tokens = 0
                total_output_tokens = 0
                total_tokens = 0

                # If raw_responses is a list, iterate over all responses
                if isinstance(raw_responses, (list, tuple)):
                    for response in raw_responses:
                        usage_data = self._extract_usage_from_single_response(response)
                        if usage_data:
                            total_input_tokens += usage_data.get("input_tokens", 0)
                            total_output_tokens += usage_data.get("output_tokens", 0)
                            total_tokens += usage_data.get("total_tokens", 0)

                # If raw_responses is a single response
                else:
                    usage_data = self._extract_usage_from_single_response(raw_responses)
                    if usage_data:
                        total_input_tokens = usage_data.get("input_tokens", 0)
                        total_output_tokens = usage_data.get("output_tokens", 0)
                        total_tokens = usage_data.get("total_tokens", 0)

                if total_input_tokens > 0 or total_output_tokens > 0 or total_tokens > 0:
                    return {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                        "total_tokens": total_tokens
                    }

        # If raw_responses haven't been accessible, check for direct usage attribute as a fallback
        if hasattr(result, 'usage'):
            usage_obj = result.usage
            if hasattr(usage_obj, 'input_tokens'):
                return {
                    "input_tokens": getattr(usage_obj, 'input_tokens', 0),
                    "output_tokens": getattr(usage_obj, 'output_tokens', 0),
                    "total_tokens": getattr(usage_obj, 'total_tokens', 0)
                }

        return default_usage

    def _extract_usage_from_single_response(self, response: Any) -> Dict[str, int]:
        """
        Extracts usage information from a single API response

        Args:
            response: Single API response object

        Returns:
            A dict containing usage information, or None if usage data isn't found.
        """
        if hasattr(response, 'usage'):
            usage_obj = response.usage
            if usage_obj:
                # if usage_obj is dict
                if isinstance(usage_obj, dict):
                    return {
                        "input_tokens": usage_obj.get("input_tokens", usage_obj.get("prompt_tokens", 0)),
                        "output_tokens": usage_obj.get("output_tokens", usage_obj.get("completion_tokens", 0)),
                        "total_tokens": usage_obj.get("total_tokens", 0)
                    }

                # If usage_obj has defined properties, try to get the attributes
                elif hasattr(usage_obj, 'input_tokens') or hasattr(usage_obj, 'prompt_tokens'):
                    input_tokens = (getattr(usage_obj, 'input_tokens', 0) or
                                  getattr(usage_obj, 'prompt_tokens', 0))
                    output_tokens = (getattr(usage_obj, 'output_tokens', 0) or
                                   getattr(usage_obj, 'completion_tokens', 0))
                    total_tokens = getattr(usage_obj, 'total_tokens', input_tokens + output_tokens)

                    return {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    }

        return None

    def _get_pipeline_usage_internal(self) -> Dict[str, int]:
        """Gets the total usage of the current pipeline"""
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        if not self.pipeline_log_file or not self.pipeline_log_file.exists():
            return total_usage

        try:
            with open(self.pipeline_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        if 'usage' in log_entry and isinstance(log_entry['usage'], dict):
                            total_usage["input_tokens"] += log_entry['usage'].get("input_tokens", 0)
                            total_usage["output_tokens"] += log_entry['usage'].get("output_tokens", 0)
                            total_usage["total_tokens"] += log_entry['usage'].get("total_tokens", 0)
                    except json.JSONDecodeError:
                        continue

            return total_usage

        except Exception as e:
            print(f"Warning: Failed to read pipeline usage from log: {e}")
            return total_usage

    def _serialize_data(self, data: Any, max_depth: int = 5, current_depth: int = 0) -> Any:
        """
        Recursively serializes data, handling common un-serializable types

        Args:
            data: Data to serialize
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            Serializable data
        """
        if current_depth >= max_depth:
            return f"Max depth of {max_depth} reached"

        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, (list, tuple, set)):
            return [self._serialize_data(item, max_depth, current_depth + 1) for item in data]
        elif isinstance(data, dict):
            return {str(k): self._serialize_data(v, max_depth, current_depth + 1) for k, v in data.items()}
        elif hasattr(data, 'to_dict') and callable(data.to_dict):
            return data.to_dict()
        elif hasattr(data, '__dict__'):
            return self._serialize_object(data, max_depth, current_depth)
        else:
            try:
                # Last resort attempt
                return str(data)
            except Exception:
                return f"Unserializable object of type {type(data).__name__}"

    def _serialize_object(self, data: Any, max_depth: int, current_depth: int) -> Any:
        """Serialize a general object"""
        # Special handling for Agent objects, avoiding circular references and excessive detail
        if 'Agent' in str(type(data)):  # Simple type check
            agent_info = {"agent_name": getattr(data, 'name', 'UnknownAgent')}
            if hasattr(data, 'tools'):
                agent_info['tools'] = [t.name for t in data.tools]
            return agent_info

        # Check if the returned value is Runner's result
        if 'Runner' in str(type(data)) and hasattr(data, 'final_output'):
            return self._serialize_data(data.final_output, max_depth, current_depth + 1)

        # Serialize the object's __dict__
        obj_dict = {}
        for key, value in data.__dict__.items():
            if key.startswith('_'):  # Skip private attributes
                continue
            obj_dict[key] = self._serialize_data(value, max_depth, current_depth + 1)

        # Add type information
        obj_dict['__type__'] = type(data).__name__

        return obj_dict

    def _write_log(self, log_data: Dict[str, Any]) -> None:
        """Writes the log to the main log file"""
        try:
            with open(self.main_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False, indent=None) + '\n')
        except Exception as e:
            print(f"Failed to write log: {e}")

    def _create_detailed_log(self, call_id: str, agent_name: str, start_log: Dict, end_log: Dict) -> None:
        """Creates a detailed log file for a single call"""
        try:
            log_content = {
                "call_id": call_id,
                "agent_name": agent_name,
                "start": start_log,
                "end": end_log
            }

            # Detailed log file path
            log_file = self.detailed_log_dir / f"{call_id}_{agent_name}.json"

            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(log_content, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Failed to create detailed log: {e}")

        except Exception as e:
            print(f"Failed to create detailed log: {e}")

    def _write_pipeline_log(self, log_data: Dict[str, Any]) -> None:
        """Writes the log to the pipeline-specific log file"""
        try:
            if self.pipeline_log_file:
                with open(self.pipeline_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_data, ensure_ascii=False, indent=None) + '\n')
        except Exception as e:
            print(f"Failed to write pipeline log: {e}")

    def _write_full_log(self, message: str, level: str = "INFO") -> None:
        """Writes the full log to the pipeline's full log file"""
        try:
            if self.pipeline_full_log_file:
                timestamp = datetime.now().isoformat()
                log_entry = f"[{timestamp}] [{level}] {message}\n"
                with open(self.pipeline_full_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)

                # Output to console as well
                print(f"[{level}] {message}")
        except Exception as e:
            print(f"Failed to write full log: {e}")

    def log_info(self, message: str) -> None:
        """Logs an info message"""
        self._write_full_log(message, "INFO")

    def log_warning(self, message: str) -> None:
        """Logs a warning message"""
        self._write_full_log(message, "WARNING")

    def log_error(self, message: str) -> None:
        """Logs an error message"""
        self._write_full_log(message, "ERROR")

    def log_debug(self, message: str) -> None:
        """Logs a debug message"""
        self._write_full_log(message, "DEBUG")

    def log_step(self, step_name: str, message: str = "") -> None:
        """Logs a step message"""
        full_message = f"=== {step_name} ===" + (f" {message}" if message else "")
        self._write_full_log(full_message, "STEP")

    def get_agent_call_stats(self) -> Dict[str, Any]:
        """Gets agent call statistics"""
        try:
            if not self.main_log_file.exists():
                return {
                    "total_calls": 0,
                    "by_agent": {},
                    "by_status": {},
                    "usage": {
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "total_tokens": 0,
                        "by_agent": {}
                    }
                }

            stats = {
                "total_calls": 0,
                "by_agent": {},
                "by_status": {"started": 0, "completed": 0, "failed": 0},
                "usage": {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "by_agent": {}
                }
            }

            # Use a dictionary to merge the 'started' and 'completed'/'failed' logs for each call_id
            agent_calls = {}

            with open(self.main_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_data = json.loads(line.strip())
                        call_id = log_data.get("call_id")
                        if not call_id:
                            continue

                        if call_id not in agent_calls:
                            agent_calls[call_id] = {}

                        agent_calls[call_id].update(log_data)

                    except json.JSONDecodeError:
                        continue

            # Aggregate statistics
            for call_id, log_data in agent_calls.items():
                stats["total_calls"] += 1

                agent_name = log_data.get("agent_name", "unknown")
                status = log_data.get("status", "unknown")

                stats["by_agent"][agent_name] = stats["by_agent"].get(agent_name, 0) + 1

                if status in stats["by_status"]:
                    stats["by_status"][status] += 1

                # Usage statistics
                if status == "completed":
                    usage_data = log_data.get("usage", {})
                    if usage_data:
                        input_tokens = usage_data.get("input_tokens", 0)
                        output_tokens = usage_data.get("output_tokens", 0)
                        total_tokens = usage_data.get("total_tokens", 0)

                        stats["usage"]["total_input_tokens"] += input_tokens
                        stats["usage"]["total_output_tokens"] += output_tokens
                        stats["usage"]["total_tokens"] += total_tokens

                        # Agent-specific usage
                        if agent_name not in stats["usage"]["by_agent"]:
                            stats["usage"]["by_agent"][agent_name] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

                        stats["usage"]["by_agent"][agent_name]["input_tokens"] += input_tokens
                        stats["usage"]["by_agent"][agent_name]["output_tokens"] += output_tokens
                        stats["usage"]["by_agent"][agent_name]["total_tokens"] += total_tokens

            return stats

        except Exception as e:
            print(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Create a global logger instance
_global_logger = None

def get_logger(log_dir: str = "pipeline/logs/agent_calls") -> AgentLogger:
    """Gets the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentLogger(log_dir)
    return _global_logger

async def log_agent_run(agent_name: str, agent, input_data: Any = None, **kwargs) -> Any:
    """
    Convenience function: log and execute an agent call

    Args:
        agent_name: Agent name
        agent: Agent object
        input_data: Input data
        **kwargs: Other parameters

    Returns:
        Agent call result
    """
    logger = get_logger()
    return await logger.log_agent_call(agent_name, agent, input_data, **kwargs)

def start_pipeline(pipeline_name: str = "") -> str:
    """
    Convenience function: Starts a new pipeline

    Args:
        pipeline_name: Pipeline name (optional)

    Returns:
        pipeline_id: Generated pipeline ID
    """
    logger = get_logger()
    return logger.start_pipeline(pipeline_name)

def end_pipeline(success: bool = True, summary: str = "") -> None:
    """
    Convenience function: Ends the current pipeline

    Args:
        success: If the pipeline succeeded
        summary: Pipeline summary information
    """
    logger = get_logger()
    logger.end_pipeline(success, summary)

def get_current_pipeline_id() -> Optional[str]:
    """
    Convenience function: Gets the current pipeline ID

    Returns:
        The current pipeline ID, or None if there's no active pipeline
    """
    logger = get_logger()
    return logger.current_pipeline_id

def log_info(message: str) -> None:
    """Convenience function: Logs info messages"""
    logger = get_logger()
    logger.log_info(message)

def log_warning(message: str) -> None:
    """Convenience function: Logs warning messages"""
    logger = get_logger()
    logger.log_warning(message)

def log_error(message: str) -> None:
    """Convenience function: Logs error messages"""
    logger = get_logger()
    logger.log_error(message)

def log_debug(message: str) -> None:
    """Convenience function: Logs debug messages"""
    logger = get_logger()
    logger.log_debug(message)

def log_step(step_name: str, message: str = "") -> None:
    """Convenience function: Logs a step"""
    logger = get_logger()
    logger.log_step(step_name, message)

def get_usage_stats() -> Dict[str, Any]:
    """
    Convenience function: Gets usage statistics.

    Returns:
        Dictionary containing usage statistics.
    """
    logger = get_logger()
    stats = logger.get_agent_call_stats()
    return stats.get("usage", {})

def get_current_pipeline_usage() -> Dict[str, int]:
    """
    Convenience function: Gets the usage statistics from current pipeline.

    Returns:
        The usage statistics for the current pipeline.
    """
    logger = get_logger()
    return logger._get_pipeline_usage_internal()

def log_usage_summary() -> None:
    """Convenience function: Prints a summary of current usage in the logs"""
    logger = get_logger()
    stats = get_usage_stats()

    if stats and stats.get("total_tokens", 0) > 0:
        total_info = f"Total session usage: Input: {stats['total_input_tokens']}, Output: {stats['total_output_tokens']}, Total: {stats['total_tokens']} tokens"
        logger.log_info(total_info)

        if logger.current_pipeline_id:
            pipeline_usage = get_current_pipeline_usage()
            if pipeline_usage.get("total_tokens", 0) > 0:
                pipeline_info = f"Current pipeline usage: Input: {pipeline_usage['input_tokens']}, Output: {pipeline_usage['output_tokens']}, Total: {pipeline_usage['total_tokens']} tokens"
                logger.log_info(pipeline_info)
    else:
        logger.log_info("No usage data available yet.")


@dataclass
class DataElement:
    """Data element class - Updated to handle new API fields"""
    time: str
    name: str
    result: Dict[str, Any]
    program: str
    analysis: str
    cognition: str
    log: str
    motivation: str
    index: int
    motivation_embedding: Optional[List[float]] = None
    parent: Optional[int] = None  # index of parent, None for root
    summary: str = ""  # Summary of the element
    score: Optional[float] = None  # Calculated score, can cache
    
    # NEW FIELDS from API update
    name_new: str = ""  # New name for the element
    parameters: str = ""  # Model parameters information
    svg_picture: str = ""  # SVG representation of the element

    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataElement':
        """Creates an instance from a dictionary - handles both old and new fields"""
        return cls(
            time=data.get('time', ''),
            name=data.get('name', ''),
            result=data.get('result', {}),
            program=data.get('program', ''),
            analysis=data.get('analysis', ''),
            cognition=data.get('cognition', ''),
            log=data.get('log', ''),
            motivation=data.get('motivation', ''),
            index=data.get('index', 0),
            motivation_embedding=data.get('motivation_embedding', None),
            parent=data.get('parent', None),
            summary=data.get('summary', ''),
            score=data.get('score', None),
            # Handle new fields with defaults
            name_new=data.get('name_new', ''),
            parameters=data.get('parameters', ''),
            svg_picture=data.get('svg_picture', '')
        )

    def __post_init__(self):
        """Post-initialization processing - ensures compatibility"""
        # Ensure required fields are not None
        if self.time is None:
            self.time = ''
        if self.name is None:
            self.name = ''
        if self.result is None:
            self.result = {}
        if self.program is None:
            self.program = ''
        if self.analysis is None:
            self.analysis = ''
        if self.cognition is None:
            self.cognition = ''
        if self.log is None:
            self.log = ''
        if self.motivation is None:
            self.motivation = ''
        if self.index is None:
            self.index = 0

    async def get_context(self) -> str:
        """
        Returns relevant context info for pipeline operations as a formatted string.
        Adjust this to include whatever fields your pipeline expects.
        """
        context_parts = []
        
        if self.name:
            context_parts.append(f"Name: {self.name}")
        
        if self.summary:
            context_parts.append(f"Summary: {self.summary}")
        
        if self.analysis:
            context_parts.append(f"Analysis: {self.analysis}")
        
        if self.cognition:
            context_parts.append(f"Cognition: {self.cognition}")
        
        if self.motivation:
            context_parts.append(f"Motivation: {self.motivation}")
        
        if self.parameters:
            context_parts.append(f"Parameters: {self.parameters}")
        
        if self.score is not None:
            context_parts.append(f"Score: {self.score}")
        
        context_parts.append(f"Index: {self.index}")
        
        return "\n".join(context_parts) + "\n"
import logging
import csv
import io
from typing import Dict, Any

logger = logging.getLogger(__name__)

def _evaluate_loss(result: dict) -> float:
    """
    Evaluates training loss

    Args:
        result: Result dictionary, containing the 'train' field

    Returns:
        float: The loss value of the last step.
    """
    if not result or not result.get('train'):
        logger.warning("training loss result is an empty string")
        return 0.0

    try:
        # First calculate test_score
        f = io.StringIO(result['train'])
        reader = csv.reader(f)
        # Skip the header
        header = next(reader)
        # Read the first data row
        values_list = next(reader)

        # Get the last value
        last_value = values_list[-1]  # Last step's loss

        if not last_value.strip():
            logger.warning("Loss value of the last step is empty")
            return 0.0

        try:
            return float(last_value)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert the loss value '{last_value}' of the last step to a float")
            return 0.0

    except StopIteration:
        logger.warning("CSV data is incomplete, missing data row")
        return 0.0
    except Exception as e:
        logger.error(f"An error occurred while processing CSV data: {e}")
        return 0.0

def _evaluate_result(result: dict) -> float:
    """
    Evaluates the result string (CSV format), returning the average score of all numerical values.
    Skips the header, calculates the average of the first data row.

    Args:
        result: Result dictionary, containing the 'test' field

    Returns:
        float: Mean of the test results.
    """
    if not result or not result.get('test'):
        logger.warning("benchmark evaluation result is an empty string")
        return 0.0

    try:
        # First calculate test_score
        f = io.StringIO(result['test'])
        reader = csv.reader(f)
        # Skip the header
        header = next(reader)
        # Read the first data row
        values_list = next(reader)
        scores = []

        # Start at the second column (skip the model name)
        for value in values_list[1:]:  # Skip first column for model name
            if not value.strip():
                continue
            try:
                scores.append(float(value))
            except (ValueError, TypeError):
                logger.warning(f"Cannot convert '{value}' to float, ignored in mean calculation")

        if not scores:
            logger.warning(f"No valid numerical values found in the result string: '{result}'")
            return 0.0

        return sum(scores) / len(scores)

    except StopIteration:
        logger.warning("CSV data is incomplete, missing data row")
        return 0.0
    except Exception as e:
        logger.error(f"Error in processing CSV data: {e}")
        return 0.0

def _has_data_rows(csv_string: str) -> bool:
    """
    Checks if the CSV string contains at least one data row (beyond the header).
    """
    if not isinstance(csv_string, str) or not csv_string.strip():
        return False

    # Use a with statement to ensure that io.StringIO is closed correctly.
    with io.StringIO(csv_string) as f:
        try:
            reader = csv.reader(f)
            _ = next(reader)  # Skip the header
            _ = next(reader)  # Try to read the first data row
            return True
        except StopIteration:
            # This means there's a header but no data rows.
            return False
        except Exception:
            # In case of other parsing errors, assume the data is valid to avoid accidental deletion.
            return True
