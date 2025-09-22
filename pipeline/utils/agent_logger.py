import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agents import Runner


class AgentLogger:
    """Agent call logger."""
    
    def __init__(self, log_dir: str = "pipline/logs/agent_calls"):
        """
        Initialize logger.
        
        Args:
            log_dir: Log file storage directory
        """
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main log file
        self.main_log_file = self.log_dir / "agent_calls.log"
        
        # Pipeline related
        self.current_pipeline_id: Optional[str] = None
        self.current_pipeline_dir: Optional[Path] = None
        self.pipeline_log_file: Optional[Path] = None
        self.pipeline_full_log_file: Optional[Path] = None  # Full log file
    
    def start_pipeline(self, pipeline_name: str = "") -> str:
        """
        Start a new pipeline process.
        
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
        
        # Create pipeline dedicated directory
        self.current_pipeline_dir = self.log_dir / self.current_pipeline_id
        self.current_pipeline_dir.mkdir(exist_ok=True)
        
        # Create pipeline log files
        self.pipeline_log_file = self.current_pipeline_dir / "pipeline.log"
        self.pipeline_full_log_file = self.current_pipeline_dir / "full.log"
        
        # Record pipeline start
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
        End current pipeline process.
        
        Args:
            success: Whether pipeline completed successfully
            summary: Pipeline summary information
        """
        if not self.current_pipeline_id:
            print("Warning: No active pipeline to end")
            return
        
        # Get pipeline usage statistics
        pipeline_usage = self._get_pipeline_usage_internal()
        
        # Record pipeline end
        pipeline_end_log = {
            "pipeline_id": self.current_pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if success else "failed",
            "summary": summary,
            "usage_summary": pipeline_usage
        }
        
        self._write_pipeline_log(pipeline_end_log)
        
        # Record pipeline end and usage information
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
        Log complete agent call and execute.
        
        Args:
            agent_name: Agent name
            agent: Agent object
            input_data: Input data
            **kwargs: Other parameters passed to Runner.run
            
        Returns:
            Agent call result
        """
        call_id = self._generate_call_id()
        timestamp = datetime.now().isoformat()
        
        # Record call start
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
            self._write_pipeline_log(start_log)  # Also write to pipeline log
        
        try:
            # Execute agent call
            result = await Runner.run(agent, input=input_data, **kwargs)
            
            # Extract usage information from agents library return result
            usage_info = self._extract_usage_from_result(result)
            
            # Record success result
            success_log = {
                "call_id": call_id,
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "status": "completed",
                "output": self._serialize_data(result.final_output if hasattr(result, 'final_output') else result),
                "full_result": self._serialize_data(result, max_depth=2),  # Limit depth to avoid being too large
                "pipeline_id": self.current_pipeline_id,  # Add pipeline_id
                "usage": usage_info  # Add usage information extracted from result
            }
            
            self._write_log(success_log)
            if self.current_pipeline_id:
                self._write_pipeline_log(success_log)  # Also write to pipeline log
            
            # Create separate detailed log file
            self._create_detailed_log(call_id, agent_name, start_log, success_log)
            
            return result
            
        except Exception as e:
            # Record error
            error_log = {
                "call_id": call_id,
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "pipeline_id": self.current_pipeline_id,  # Add pipeline_id
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}  # Usage is 0 on error
            }
            
            self._write_log(error_log)
            if self.current_pipeline_id:
                self._write_pipeline_log(error_log)  # Also write to pipeline log
            
            # Create error detailed log file
            self._create_detailed_log(call_id, agent_name, start_log, error_log)
            
            # Re-raise exception
            raise
    
    def _generate_call_id(self) -> str:
        """Generate unique call ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"call_{timestamp}"
    
    def _extract_usage_from_result(self, result: Any) -> Dict[str, int]:
        """
        Extract usage information from agents.Runner return result.
        Get original API response usage data from raw_responses.
        
        Args:
            result: Return result from Runner.run
            
        Returns:
            Dictionary containing usage information
        """
        default_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        # First check if there is raw_responses attribute
        if hasattr(result, 'raw_responses'):
            raw_responses = result.raw_responses
            
            if raw_responses:
                # Accumulate usage information from all responses
                total_input_tokens = 0
                total_output_tokens = 0
                total_tokens = 0
                
                # If raw_responses is a list, iterate through all responses
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
        
        # Check if there is direct usage attribute as fallback
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
        Extract usage information from single API response.
        
        Args:
            response: Single API response object
            
        Returns:
            Dictionary containing usage information, returns None if not found
        """
        if hasattr(response, 'usage'):
            usage_obj = response.usage
            
            if usage_obj:
                # If usage_obj is a dictionary
                if isinstance(usage_obj, dict):
                    return {
                        "input_tokens": usage_obj.get("input_tokens", usage_obj.get("prompt_tokens", 0)),
                        "output_tokens": usage_obj.get("output_tokens", usage_obj.get("completion_tokens", 0)),
                        "total_tokens": usage_obj.get("total_tokens", 0)
                    }
                
                # If usage_obj is an object, try to get attributes
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
        """
        Internal method: Get current pipeline usage.
        
        Returns:
            Current pipeline usage statistics
        """
        if not self.current_pipeline_id or not self.pipeline_log_file or not self.pipeline_log_file.exists():
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        try:
            pipeline_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            
            with open(self.pipeline_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_data = json.loads(line.strip())
                        
                        # Only accumulate usage from final 'completed' log entries
                        if log_data.get("status") == "completed":
                            usage_data = log_data.get("usage", {})
                            if usage_data:
                                pipeline_usage["input_tokens"] += usage_data.get("input_tokens", 0)
                                pipeline_usage["output_tokens"] += usage_data.get("output_tokens", 0)
                                pipeline_usage["total_tokens"] += usage_data.get("total_tokens", 0)
                                
                    except json.JSONDecodeError:
                        continue
            
            return pipeline_usage
            
        except Exception as e:
            print(f"Failed to get pipeline usage: {e}")
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    def _serialize_data(self, data: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
        """
        Serialize data to JSON-serializable format.
        
        Args:
            data: Data to serialize
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
        """
        if current_depth > max_depth:
            return "<max_depth_reached>"
        if data is None:
            return None
        elif isinstance(data, (str, int, float, bool)):
            return data
        elif isinstance(data, (list, tuple)):
            return [self._serialize_data(item, max_depth, current_depth + 1) for item in data[:10]]  # Limit list length
        # --- ROBUST FIX ---
        elif isinstance(data, dict) or hasattr(data, 'items'):
            try:
                # objects with .items() method
                items = list(data.items())[:20]
            except Exception:
                items = []
            return {k: self._serialize_data(v, max_depth, current_depth + 1) for k, v in items}
        elif hasattr(data, "__dict__"):
            items = list(vars(data).items())[:20]
            return {k: self._serialize_data(v, max_depth, current_depth + 1) for k, v in items}
        else:
            return self._serialize_object(data, max_depth, current_depth)

        
            
    def _serialize_object(self, data: Any, max_depth: int, current_depth: int) -> Any:
        """Serialize complex objects."""
        try:
            # 1. Check if it's Pydantic BaseModel
            if hasattr(data, 'model_dump'):
                # Pydantic v2
                try:
                    model_data = data.model_dump()
                    return {
                        "_type": "PydanticModel",
                        "_class": type(data).__name__,
                        "_module": getattr(type(data), '__module__', 'unknown'),
                        "data": self._serialize_data(model_data, max_depth, current_depth + 1)
                    }
                except Exception:
                    pass
            
            # 2. Check if it's Pydantic BaseModel (v1)
            if hasattr(data, 'dict') and hasattr(data, '__fields__'):
                # Pydantic v1
                try:
                    model_data = data.dict()
                    return {
                        "_type": "PydanticModel",
                        "_class": type(data).__name__,
                        "_module": getattr(type(data), '__module__', 'unknown'),
                        "data": self._serialize_data(model_data, max_depth, current_depth + 1)
                    }
                except Exception:
                    pass
            
            # 3. Check if it has to_dict method
            if hasattr(data, 'to_dict') and callable(getattr(data, 'to_dict')):
                try:
                    dict_data = data.to_dict()
                    return {
                        "_type": "ObjectWithToDict",
                        "_class": type(data).__name__,
                        "_module": getattr(type(data), '__module__', 'unknown'),
                        "data": self._serialize_data(dict_data, max_depth, current_depth + 1)
                    }
                except Exception:
                    pass
            
            # 4. Check if it's dataclass
            if hasattr(data, '__dataclass_fields__'):
                try:
                    from dataclasses import asdict
                    dict_data = asdict(data)
                    return {
                        "_type": "Dataclass", 
                        "_class": type(data).__name__,
                        "_module": getattr(type(data), '__module__', 'unknown'),
                        "data": self._serialize_data(dict_data, max_depth, current_depth + 1)
                    }
                except Exception:
                    pass
            
            # 5. Check if it's regular object with __dict__ attribute
            if hasattr(data, '__dict__'):
                try:
                    return {
                        "_type": "Object",
                        "_class": type(data).__name__,
                        "_module": getattr(type(data), '__module__', 'unknown'),
                        "data": {k: self._serialize_data(v, max_depth, current_depth + 1) 
                               for k, v in data.__dict__.items() if not k.startswith('_')}
                    }
                except Exception:
                    pass
            
            # 6. Try JSON serialization
            try:
                json.dumps(data)
                return data
            except (TypeError, ValueError):
                pass
            
            # 7. Finally try to convert to string
            try:
                str_repr = str(data)
                return {
                    "_type": "String",
                    "_class": type(data).__name__,
                    "_module": getattr(type(data), '__module__', 'unknown'),
                    "data": str_repr[:1000] + ("..." if len(str_repr) > 1000 else "")
                }
            except Exception:
                return {
                    "_type": "UnserializableObject",
                    "_class": type(data).__name__,
                    "_module": getattr(type(data), '__module__', 'unknown'),
                    "error": "Cannot serialize this object"
                }
                
        except Exception as e:
            return {
                "_type": "SerializationError",
                "_class": type(data).__name__ if hasattr(data, '__class__') else 'unknown',
                "error": str(e)
            }
    
    def _write_log(self, log_data: Dict[str, Any]) -> None:
        """Write log to main log file."""
        try:
            with open(self.main_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False, indent=None) + '\n')
        except Exception as e:
            print(f"Failed to write log: {e}")
    
    def _create_detailed_log(self, call_id: str, agent_name: str, start_log: Dict, end_log: Dict) -> None:
        """Create detailed single call log file."""
        try:
            detailed_log = {
                "call_summary": {
                    "call_id": call_id,
                    "agent_name": agent_name,
                    "start_time": start_log["timestamp"],
                    "end_time": end_log["timestamp"],
                    "status": end_log["status"],
                    "pipeline_id": self.current_pipeline_id
                },
                "start_log": start_log,
                "end_log": end_log
            }
            
            filename = f"{call_id}_{agent_name}.json"
            
            # If there's an active pipeline, save to pipeline directory, otherwise save to general directory
            if self.current_pipeline_id and self.current_pipeline_dir:
                detailed_file = self.current_pipeline_dir / filename
            else:
                detailed_file = self.log_dir / "detailed" / filename
                detailed_file.parent.mkdir(exist_ok=True)
            
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_log, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Failed to create detailed log: {e}")
    
    def _write_pipeline_log(self, log_data: Dict[str, Any]) -> None:
        """Write log to pipeline dedicated log file."""
        try:
            if self.pipeline_log_file:
                with open(self.pipeline_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_data, ensure_ascii=False, indent=None) + '\n')
        except Exception as e:
            print(f"Failed to write pipeline log: {e}")
    
    def _write_full_log(self, message: str, level: str = "INFO") -> None:
        """Write full log to pipeline full log file."""
        try:
            if self.pipeline_full_log_file:
                timestamp = datetime.now().isoformat()
                log_entry = f"[{timestamp}] [{level}] {message}\n"
                with open(self.pipeline_full_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
            # Also output to console
            print(f"[{level}] {message}")
        except Exception as e:
            print(f"Failed to write full log: {e}")
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self._write_full_log(message, "INFO")
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self._write_full_log(message, "WARNING")
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self._write_full_log(message, "ERROR")
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self._write_full_log(message, "DEBUG")
    
    def log_step(self, step_name: str, message: str = "") -> None:
        """Log step message."""
        full_message = f"=== {step_name} ===" + (f" {message}" if message else "")
        self._write_full_log(full_message, "STEP")
    
    def get_agent_call_stats(self) -> Dict[str, Any]:
        """Get agent call statistics."""
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
            
            # Use dictionary to merge 'started' and 'completed'/'failed' logs for each call_id
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
                        
                        # Usage statistics by agent
                        if agent_name not in stats["usage"]["by_agent"]:
                            stats["usage"]["by_agent"][agent_name] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                        
                        stats["usage"]["by_agent"][agent_name]["input_tokens"] += input_tokens
                        stats["usage"]["by_agent"][agent_name]["output_tokens"] += output_tokens
                        stats["usage"]["by_agent"][agent_name]["total_tokens"] += total_tokens

            return stats
            
        except Exception as e:
            print(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Create global logger instance
_global_logger = None

def get_logger(log_dir: str = "logs/agent_calls") -> AgentLogger:
    """Get global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = AgentLogger("pipeline/logs/agent_calls")
    return _global_logger

async def log_agent_run(agent_name: str, agent, input_data: Any = None, **kwargs) -> Any:
    """
    Convenience function: Log and execute agent call.
    
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
    Convenience function: Start a new pipeline process.
    
    Args:
        pipeline_name: Pipeline name (optional)
        
    Returns:
        pipeline_id: Generated pipeline ID
    """
    logger = get_logger()
    return logger.start_pipeline(pipeline_name)

def end_pipeline(success: bool = True, summary: str = "") -> None:
    """
    Convenience function: End current pipeline process.
    
    Args:
        success: Whether pipeline completed successfully
        summary: Pipeline summary information
    """
    logger = get_logger()
    logger.end_pipeline(success, summary)

def get_current_pipeline_id() -> Optional[str]:
    """
    Convenience function: Get current pipeline ID.
    
    Returns:
        Current pipeline ID, returns None if no active pipeline
    """
    logger = get_logger()
    return logger.current_pipeline_id

def log_info(message: str) -> None:
    """Convenience function: Log info message."""
    logger = get_logger()
    logger.log_info(message)

def log_warning(message: str) -> None:
    """Convenience function: Log warning message."""
    logger = get_logger()
    logger.log_warning(message)

def log_error(message: str) -> None:
    """Convenience function: Log error message."""
    logger = get_logger()
    logger.log_error(message)

def log_debug(message: str) -> None:
    """Convenience function: Log debug message."""
    logger = get_logger()
    logger.log_debug(message)

def log_step(step_name: str, message: str = "") -> None:
    """Convenience function: Log step message."""
    logger = get_logger()
    logger.log_step(step_name, message)

def get_usage_stats() -> Dict[str, Any]:
    """
    Convenience function: Get usage statistics.
    
    Returns:
        Statistics dictionary containing usage information
    """
    logger = get_logger()
    stats = logger.get_agent_call_stats()
    return stats.get("usage", {})

def get_current_pipeline_usage() -> Dict[str, int]:
    """
    Convenience function: Get current pipeline usage.
    
    Returns:
        Current pipeline usage statistics
    """
    logger = get_logger()
    return logger._get_pipeline_usage_internal()

def log_usage_summary() -> None:
    """Convenience function: Print current usage summary in log."""
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
