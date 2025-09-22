import os
import subprocess
import asyncio
from typing import Tuple
from pipeline.config import Config
from pipeline.utils.agent_logger import log_agent_run
from .model import debugger, trainer
from .prompts import Debugger_input

async def evaluation(name: str, motivation: str) -> bool:
    """
    Evaluate training performance for a given experiment.
    
    Args:
        name: Experiment name
        motivation: Experiment motivation
        
    Returns:
        True if training successful, False otherwise
    """
    success, error_msg = await run_training(name, motivation)
    
    if not success:
        print(f"Training failed: {error_msg}")
        return False
    
    save(name)
    return True

async def run_training(name: str, motivation: str) -> Tuple[bool, str]:
    """
    Run Training script with debugging retry mechanism.
    Uses local execution fallback when agent fails.
    
    Args:
        name: Experiment name
        motivation: Experiment motivation
        
    Returns:
        Tuple of (success_flag, error_message)
    """
    try:
        debug = False
        previous_error = ""
        
        for attempt in range(Config.MAX_DEBUG_ATTEMPT):
            if debug:
                debug_result = await log_agent_run(
                    "debugger",
                    debugger,
                    Debugger_input(motivation, previous_error)
                )
                
                changes_made = debug_result.final_output.changes_made
                print(f"Debug changes for {name}: {changes_made}")
            
            # Try agent-based training first
            try:
                train_result = await log_agent_run(
                    "trainer",
                    trainer,
                    f"""Please run the training script:
- Execute bash {Config.BASH_SCRIPT} with parameter: {name}
- Only return success=True if script exits with code 0"""
                )
                
                if train_result.final_output.success:
                    print(f"Training successful for {name} (via agent)")
                    return True, ""
                else:
                    print(f"Agent training failed, trying local execution...")
                    # Fall back to local execution
                    success, error = await run_training_local(name)
                    if success:
                        print(f"Training successful for {name} (via local fallback)")
                        return True, ""
                    else:
                        print(f"Local training also failed: {error}")
                        
            except Exception as agent_error:
                print(f"Agent execution failed: {agent_error}, trying local execution...")
                success, error = await run_training_local(name)
                if success:
                    print(f"Training successful for {name} (via local fallback)")
                    return True, ""
                else:
                    print(f"Local training failed: {error}")
            
            # Set debug flag and prepare error message for next iteration
            debug = True
            
            # Read debug file content as detailed error information
            try:
                if not os.path.exists(Config.DEBUG_FILE):
                    with open(Config.DEBUG_FILE, 'w', encoding='utf-8') as f:
                        f.write("")
                
                with open(Config.DEBUG_FILE, 'r', encoding='utf-8') as f:
                    debug_content = f.read()
                
                previous_error = f"Training failed. Debug info:\n{debug_content}"
                
            except Exception as e:
                previous_error = (
                    f"Training failed. Cannot read debug file {Config.DEBUG_FILE}: {str(e)}"
                )
            
            print(f"Training failed for {name} (attempt {attempt + 1}): {previous_error}")
            
            # If this is the last attempt, return failure
            if attempt == Config.MAX_DEBUG_ATTEMPT - 1:
                return False, (
                    f"Training failed after {Config.MAX_DEBUG_ATTEMPT} attempts. "
                    f"Final error: {previous_error}"
                )
            
            continue
            
    except Exception as e:
        error_msg = f"Unexpected error during training: {str(e)}"
        print(error_msg)
        return False, error_msg

async def run_training_local(name: str) -> Tuple[bool, str]:
    """
    Run training script locally as fallback when agent execution fails.
    
    Args:
        name: Experiment name
        
    Returns:
        Tuple of (success_flag, error_message)
    """
    try:
        # Ensure debug directory exists
        os.makedirs(os.path.dirname(Config.DEBUG_FILE), exist_ok=True)
        
        # Check if experiment file exists first
        config = Config()
        try:
            experiment_path = config.find_experiment_script(name)
            print(f"Found experiment script: {experiment_path}")
        except FileNotFoundError as e:
            error_msg = str(e)
            with open(Config.DEBUG_FILE, 'w', encoding='utf-8') as f:
                f.write(error_msg)
            return False, error_msg
        
        # Execute the training script locally
        cmd = ["bash", Config.BASH_SCRIPT, name]
        print(f"Executing locally: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Write output to debug file
        debug_content = f"Command: {' '.join(cmd)}\n"
        debug_content += f"Exit code: {result.returncode}\n"
        debug_content += f"STDOUT:\n{result.stdout}\n"
        debug_content += f"STDERR:\n{result.stderr}\n"
        
        with open(Config.DEBUG_FILE, 'w', encoding='utf-8') as f:
            f.write(debug_content)
        
        if result.returncode == 0:
            return True, ""
        else:
            error_msg = f"Local execution failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f"\nSTDERR: {result.stderr}"
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = "Training script timed out after 5 minutes"
        with open(Config.DEBUG_FILE, 'w', encoding='utf-8') as f:
            f.write(error_msg)
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Local execution error: {str(e)}"
        with open(Config.DEBUG_FILE, 'w', encoding='utf-8') as f:
            f.write(error_msg)
        return False, error_msg

def save(name: str) -> None:
    """
    Save source file content to code pool (pipeline/pool) with given name.
    
    Args:
        name: File name to save as
    """
    source_file = getattr(Config, 'SOURCE_FILE', 'pipeline/base_program.py')
    
    with open(source_file, "r", encoding='utf-8') as f:
        content = f.read()
    
    with open(f"{Config.CODE_POOL}/{name}.py", "w", encoding='utf-8') as f:
        f.write(content)
