#!/usr/bin/env python3
import os
import asyncio
import traceback
import torch
import functools
# ──────────────────────────────────────────────────────────────────────────────
# Globally wrap every Module.forward with MPS‐fallback + dtype adjustment
def mps_fallback(fn):
    @functools.wraps(fn)
    def wrapper(module, x, *args, **kwargs):
        # Cast FP16→FP32 on MPS if needed
        if x.device.type == 'mps' and x.dtype == torch.float16:
            x = x.to(torch.float32)
        try:
            return fn(module, x, *args, **kwargs)
        except RuntimeError as e:
            if 'mps' in str(e).lower():
                return fn(module, x.to('cpu'), *args, **kwargs)
            raise
    return wrapper

# Apply it before any model.forward is called
torch.nn.Module.forward = mps_fallback(torch.nn.Module.forward)
# ──────────────────────────────────────────────────────────────────────────────




from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncAzureOpenAI

from pipeline.analyse import analyse
from pipeline.database.interface import program_sample, update
from pipeline.eval import evaluation
from pipeline.evolve import evolve
from pipeline.utils.agent_logger import (
    start_pipeline,
    log_step,
    log_info,
    log_error,
    log_warning,
    end_pipeline,
)

from database.mongodb_database import MongoDatabase

# Load Azure OpenAI settings from environment
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("OPENAI_API_VERSION")

# Create and register the Azure client
client = AsyncAzureOpenAI(
    api_key=api_key,
    base_url=endpoint,
    api_version=api_version
)
set_default_openai_client(client)
set_default_openai_api("chat_completions")

async def ensure_candidates_synced():
    """Rebuild candidate set from DB before experiments."""
    db = MongoDatabase()
    result = await db.rebuild_candidates_from_scored_elements()
    log_info(f"Candidate set rebuilt from MongoDB: {result}")

def check_mps():
    """Check for Apple Silicon MPS support."""
    if torch.backends.mps.is_available():
        log_info("MPS backend is available (Apple Silicon GPU detected by PyTorch)!")
        if torch.backends.mps.is_built():
            log_info("MPS backend is built into PyTorch.")
        else:
            log_warning("MPS backend is NOT built into PyTorch (unexpected).")
    else:
        log_warning("MPS backend is NOT available. PyTorch is running on CPU.")

async def run_single_experiment() -> bool:
    """Run a single experiment loop with categorized logging."""
    pipeline_id = start_pipeline("experiment")
    try:
        # Step 1: Program sampling
        log_step("Program Sampling", "Start sampling program from database")
        result = await program_sample()
        log_info(f"Raw program_sample() result: {repr(result)}")

        # Handle tuple unpacking from program_sample
        if isinstance(result, tuple) and len(result) == 2:
            context, parent = result
        else:
            context, parent = result, None
            
        if not context:
            log_error("No context returned from program_sample. Exiting step.")
            end_pipeline(False, "Sampling failed")
            return False

        log_info(f"Program sampling completed, context length: {len(str(context))}")

        # Step 2: Evolution
        log_step("Program Evolution", "Start evolving new program")
        name, motivation = await evolve(context)
        if name == "Failed":
            log_error("Program evolution failed")
            end_pipeline(False, "Evolution failed")
            return False
        log_info(f"Program evolution successful, generated program: {name}")
        log_info(f"Evolution motivation: {motivation}")

        # Step 3: Evaluation
        log_step("Program Evaluation", f"Start evaluating program {name}")
        success = await evaluation(name, motivation)
        if not success:
            log_error(f"Program {name} evaluation failed")
            end_pipeline(False, "Evaluation failed")
            return False
        log_info(f"Program {name} evaluation successful")

        # Step 4: Analysis
        log_step("Result Analysis", f"Start analyzing program {name} results")
        analysis_result = await analyse(name, motivation, parent=parent)
        log_info(f"Analysis completed, result: {analysis_result}")

        # Step 5: Update database
        log_step("Database Update", "Update results to database")
        update(analysis_result)
        log_info("Database update completed")

        # Experiment completed
        log_info("Experiment pipeline completed successfully")
        end_pipeline(True, f"Experiment completed successfully, program: {name}, result: {analysis_result}")
        return True

    except KeyboardInterrupt:
        log_warning("User interrupted experiment")
        log_error(f"Traceback: {traceback.format_exc()}")
        end_pipeline(False, "User interrupted experiment")
        return False

    except Exception as e:
        log_error(f"Experiment pipeline unexpected error: {e}")
        log_error(f"Traceback: {traceback.format_exc()}")
        end_pipeline(False, f"Unexpected error: {e}")
        return False

async def main():
    """Main function: continuous experiment execution."""
    set_tracing_disabled(True)
    check_mps()

    # Sync candidate set
    await ensure_candidates_synced()

    log_info("Starting continuous experiment pipeline...")
    experiment_count = 0
    while True:
        try:
            experiment_count += 1
            log_info(f"Starting experiment {experiment_count}")
            success = await run_single_experiment()
            if success:
                log_info(f"Experiment {experiment_count} completed successfully.")
            else:
                log_warning(f"Experiment {experiment_count} failed, retrying in 60 seconds.")
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            log_warning("Continuous experiment interrupted by user")
            log_error(f"Traceback: {traceback.format_exc()}")
            break
        except Exception as e:
            log_error(f"Main loop unexpected error: {e}")
            log_error(f"Traceback: {traceback.format_exc()}")
            log_info("Retrying in 60 seconds...")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
