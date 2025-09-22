from .prompt import Planner_input, Motivation_checker_input, Deduplication_input, CodeChecker_input
from .model import planner, motivation_checker, deduplication, code_checker
from agents import exceptions, set_tracing_disabled
from typing import List, Tuple
from pipeline.config import Config
from pipeline.database.mongo_database import create_client
from pipeline.utils.agent_logger import log_agent_run
import traceback  # For better error reporting
import logging
import os

# -------------------------------------------------------------------------
# Ensure that all experiment scripts are saved to pipeline/pool/ directory.
# -------------------------------------------------------------------------
# Define the pool directory and ensure it exists on module load
POOL_DIR = os.path.join(os.getcwd(), "pipeline", "pool")
os.makedirs(POOL_DIR, exist_ok=True)

# Set up detailed logging for all orchestration and audit events.
logging.basicConfig(
    filename="experiment_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# NOTE:
# For per-epoch/per-evaluation metrics and diagnostics (loss, accuracy, memory, compute usage),
# instrumentation must be added to your main training loop (in evolve.py or similar) --
# NOT in this orchestration/interface script.

async def evolve(context: str) -> Tuple[str, str]:
    logging.info("[EVOLVE] Starting evolve with context length %d", len(context) if context else 0)
    
    # Store the original Config.SOURCE_FILE to restore later if needed
    original_source_file = Config.SOURCE_FILE
    
    for attempt in range(getattr(Config, 'MAX_RETRY_ATTEMPTS', 3)):
        # Read the current source file (using the original path initially)
        with open(original_source_file, 'r') as f:
            original_source = f.read()

        logging.info("[EVOLVE] Attempt %d/%d: Generating model/motivation", attempt + 1, getattr(Config, 'MAX_RETRY_ATTEMPTS', 3))
        print(f"DEBUG: SOURCE_FILE before generation: {len(original_source)} chars")

        # Generate a new experiment name and motivation
        name, motivation = await gen(context)

        # NOW set up the experiment file path in pool/
        experiment_file = os.path.join(POOL_DIR, f"{name}.py")
        os.makedirs(os.path.dirname(experiment_file), exist_ok=True)
        
        # Copy the original source to the new experiment file
        with open(experiment_file, 'w') as f:
            f.write(original_source)
        
        # Override Config.SOURCE_FILE so write_code_file and file writes go into pool/
        Config.SOURCE_FILE = experiment_file

        # DEBUG: Check the new source file after generation
        with open(Config.SOURCE_FILE, 'r') as f:
            new_source = f.read()
        print(f"DEBUG: SOURCE_FILE after generation: {len(new_source)} chars")
        print(f"DEBUG: File changed: {new_source != original_source}")
        print(f"DEBUG: Contains CPAGHR: {'CPAGHR' in new_source}")

        # Save log of generated motivation and code for auditing
        logging.info(
            "[EVOLVE][Attempt %d] Motivation name: %s\nMotivation:\n%s",
            attempt + 1, name, motivation
        )

        try:
            if await check_code_correctness(motivation):
                logging.info("[EVOLVE][Attempt %d] Motivation/code passed code correctness check.", attempt + 1)
                return name, motivation
            else:
                logging.warning("[EVOLVE][Attempt %d] Motivation/code did NOT pass code correctness check.", attempt + 1)
        except Exception as e:
            logging.error(
                "[EVOLVE][Attempt %d] Exception during code correctness check.\nMotivation: %s\nException: %s\nTraceback: %s",
                attempt + 1, motivation, str(e), traceback.format_exc()
            )

        # Rollback file on failure - restore original source to the experiment file
        try:
            with open(Config.SOURCE_FILE, 'w') as f:
                f.write(original_source)
            logging.info("[EVOLVE][Attempt %d] SOURCE_FILE rolled back to original after failure.", attempt + 1)
        except Exception as e:
            logging.error(
                "[EVOLVE][Attempt %d] Error rolling back SOURCE_FILE: %s\nTraceback: %s",
                attempt + 1, str(e), traceback.format_exc()
            )

        print("Try new motivations")

    # Restore original Config.SOURCE_FILE if all attempts failed
    Config.SOURCE_FILE = original_source_file
    logging.error("[EVOLVE] All attempts failed. Returning 'Failed', 'evolve error'")
    return "Failed", "evolve error"


async def gen(context: str) -> Tuple[str, str]:
    logging.info("[GEN] Starting generation with context length %d", len(context) if context else 0)
    
    # Read from the current Config.SOURCE_FILE (which should be the original at this point)
    with open(Config.SOURCE_FILE, 'r') as f:
        original_source = f.read()

    repeated_result = None
    motivation = None

    for attempt in range(getattr(Config, 'MAX_RETRY_ATTEMPTS', 3)):
        try:
            # Restore original file before new attempt (to current Config.SOURCE_FILE)
            with open(Config.SOURCE_FILE, 'w') as f:
                f.write(original_source)
            logging.info("[GEN][Attempt %d] SOURCE_FILE restored before generation.", attempt + 1)

            # Run planner or deduplication as needed
            if attempt == 0:
                input = Planner_input(context)
                stage = "planner"
            else:
                repeated_context = await get_repeated_context(repeated_result.repeated_index)
                input = Deduplication_input(context, repeated_context)
                stage = "deduplication"

            logging.info("[GEN][Attempt %d] Running %s agent...", attempt + 1, stage)
            plan = await log_agent_run(
                stage,
                planner if stage == "planner" else deduplication,
                input
            )

            # Defensive check:
            plan_result = plan.final_output
            if isinstance(plan_result, dict):
                print("Warning: plan.final_output is a dict, expected an object/class. Keys:", plan_result.keys())
            else:
                print("plan.final_output is of type:", type(plan_result))

            name, motivation = plan.final_output.name, plan.final_output.motivation

            logging.info("[GEN][Attempt %d] Generated plan. Name: %s\nMotivation:\n%s", attempt + 1, name, motivation)

            repeated_result = await check_repeated_motivation(motivation)
            if repeated_result.is_repeated:
                msg = f"Attempt {attempt + 1}: Motivation repeated, index is {repeated_result.repeated_index}"
                print(msg)
                logging.warning("[GEN][Attempt %d] %s", attempt + 1, msg)
                if attempt == getattr(Config, 'MAX_RETRY_ATTEMPTS', 3) - 1:
                    logging.error("[GEN][Attempt %d] Maximum retry attempts reached, unable to generate non-repeated motivation.", attempt + 1)
                    raise Exception("Maximum retry attempts reached, unable to generate non-repeated motivation")
                continue
            else:
                print(f"Attempt {attempt + 1}: Motivation not repeated, continue execution")
                print(motivation)
                logging.info("[GEN][Attempt %d] Motivation accepted:\n%s", attempt + 1, motivation)
                return name, motivation

        except exceptions.MaxTurnsExceeded as e:
            msg = f"Attempt {attempt + 1} exceeded maximum dialogue turns"
            print(msg)
            logging.warning("[GEN][Attempt %d] %s\nException: %s", attempt + 1, msg, str(e))
        except Exception as e:
            msg = f"Attempt {attempt + 1} error: {e}"
            print(msg)
            logging.error("[GEN][Attempt %d] Exception: %s\nTraceback: %s", attempt + 1, str(e), traceback.format_exc())
            raise e


async def check_code_correctness(motivation) -> bool:
    """Check code correctness and log all results and errors."""
    logging.info("[CODE CHECKER] Checking code correctness.\nMotivation:\n%s", motivation)
    for attempt in range(getattr(Config, 'MAX_RETRY_ATTEMPTS', 3)):
        try:
            code_checker_result = await log_agent_run(
                "code_checker",
                code_checker,
                CodeChecker_input(motivation=motivation),
                max_turns=100
            )
            if code_checker_result.final_output.success:
                print("Code checker passed - code looks correct")
                logging.info("[CODE CHECKER][Attempt %d] Code checker passed.", attempt + 1)
                return True
            else:
                error_msg = code_checker_result.final_output.error
                print(f"Code checker found issues: {error_msg}")
                logging.warning("[CODE CHECKER][Attempt %d] Failure. Error: %s\nInput motivation:\n%s", attempt + 1, error_msg, motivation)
                if attempt == getattr(Config, 'MAX_RETRY_ATTEMPTS', 3) - 1:
                    print("Reaching checking limits")
                    logging.error("[CODE CHECKER][Attempt %d] Checking limits reached - marking as failed.\nMotivation:\n%s", attempt + 1, motivation)
                    return False
                continue
        except exceptions.MaxTurnsExceeded as e:
            print("Code checker exceeded maximum turns")
            logging.warning("[CODE CHECKER][Attempt %d] MaxTurnsExceeded Exception: %s\nMotivation:\n%s", attempt + 1, str(e), motivation)
            return False
        except Exception as e:
            print(f"Code checker error: {e}")
            logging.error("[CODE CHECKER][Attempt %d] Exception: %s\nTraceback:%s\nMotivation:\n%s", attempt + 1, str(e), traceback.format_exc(), motivation)
            return False


async def check_repeated_motivation(motivation: str):
    logging.info("[CHECK REPETITION] Checking repetition for motivation:\n%s", motivation)
    client = create_client()
    similar_elements = client.search_similar_motivations(motivation)
    context = similar_motivation_context(similar_elements)
    input = Motivation_checker_input(context, motivation)
    repeated_result = await log_agent_run(
        "motivation_checker",
        motivation_checker,
        input
    )
    logging.info("[CHECK REPETITION] Motivation repetition check result: %s", repeated_result.final_output)
    return repeated_result.final_output


def similar_motivation_context(similar_elements: list) -> str:
    if not similar_elements:
        logging.info("[SIMILAR MOTIVATION CONTEXT] No previous motivations found.")
        return "No previous motivations found for comparison."

    context = "### PREVIOUS RESEARCH MOTIVATIONS\n\n"
    for i, element in enumerate(similar_elements, 1):
        context += f"**Reference #{i} (Index: {element.index})**\n``````\n\n"
    context += f"**Total Previous Motivations**: {len(similar_elements)}\n"
    context += "**Analysis Scope**: Compare target motivation against each reference above\n"
    logging.info("[SIMILAR MOTIVATION CONTEXT] Generated context for %d elements.", len(similar_elements))
    return context


async def get_repeated_context(repeated_index: list[int]) -> str:
    client = create_client()
    repeated_elements = [client.get_elements_by_index(index) for index in repeated_index]
    if not repeated_elements:
        logging.info("[REPEATED CONTEXT] No repeated experimental context found.")
        return "No repeated experimental context available."

    structured_context = "### REPEATED EXPERIMENTAL PATTERNS ANALYSIS\n\n"
    for i, element in enumerate(repeated_elements, 1):
        structured_context += f"**Experiment #{i} - Index {element.index}**\n``````\n\n"
    structured_context += f"- **Total Repeated Experiments**: {len(repeated_elements)}\n"
    structured_context += f"- **Innovation Challenge**: Break free from these established pattern spaces\n"
    structured_context += f"- **Differentiation Requirement**: Implement orthogonal approaches that explore fundamentally different design principles\n\n"
    structured_context += f"Key Insight: The above experiments represent exhausted design spaces. Your task is to identify and implement approaches that operate on completely different mathematical, biological, or physical principles to achieve breakthrough innovation.\n"
    logging.info("[REPEATED CONTEXT] Generated repeated context for %d elements.", len(repeated_elements))
    return structured_context

# End of orchestration script.

# Reminder: Per-epoch/per-step experiment metrics must be logged in the training loop (not here).
