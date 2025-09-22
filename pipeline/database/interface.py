from typing import Tuple, Optional

from pipeline.config import Config
from .element import DataElement
from database.mongodb_database import create_mongo_database
from pipeline.utils.agent_logger import log_error

# Single global database instance
db = create_mongo_database()

async def program_sample() -> Tuple[str, Optional[int]]:
    """
    Sample a program node and build the context string.
    Returns (context, parent_index). Never returns None.
    """
    candidate_list = db.candidate_sample_from_range(1, 10, 1)
    if not candidate_list:
        log_error("candidate_sample_from_range returned empty list!")
        return "", None

    parent_element = candidate_list[0]
    ref_elements = db.candidate_sample_from_range(11, 50, 4)

    # Build context
    context = ""
    try:
        context += await parent_element.get_context()
        for elem in ref_elements:
            context += await elem.get_context()
    except Exception as e:
        log_error(f"Error building context: {e}")
        return "", None

    parent = parent_element.index

    # Write the program to the configured source file
    try:
        with open(Config.SOURCE_FILE, 'w', encoding='utf-8') as f:
            f.write(parent_element.program)
    except Exception as e:
        log_error(f"Failed to write program to SOURCE_FILE: {e}")

    return context, parent

def update(result: DataElement) -> bool:
    """
    Insert a new experiment result into MongoDB.
    """
    try:
        db.add_element_from_dict(result.to_dict())
        return True
    except Exception as e:
        log_error(f"Failed to update database: {e}")
        return False
