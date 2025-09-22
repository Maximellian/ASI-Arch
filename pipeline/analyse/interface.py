import csv
import io
import re
from datetime import datetime
from typing import Optional, Any
from agents import exceptions
from pipeline.config import Config
from database.util import DataElement
from database.mongodb_database import MongoDatabase
db = MongoDatabase
from pipeline.tools.tools import run_rag
from pipeline.utils.agent_logger import log_agent_run
from .model import analyzer
from .prompts import Analyzer_input

def extract_original_name(timestamped_name: str) -> str:
    """Extract original name from timestamped name."""
    # Remove timestamp prefix (format: YYYYMMDD-HH:MM:SS-)
    timestamp_pattern = r'^\d{8}-\d{2}:\d{2}:\d{2}-'
    original_name = re.sub(timestamp_pattern, '', timestamped_name)
    return original_name

async def analyse(
    name: str,
    motivation: str,
    program_file_path: str = Config.SOURCE_FILE,
    result_file_path: str = Config.RESULT_FILE,
    result_file_path_test: str = Config.RESULT_FILE_TEST,
    parent: int = None
) -> DataElement:
    """Analyze experiment results and generate comprehensive analysis."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract original name (remove timestamp)
    original_name = extract_original_name(name)
    
    # Get program content
    program = _read_program_file(program_file_path)
    
    # Get training and test result content
    train_result = _read_csv_file(result_file_path)
    test_result = _read_csv_file(result_file_path_test)
    
    # Create result dictionary
    result_dict = {
        'train': train_result,
        'test': test_result
    }
    
    # Use timestamped name for lookup, but replace with original name
    increment(result_dict, name, original_name, 'train')
    increment(result_dict, name, original_name, 'test')

    db = MongoDatabase()
    # Fetch reference elements from the database
    ref_elements = db.getcontextualnodes(parent) if parent else {"directparent": None, "strongestsiblings": [], "grandparent": None}
    result_content = f"""### Current Experiment Results: 
    **Training Progression**: {result_dict["train"]}
    **Evaluation Results**: {result_dict["test"]}
    """
    
    analysis = await run_analyzer(original_name, result_content, motivation, ref_elements)
    
    # Get paper content
    paper_query = analysis.experimental_results_analysis
    paper_result = run_rag(paper_query)
    if paper_result.get('success') and 'results' in paper_result:
        paper_content = paper_result['results']
    else:
        paper_content = paper_result.get('error', 'RAG query failed or no results')
    content_str = str(paper_content)
    analysis_result = (
        analysis.design_evaluation + 
        analysis.experimental_results_analysis +
        analysis.expectation_vs_reality_comparison +
        analysis.theoretical_explanation_with_evidence +
        analysis.synthesis_and_insights
    )
    
    result = DataElement(
        time=current_time,
        name=original_name,  # Use original name (without timestamp)
        result=result_dict,
        program=program,
        motivation=motivation,
        analysis=analysis_result,
        cognition=content_str,
        log=None,
        parent=parent
    )
    
    return result

def _read_program_file(file_path: str) -> str:
    """Read program file content with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: Program file '{file_path}' not found"
    except Exception as e:
        return f"Error reading program file: {str(e)}"

def _read_csv_file(file_path: str) -> str:
    """Read CSV file content with error handling."""
    try:
        result_lines = []
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                result_lines.append(','.join(row))
        return '\n'.join(result_lines)
    except FileNotFoundError:
        return f"Error: Result file '{file_path}' not found"
    except Exception as e:
        return f"Error reading result file: {str(e)}"

def increment(result: dict, timestamped_name: str, original_name: str, key: str) -> None:
    """Extract corresponding timestamped_name row from CSV, but replace with original_name."""
    csv_data = result[key]
    string_io = io.StringIO(csv_data)
    reader = csv.reader(string_io)
    
    try:
        header = next(reader)
        matched_row = None
        
        for row in reader:
            if row and row[0].strip() == timestamped_name:
                matched_row = row
                # Replace timestamped_name in first column with original_name
                if matched_row:
                    matched_row[0] = original_name
                break
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(header)
        if matched_row:
            writer.writerow(matched_row)
        
        result[key] = output.getvalue().strip()
    except StopIteration:
        pass

async def run_analyzer(
    name: str, 
    result_content: str, 
    motivation: str, 
    ref_elements: dict
) -> Optional[Any]:
    """Run analyzer with retry mechanism."""
    ref_context = _build_reference_context(ref_elements)
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        try:
            analyzer_result = await log_agent_run(
                "analyzer",
                analyzer,
                Analyzer_input(name, result_content, motivation, ref_context),
            )
            return analyzer_result.final_output
            
        except exceptions.MaxTurnsExceeded as e:
            print(f"Analyzer exceeded maximum turns, attempt {attempt + 1}")
        except Exception as e:
            print(f"Analyzer error on attempt {attempt + 1}: {e}")
    
    return None

def _build_reference_context(ref_elements: dict) -> str:
    """Build reference context string from reference elements."""
    ref_context = "# Reference Experiments\n"
    
    if ref_elements.get("direct_parent"):
        ref_context += "### Direct Parent\n"
        ref_context += _ref_elements_context(DataElement(**ref_elements["direct_parent"]))
        ref_context += "\n\n"
    if ref_elements.get("strongest_siblings"):
        ref_context += "### Strongest Siblings\n"
        for sibling in ref_elements["strongest_siblings"]:
            ref_context += _ref_elements_context(DataElement(**sibling))
        ref_context += "\n\n"
    if ref_elements.get("grandparent"):
        ref_context += "### Grandparent\n"
        ref_context += _ref_elements_context(DataElement(**ref_elements["grandparent"]))
        ref_context += "\n\n"
    
    return ref_context

def _ref_elements_context(ref_element: DataElement) -> str:
    """Generate context string for a reference element."""
    return f"""### Reference Experiment {ref_element.name}
#### Experiment Motivation
{ref_element.motivation}
#### Experiment Result
**Training Progression**: {ref_element.result["train"]}
**Evaluation Results**: {ref_element.result["test"]}
"""

def save(name: str) -> None:
    """Save source file content to code pool (pipeline/pool) with given name."""
    with open(Config.SOURCE_FILE, "r", encoding='utf-8') as f:
        content = f.read()
    with open(f"{Config.CODE_POOL}/{name}.py", "w", encoding='utf-8') as f:
        f.write(content)
