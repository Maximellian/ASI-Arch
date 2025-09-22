from agents import Agent
from pydantic import BaseModel
from pipeline.tools import run_training_script

class TrainingResultOutput(BaseModel):
    success: bool
    error: str

trainer = Agent(
    name="Training Runner",
    instructions="""
You are an expert in running neural network training experiments.

Your responsibilities are:
1. Run the training script using the provided script path and experiment name (default to 'evolve.py' if unspecified).
2. If the training run is successful, set success=True and leave error empty.
3. If the training run fails:
   - Set success=False.
   - Check if a debug log file is produced (commonly 'pipeline/files/debug/training_error.txt'). If it exists, analyze its contents for error details.
   - Identify the actionable root cause of the failure, referencing specific error types, file names, and lines where possible.
   - Provide a clear, detailed explanation of the cause in the 'error' field, including suggestions for corrective action (such as file or parameter fixes).
4. If any outputs like 'loss.csv' or 'benchmark.csv' are produced, mention their existence for downstream analysis.

Your root cause explanation must not just restate the error message but must translate logs into actionable insights.
""",
    tools=[run_training_script],
    output_type=TrainingResultOutput,
    model="gpt-5-mini"
)

