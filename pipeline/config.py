import os
from pathlib import Path

class Config:
    """Configuration settings for the experiment."""
    _instance = None

    # Target file
    SOURCE_FILE: str = "pipeline/base_program.py"

    # Training script
    BASH_SCRIPT: str = "pipeline/train.sh"

    # Experiment results
    RESULT_FILE: str = "pipeline/files/analysis/loss.csv"
    RESULT_FILE_TEST: str = "pipeline/files/analysis/benchmark.csv"

    # Debug file
    DEBUG_FILE: str = "pipeline/files/debug/training_error.txt"

    # Code pool directory where agent-generated experiments live
    CODE_POOL: str = "pipeline/pool"

    # Maximum number of debug attempts
    MAX_DEBUG_ATTEMPT: int = 5

    # Maximum number of retry attempts
    MAX_RETRY_ATTEMPTS: int = 10

    # RAG service URL
    RAG: str = "http://localhost:13142"

    # Database URL
    DATABASE: str = "http://localhost:8001"

    def __new__(cls):
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def find_experiment_script(self, experiment_name: str) -> str:
        """
        Find the agent-generated experiment script in CODE_POOL.
        
        Args:
            experiment_name: Name of the experiment (e.g., 'delta_net_recurrent_ema')
            
        Returns:
            Full path to the experiment script
            
        Raises:
            FileNotFoundError: If the experiment script doesn't exist
        """
        # Search for exact match first
        exact_path = os.path.join(self.CODE_POOL, f"{experiment_name}.py")
        if os.path.exists(exact_path):
            return exact_path
        
        # Search for partial matches (handles naming variations)
        code_pool_path = Path(self.CODE_POOL)
        if code_pool_path.exists():
            for script_file in code_pool_path.glob("*.py"):
                if experiment_name in script_file.stem:
                    return str(script_file)
        
        # List available experiments for helpful error message
        available_experiments = []
        if code_pool_path.exists():
            available_experiments = [f.stem for f in code_pool_path.glob("*.py")]
        
        raise FileNotFoundError(
            f"Experiment script '{experiment_name}.py' not found in {self.CODE_POOL}. "
            f"Available experiments: {available_experiments}"
        )

    def get_experiment_module_path(self, experiment_name: str) -> str:
        """Get the Python module path for importing the experiment."""
        script_path = self.find_experiment_script(experiment_name)
        return script_path
        
    def list_available_experiments(self) -> list:
        """List all available experiment scripts in the code pool."""
        code_pool_path = Path(self.CODE_POOL)
        if not code_pool_path.exists():
            return []
        
        return [f.stem for f in code_pool_path.glob("*.py")]
