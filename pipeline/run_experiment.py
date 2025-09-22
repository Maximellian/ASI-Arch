# pipeline/run_experiment.py
import sys, asyncio, traceback

from pipeline.evolve.interface import evolve  # or the correct function

if __name__ == "__main__":
    try:
        context = sys.argv[1] if len(sys.argv) > 1 else "default context"
        result = asyncio.run(evolve(context))
        print("Evolve result:", result)
    except Exception as e:
        print("Exception in run_experiment.py:", e)
        traceback.print_exc()
        sys.exit(1)

