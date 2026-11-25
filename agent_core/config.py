"""Settings for the data-science agent."""

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Seconds to reserve before the harness deadline.
SAFETY_MARGIN_S = 3

# Maximum number of tokens generated per model reply.
MAX_NEW_TOKENS = 512

# Maximum number of conversations per model batch.
LLM_BATCH_SIZE = 15

# Maximum number of generated programs running at the same time.
MAX_WORKERS = 3

EXECUTION_RESERVE_S = 12.0

# Maximum time in seconds for one generated program.
PROGRAM_TIMEOUT_S = 8.0
