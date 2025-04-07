# REST interface constants
DEFAULT_REST_HOST = "127.0.0.1"
DEFAULT_REST_PORT = 8080

# Exit Codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# Transcription processing
DEFAULT_TRANSCRIPTION_LIMIT = 1000
DEFAULT_TRANSCRIPTION_PROCESSING_LIMIT = 2

# Manager specific constants
CYCLE_DURATION = 3600  # seconds (1 hour)
STATUS_CHECK_TIMEOUT = 300  # seconds (5 minutes) for idle check
COUNT_UPDATE_INTERVAL = 60  # seconds (1 minute)
MAIN_LOOP_SLEEP = 1  # second
POD_STATUS_CHECK_INTERVAL = 5 # seconds (how often to check /status when waiting for idle)
POD_REQUEST_TIMEOUT = 15 # seconds (default timeout for HTTP requests to the pod)
RUNPOD_CONFIG_FILENAME = "transcription-pipeline.config.yaml"
RUNPOD_CONFIG_DIR = "runpod-singleton"
