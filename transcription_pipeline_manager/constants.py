# REST interface constants
DEFAULT_REST_HOST = "127.0.0.1"
DEFAULT_REST_PORT = 8080
NGROK_DOMAIN = "https://completely-quiet-bobcat.ngrok-free.app"

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
POD_URL_TEMPLATE = "https://%s-8080.proxy.runpod.net"

# State constants for the manager main loop
STATE_STARTING_CYCLE = "STARTING_CYCLE"
STATE_ATTEMPTING_POD_START = "ATTEMPTING_POD_START"
STATE_WAITING_FOR_IDLE = "WAITING_FOR_IDLE"
STATE_ATTEMPTING_PIPELINE_RUN = "ATTEMPTING_PIPELINE_RUN"
STATE_UPDATING_COUNTS = "UPDATING_COUNTS"
STATE_WAITING_AFTER_FAILURE = "WAITING_AFTER_FAILURE"
VALID_STATES = [
    STATE_STARTING_CYCLE,
    STATE_ATTEMPTING_POD_START,
    STATE_WAITING_FOR_IDLE,
    STATE_ATTEMPTING_PIPELINE_RUN,
    STATE_UPDATING_COUNTS,
    STATE_WAITING_AFTER_FAILURE,
]
