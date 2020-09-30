DATABASE = {
    "name": "faceshifter",
    "ip": "localhost",
    "port": 27017,
}

GENERATOR_MODEL_PATH = "faceshifter/model_weights/G_latest.pth"
ARCFACE_MODEL_PATH = "./faceshifter/model_weights/model_ir_se50.pth"

UPLOAD_PATH = "web/uploads/"
MEDIA_RESULTS_PATH = "web/media_results/"
MEDIA_RESULTS_URL = '/media_result/'
SERVER_PORT = 8888

MAX_QUEUE_SIZE = 10000
INFERENCE_WORKERS_COUNT = 3
