import logging
import os
import sys

from pythonjsonlogger import jsonlogger

def setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)

    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
            "message": "message",
        },
        datefmt="%Y-%m-%dT%H:%M:%S",
        json_ensure_ascii=False,
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("logs/churn_service.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)

def get_logger(name: str = "churn_service") -> logging.Logger:
    return logging.getLogger(name)
