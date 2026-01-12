# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import torch


def set_basic_config(level):
    """
    This function sets the global logging format and level. It will be called when import verl
    """
    import sys
    log_dir = os.environ.get("VERL_LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Get rank for unique log file
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    except:
        rank = 0

    log_file = os.path.join(log_dir, f"verl_log_rank{rank}.txt")

    # Configure root logger with both file and stream handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(name)s:%(message)s")

    # File handler - always write to file for debugging
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Stream handler - also output to console
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    print(f"[VERL LOG] Logging configured: file={log_file}, level={level}", flush=True)


def log_to_file(string):
    print(string)
    if os.path.isdir("logs"):
        with open(f"logs/log_{torch.distributed.get_rank()}", "a+") as f:
            f.write(string + "\n")
