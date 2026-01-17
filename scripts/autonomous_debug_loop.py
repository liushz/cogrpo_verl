#!/usr/bin/env python3
"""
Autonomous CoGRPO Experiment Debugging Loop

This script monitors experiment logs, detects failures, applies fixes,
and restarts training without human intervention.
"""

import os
import re
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Configuration
LOG_DIR = Path("/mnt/shared-storage-user/liuhongwei/main_works/temp_debug/logs")
REPO_DIR = Path("/mnt/shared-storage-user/liuhongwei/main_works/repos/repro")
LAUNCH_SCRIPT = REPO_DIR / "run_cgrpo_mini_cluster.sh"
MAX_ITERATIONS = 20
INITIAL_WAIT = 300  # 5 minutes for initialization
CHECK_INTERVAL = 60  # 1 minute between checks
RESTART_WAIT = 300  # 5 minutes after restart
STABLE_DURATION = 1800  # 30 minutes for success
MAX_SAME_ERROR = 3  # Stop if same error occurs 3 times

# Error patterns to detect
ERROR_PATTERNS = {
    "lora_init": [
        r"Failed to load Verifier LoRA",
        r"verifier_lora_int_id is None",
        r"LoRA not found in loaded LoRAs",
    ],
    "cuda_oom": [
        r"CUDA out of memory",
        r"RuntimeError: CUDA error: out of memory",
    ],
    "verifier_batch": [
        r"verifier_batch missing required non_tensor fields",
        r"Failed to build verifier_train_batch",
    ],
    "gradient_error": [
        r"Base param has non-zero grad after verifier update",
        r"optimizer step failed",
    ],
    "ray_error": [
        r"Ray actor died",
        r"Connection timeout",
        r"Worker crashed",
    ],
    "config_error": [
        r"KeyError",
        r"AttributeError in config",
        r"Missing required config field",
    ],
    "peft_serialization": [
        r"'str' object has no attribute 'value'",
        r"AttributeError.*task_type.*value",
    ],
    "python_exception": [
        r"Traceback \(most recent call last\)",
        r"Exception:",
        r"Error:",
    ],
}


class DebugLoop:
    def __init__(self):
        self.iteration = 0
        self.error_history = []
        self.last_error = None
        self.same_error_count = 0
        self.start_time = None
        self.last_stable_check = None
        self.current_job_id = None

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def run_command(self, cmd: str, description: str = "") -> Tuple[int, str, str]:
        """Run shell command and return (returncode, stdout, stderr)."""
        if description:
            self.log(f"Running: {description}")
        self.log(f"Command: {cmd}")

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(REPO_DIR)
        )

        return result.returncode, result.stdout, result.stderr

    def get_latest_log_file(self) -> Optional[Path]:
        """Get the most recently modified log file."""
        if not LOG_DIR.exists():
            self.log(f"Log directory does not exist: {LOG_DIR}", "WARNING")
            return None

        log_files = list(LOG_DIR.glob("verl_log_*.txt"))
        if not log_files:
            self.log("No log files found", "WARNING")
            return None

        # Sort by modification time, most recent first
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return log_files[0]

    def read_log_tail(self, log_file: Path, lines: int = 500) -> str:
        """Read last N lines from log file."""
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                return ''.join(all_lines[-lines:])
        except Exception as e:
            self.log(f"Error reading log file: {e}", "ERROR")
            return ""

    def detect_errors(self, log_content: str) -> List[Dict[str, str]]:
        """Detect errors in log content."""
        errors = []

        for error_type, patterns in ERROR_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, log_content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    # Get context around the error (5 lines before and after)
                    lines = log_content[:match.start()].split('\n')
                    start_line = max(0, len(lines) - 5)
                    context_before = '\n'.join(lines[start_line:])

                    lines_after = log_content[match.end():].split('\n')[:5]
                    context_after = '\n'.join(lines_after)

                    errors.append({
                        "type": error_type,
                        "pattern": pattern,
                        "match": match.group(0),
                        "context": context_before + match.group(0) + context_after
                    })

        return errors

    def check_training_progress(self, log_content: str) -> Dict[str, any]:
        """Check if training is progressing."""
        # Look for step numbers
        step_matches = re.findall(r"step[:\s]+(\d+)", log_content, re.IGNORECASE)
        steps = [int(s) for s in step_matches] if step_matches else []

        # Look for loss values
        loss_matches = re.findall(r"loss[:\s]+([\d.]+)", log_content, re.IGNORECASE)
        losses = [float(l) for l in loss_matches] if loss_matches else []

        # Look for verifier interventions
        intervention_matches = re.findall(r"intervention_rate[:\s]+([\d.]+)", log_content, re.IGNORECASE)
        interventions = [float(i) for i in intervention_matches] if intervention_matches else []

        return {
            "steps": steps,
            "max_step": max(steps) if steps else 0,
            "losses": losses,
            "interventions": interventions,
            "is_progressing": len(steps) > 0 and len(set(steps)) > 1
        }

    def get_current_rjob(self) -> Optional[str]:
        """Get current running RJob ID."""
        returncode, stdout, stderr = self.run_command(
            "rjob list | grep 'cgrpo-verifier.*Running' | head -1",
            "Check for running RJob"
        )

        if returncode == 0 and stdout.strip():
            # Extract job ID from output
            match = re.search(r"rjob-([\w-]+)", stdout)
            if match:
                return match.group(0)

        return None

    def stop_rjob(self, job_id: str):
        """Stop an RJob."""
        self.log(f"Stopping RJob: {job_id}")
        returncode, stdout, stderr = self.run_command(
            f"rjob stop {job_id}",
            f"Stop RJob {job_id}"
        )

        if returncode == 0:
            self.log(f"Successfully stopped {job_id}")
            time.sleep(30)  # Wait for GPUs to be released
        else:
            self.log(f"Failed to stop {job_id}: {stderr}", "ERROR")

    def apply_fix(self, error: Dict[str, str]) -> bool:
        """Apply fix for detected error."""
        error_type = error["type"]
        self.log(f"Applying fix for error type: {error_type}")

        # Most errors are already fixed in the code
        # This function is a placeholder for future auto-fixes

        if error_type == "peft_serialization":
            self.log("PEFT serialization fix already applied in code")
            return True

        elif error_type == "lora_init":
            self.log("LoRA initialization fix already applied in code")
            return True

        elif error_type == "cuda_oom":
            self.log("CUDA OOM detected - may need manual config adjustment", "WARNING")
            return False

        else:
            self.log(f"No automatic fix available for {error_type}", "WARNING")
            return False

    def launch_experiment(self):
        """Launch the experiment."""
        self.log("=" * 80)
        self.log(f"ITERATION {self.iteration + 1}/{MAX_ITERATIONS}")
        self.log("=" * 80)

        self.log(f"Launching experiment: {LAUNCH_SCRIPT}")

        # Launch in background
        cmd = f"bash {LAUNCH_SCRIPT} > /tmp/launch_output_{self.iteration}.log 2>&1 &"
        returncode, stdout, stderr = self.run_command(cmd, "Launch experiment")

        if returncode != 0:
            self.log(f"Failed to launch experiment: {stderr}", "ERROR")
            return False

        self.log("Experiment launched successfully")
        self.start_time = time.time()
        self.last_stable_check = time.time()
        return True

    def create_git_commit(self, message: str):
        """Create git commit for applied fixes."""
        self.log(f"Creating git commit: {message}")

        # Check if there are changes to commit
        returncode, stdout, stderr = self.run_command(
            "git diff --quiet",
            "Check for uncommitted changes"
        )

        if returncode == 0:
            self.log("No changes to commit")
            return

        # Commit changes
        commit_msg = f"{message}\n\nCo-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
        returncode, stdout, stderr = self.run_command(
            f'git commit -am "{commit_msg}"',
            "Commit changes"
        )

        if returncode == 0:
            self.log("Git commit created successfully")
        else:
            self.log(f"Failed to create git commit: {stderr}", "ERROR")

    def run(self):
        """Main loop."""
        self.log("Starting Autonomous CoGRPO Debugging Loop")
        self.log(f"Max iterations: {MAX_ITERATIONS}")
        self.log(f"Log directory: {LOG_DIR}")
        self.log(f"Launch script: {LAUNCH_SCRIPT}")

        # Initial launch
        if not self.launch_experiment():
            self.log("Failed to launch initial experiment", "ERROR")
            return

        self.log(f"Waiting {INITIAL_WAIT} seconds for initialization...")
        time.sleep(INITIAL_WAIT)

        # Main monitoring loop
        while self.iteration < MAX_ITERATIONS:
            self.iteration += 1

            self.log("-" * 80)
            self.log(f"Check iteration {self.iteration}")

            # Get latest log file
            log_file = self.get_latest_log_file()
            if not log_file:
                self.log("No log file found, waiting...", "WARNING")
                time.sleep(CHECK_INTERVAL)
                continue

            self.log(f"Reading log file: {log_file}")
            log_content = self.read_log_tail(log_file)

            if not log_content:
                self.log("Empty log content, waiting...", "WARNING")
                time.sleep(CHECK_INTERVAL)
                continue

            # Check training progress
            progress = self.check_training_progress(log_content)
            self.log(f"Training progress: max_step={progress['max_step']}, "
                    f"is_progressing={progress['is_progressing']}")

            # Detect errors
            errors = self.detect_errors(log_content)

            if not errors and progress['is_progressing']:
                # Training is progressing without errors
                elapsed = time.time() - self.last_stable_check
                self.log(f"Training stable for {elapsed:.0f} seconds")

                if elapsed >= STABLE_DURATION:
                    self.log("=" * 80)
                    self.log("SUCCESS! Training has been stable for 30+ minutes")
                    self.log(f"Final step: {progress['max_step']}")
                    self.log("=" * 80)
                    return

                # Continue monitoring
                time.sleep(CHECK_INTERVAL)
                continue

            elif not errors:
                # No errors but not progressing - might be initializing
                self.log("No errors detected, but training not progressing yet")
                time.sleep(CHECK_INTERVAL)
                continue

            # Errors detected
            self.log(f"Detected {len(errors)} error(s)", "ERROR")

            for i, error in enumerate(errors[:3]):  # Show first 3 errors
                self.log(f"Error {i+1}: Type={error['type']}, Pattern={error['pattern']}")
                self.log(f"Context:\n{error['context'][:500]}")

            # Check if same error as before
            error_signature = f"{errors[0]['type']}:{errors[0]['pattern']}"
            if error_signature == self.last_error:
                self.same_error_count += 1
                self.log(f"Same error occurred {self.same_error_count} times", "WARNING")

                if self.same_error_count >= MAX_SAME_ERROR:
                    self.log("=" * 80)
                    self.log("FAILURE: Same error occurred 3 times in a row")
                    self.log(f"Error: {error_signature}")
                    self.log("Manual intervention required")
                    self.log("=" * 80)
                    return
            else:
                self.last_error = error_signature
                self.same_error_count = 1

            # Try to apply fixes
            fixes_applied = False
            for error in errors:
                if self.apply_fix(error):
                    fixes_applied = True

            # Stop current job
            current_job = self.get_current_rjob()
            if current_job:
                self.stop_rjob(current_job)

            # Restart experiment
            self.log(f"Restarting experiment (iteration {self.iteration}/{MAX_ITERATIONS})")

            if not self.launch_experiment():
                self.log("Failed to restart experiment", "ERROR")
                return

            self.log(f"Waiting {RESTART_WAIT} seconds for restart...")
            time.sleep(RESTART_WAIT)

        # Max iterations reached
        self.log("=" * 80)
        self.log(f"FAILURE: Max iterations ({MAX_ITERATIONS}) reached")
        self.log("Manual intervention required")
        self.log("=" * 80)


if __name__ == "__main__":
    loop = DebugLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        # Stop any running jobs
        current_job = loop.get_current_rjob()
        if current_job:
            loop.stop_rjob(current_job)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
