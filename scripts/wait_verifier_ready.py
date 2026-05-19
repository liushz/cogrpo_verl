#!/usr/bin/env python3
import json
import os
import re
import time
import urllib.error
import urllib.request


def as_bool(raw, default=False):
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def parse_urls():
    raw_dict = (os.environ.get("COGRPO_VERIFIER_SERVER_URL_DICT", "") or "").strip()
    raw_urls = (os.environ.get("COGRPO_VERIFIER_SERVER_URLS", "") or "").strip()
    raw_dict_file = (
        os.environ.get("COGRPO_VERIFIER_URL_DICT_FILE", os.environ.get("VERIFIER_SERVER_URL_DICT_FILE", "")) or ""
    ).strip()
    if (not raw_dict) and raw_dict_file:
        try:
            with open(raw_dict_file, "r", encoding="utf-8") as f:
                raw_dict = (f.read() or "").strip()
        except Exception:
            raw_dict = ""
    urls = []
    if raw_dict:
        try:
            obj = json.loads(raw_dict)
            if isinstance(obj, dict):
                for _, value in obj.items():
                    if isinstance(value, str) and value.strip():
                        urls.append(value.strip().rstrip("/"))
        except Exception:
            pass
    if (not urls) and raw_urls:
        for item in raw_urls.split(","):
            url = item.strip().rstrip("/")
            if url:
                urls.append(url)
    dedup = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            dedup.append(url)
    return dedup


def has_verifier_config():
    raw_dict = (os.environ.get("COGRPO_VERIFIER_SERVER_URL_DICT", "") or "").strip()
    raw_urls = (os.environ.get("COGRPO_VERIFIER_SERVER_URLS", "") or "").strip()
    raw_dict_file = (
        os.environ.get("COGRPO_VERIFIER_URL_DICT_FILE", os.environ.get("VERIFIER_SERVER_URL_DICT_FILE", "")) or ""
    ).strip()
    return bool(raw_dict or raw_urls or raw_dict_file)


def check_models(base_url, expected_model):
    last_msg = "models_http=404"
    body = None
    for suffix in ("/v1/models", "/models"):
        req = urllib.request.Request(f"{base_url}{suffix}", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                code = int(getattr(resp, "status", resp.getcode()))
                body = resp.read()
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            body = None
        except Exception as exc:
            last_msg = f"models_err={type(exc).__name__}:{exc}"
            continue
        if code == 200:
            last_msg = "ok"
            break
        last_msg = f"models_http={code}"
        body = None
    if body is None:
        return False, last_msg
    if not expected_model:
        return True, "ok"
    try:
        obj = json.loads(body.decode("utf-8", errors="ignore"))
        model_ids = []
        data = obj.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    model_id = str(item.get("id") or "").strip()
                    if model_id:
                        model_ids.append(model_id)
        if expected_model in model_ids:
            return True, "ok"
        return False, f"model_missing={expected_model}"
    except Exception as exc:
        return False, f"models_parse_err={type(exc).__name__}:{exc}"


def check_update_endpoint(base_url):
    base_candidates = [base_url]
    if base_url.endswith("/v1"):
        base_candidates.append(base_url[:-3].rstrip("/"))
    for base in base_candidates:
        for suffix in ("/update_weights", "/update_weights_from_tensor"):
            # Probe endpoint existence without sending an invalid empty JSON body,
            # which otherwise creates misleading 422 noise in verifier logs.
            req = urllib.request.Request(f"{base}{suffix}", method="OPTIONS")
            try:
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    code = int(getattr(resp, "status", resp.getcode()))
            except urllib.error.HTTPError as exc:
                code = int(exc.code)
            except Exception as exc:
                return False, f"update_err={type(exc).__name__}:{exc}"
            if code != 404:
                return True, f"update_http={code}"
    return False, "update_http=404"


def suspicious(text):
    text = str(text or "").strip()
    if len(text) < 4:
        return True
    sample = text[:128]
    if re.fullmatch(r"[!！]+", text):
        return True
    if set(sample) <= set("!！<>/thiknTHIKN \n\r\t"):
        return True
    punct_chars = set("!！?？.,，。;；:-_<>/")
    punct = sum(ch in punct_chars for ch in sample)
    if punct / max(1, len(sample)) >= 0.6:
        return True
    if sample.lower().startswith("<th") and "!" in sample:
        return True
    return False


def check_chat(base_url, expected_model, request_logprobs):
    if not expected_model:
        return True, "chat_skip_no_model"

    payload = {
        "model": expected_model,
        "messages": [
            {"role": "system", "content": "You are a careful math assistant."},
            {"role": "user", "content": "What is 2+2? Answer briefly."},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 64,
        "stream": False,
    }
    if request_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            code = int(getattr(resp, "status", resp.getcode()))
            body = resp.read()
    except urllib.error.HTTPError as exc:
        code = int(exc.code)
        try:
            body = exc.read()
        except Exception:
            body = b""
        return False, f"chat_http={code}:{body.decode('utf-8', errors='ignore')[:160]}"
    except Exception as exc:
        return False, f"chat_err={type(exc).__name__}:{exc}"

    if code != 200:
        return False, f"chat_http={code}"
    try:
        obj = json.loads(body.decode("utf-8", errors="ignore"))
        text = str((((obj.get("choices") or [{}])[0].get("message") or {}).get("content")) or "")
    except Exception as exc:
        return False, f"chat_parse_err={type(exc).__name__}:{exc}"
    if suspicious(text):
        preview = text[:120].replace("\n", "\\n")
        return False, f"chat_suspicious={preview}"
    return True, "ok"


def main():
    expected_model = (os.environ.get("COGRPO_VERIFIER_MODEL", "") or "").strip()
    service_adapter_name = (os.environ.get("COGRPO_VERIFIER_SERVICE_ADAPTER_NAME", "verifier_lora") or "verifier_lora").strip()
    timeout_s = int((os.environ.get("COGRPO_VERIFIER_WAIT_TIMEOUT_S", "43200") or "43200").strip())
    if timeout_s < 0:
        timeout_s = 0
    interval_s = max(1, int((os.environ.get("COGRPO_VERIFIER_WAIT_INTERVAL_S", "10") or "10").strip()))
    check_update_mode = (
        (os.environ.get("COGRPO_VERIFIER_WAIT_CHECK_UPDATE_ENDPOINT", "auto") or "auto").strip().lower()
    )
    lora_enabled = as_bool(os.environ.get("COGRPO_VERIFIER_LORA_ENABLE", "0"))
    update_base = as_bool(os.environ.get("COGRPO_VERIFIER_UPDATE_BASE", "0"))
    strict_sync = as_bool(os.environ.get("COGRPO_STRICT_VERIFIER_SYNC", "1"))
    sync_freq = int((os.environ.get("COGRPO_VERIFIER_LORA_SYNC_FREQ", "1") or "1").strip())
    request_logprobs = as_bool(os.environ.get("COGRPO_VERIFIER_REQUEST_LOGPROBS", "0"))
    if not expected_model and lora_enabled and (not update_base) and sync_freq > 0:
        expected_model = service_adapter_name
    need_update = check_update_mode in ("1", "true", "yes", "on") or (
        check_update_mode == "auto" and (lora_enabled or update_base) and strict_sync and sync_freq > 0
    )
    need_chat = True

    deadline = None if timeout_s == 0 else (time.time() + timeout_s)
    round_idx = 0
    while True:
        round_idx += 1
        urls = parse_urls()
        if not urls:
            if has_verifier_config():
                print(
                    "[xtuner-cf8-bsz8][wait] ERROR: verifier config is set but no valid urls could be parsed.",
                    flush=True,
                )
                return 4
            print("[xtuner-cf8-bsz8][wait] verifier urls empty; skip wait gate.", flush=True)
            return 0
        failures = []
        for url in urls:
            ok_models, msg_models = check_models(url, expected_model)
            if not ok_models:
                failures.append(f"{url}:{msg_models}")
                continue
            if need_chat:
                ok_chat, msg_chat = check_chat(url, expected_model, request_logprobs)
                if not ok_chat:
                    failures.append(f"{url}:{msg_chat}")
                    continue
            if need_update:
                ok_update, msg_update = check_update_endpoint(url)
                if not ok_update:
                    failures.append(f"{url}:{msg_update}")

        if not failures:
            print(
                f"[xtuner-cf8-bsz8][wait] verifier ready for {len(urls)} urls; "
                f"model={expected_model or '<skip>'} need_update={need_update}",
                flush=True,
            )
            return 0

        if deadline is not None and time.time() >= deadline:
            print(
                "[xtuner-cf8-bsz8][wait] timeout waiting verifier urls; "
                f"timeout_s={timeout_s} failures={failures[:6]}",
                flush=True,
            )
            return 3

        print(
            f"[xtuner-cf8-bsz8][wait] not ready round={round_idx} "
            f"retry_in={interval_s}s failures={failures[:4]}",
            flush=True,
        )
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
