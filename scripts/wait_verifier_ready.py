#!/usr/bin/env python3
import json
import os
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
    payload = b"{}"
    headers = {"Content-Type": "application/json"}
    base_candidates = [base_url]
    if base_url.endswith("/v1"):
        base_candidates.append(base_url[:-3].rstrip("/"))
    for base in base_candidates:
        for suffix in ("/update_weights", "/update_weights_from_tensor"):
            req = urllib.request.Request(f"{base}{suffix}", data=payload, headers=headers, method="POST")
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


def main():
    urls = parse_urls()
    if not urls:
        print("[xtuner-cf8-bsz8][wait] verifier urls empty; skip wait gate.", flush=True)
        return 0

    expected_model = (os.environ.get("COGRPO_VERIFIER_MODEL", "") or "").strip()
    timeout_s = int((os.environ.get("COGRPO_VERIFIER_WAIT_TIMEOUT_S", "7200") or "7200").strip())
    if timeout_s < 0:
        timeout_s = 0
    interval_s = max(1, int((os.environ.get("COGRPO_VERIFIER_WAIT_INTERVAL_S", "10") or "10").strip()))
    check_update_mode = (
        (os.environ.get("COGRPO_VERIFIER_WAIT_CHECK_UPDATE_ENDPOINT", "auto") or "auto").strip().lower()
    )
    lora_enabled = as_bool(os.environ.get("COGRPO_VERIFIER_LORA_ENABLE", "0"))
    strict_sync = as_bool(os.environ.get("COGRPO_STRICT_VERIFIER_SYNC", "1"))
    sync_freq = int((os.environ.get("COGRPO_VERIFIER_LORA_SYNC_FREQ", "1") or "1").strip())
    need_update = check_update_mode in ("1", "true", "yes", "on") or (
        check_update_mode == "auto" and lora_enabled and strict_sync and sync_freq > 0
    )

    deadline = None if timeout_s == 0 else (time.time() + timeout_s)
    round_idx = 0
    while True:
        round_idx += 1
        failures = []
        for url in urls:
            ok_models, msg_models = check_models(url, expected_model)
            if not ok_models:
                failures.append(f"{url}:{msg_models}")
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
