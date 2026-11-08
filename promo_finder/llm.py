"""Model-agnostic LLM client. Swap models with env vars only.

  LLM_PROVIDER = gemini | openai | baseline
  LLM_MODEL    = e.g. gemini-2.5-flash, llama-3.3-70b-versatile, gpt-4o-mini
  LLM_BASE_URL = for openai-compatible APIs (Groq, OpenRouter, a Kuku gateway)
  LLM_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY / GROQ_API_KEY
  LLM_FALLBACK_MODELS = comma list tried in order when a model is out of quota/overloaded
  LLM_THINKING = low (default, fast) | minimal | high | default
  COLD_VIEWER_MODEL = optional different model for the cold-viewer test

Every call is cached on disk (same prompt -> no new API call) and logged
with tokens + latency so the run report can show real cost.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    raw = p.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")          # PowerShell 5 `echo > .env`
    else:
        text = raw.decode("utf-8-sig", "replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class LLMError(RuntimeError):
    pass


MAX_SLEEP = 60          # never wait longer than this between retries
REQUEST_TIMEOUT = 300   # seconds to wait for one answer


def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(text[s: e + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Model did not return valid JSON: {text[:300]}")


# Fallback only: at runtime we list the models the key can use and pick the newest stable Flash.
GEMINI_PREFERENCE = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash"]


def _version_key(name: str):
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    major, minor = (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)
    preview = any(x in name for x in ("preview", "exp"))
    return (not preview, major, minor, -len(name))


class ModelUnavailable(LLMError):
    pass


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 cache_dir: str | Path = ".cache", min_interval: float | None = None,
                 temperature: float = 0.4, log=None):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or self._guess_provider()).lower()
        self.api_key = api_key or os.getenv("LLM_API_KEY") or self._key_for(self.provider)
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or self._default_base()).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or ""
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # free tiers are rate limited; space out calls (seconds)
        self.min_interval = float(min_interval if min_interval is not None
                                  else os.getenv("LLM_MIN_INTERVAL", "4" if self.provider == "gemini" else "1"))
        self._last_call = 0.0
        self.calls: list[dict] = []
        self.deadline: float | None = None          # epoch seconds; set per run by the pipeline
        self.thinking = os.getenv("LLM_THINKING", "low").lower()   # low | minimal | high | default
        self._no_thinking_cfg: dict[str, bool] = {}
        self._dead_calls = 0          # circuit breaker: consecutive calls that failed completely
        self.log = log or (lambda msg: print(msg, flush=True))
        if self.provider != "baseline" and not self.api_key:
            raise LLMError(f"No API key found for provider '{self.provider}'. Put it in .env")
        self._fallback_models = [m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
        if self.provider == "gemini" and not self.model:
            self.model = self._pick_gemini_model()
        if not self.model and self.provider == "openai":
            self.model = "gpt-4o-mini"
        if self.provider == "baseline":
            self.model = "rules-only"

    # -- setup helpers ---------------------------------------------------
    @staticmethod
    def _guess_provider() -> str:
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "baseline"

    @staticmethod
    def _key_for(provider: str) -> str | None:
        if provider == "gemini":
            return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")

    def _default_base(self) -> str:
        if self.provider == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta"
        if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            return "https://api.groq.com/openai/v1"
        return "https://api.openai.com/v1"

    def _pick_gemini_model(self) -> str:
        try:
            r = requests.get(f"{self.base_url}/models", params={"key": self.api_key, "pageSize": 200}, timeout=30)
            r.raise_for_status()
            names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
        except Exception as e:  # noqa: BLE001
            self.log(f"[llm] could not list models ({e}); defaulting to {GEMINI_PREFERENCE[0]}")
            return GEMINI_PREFERENCE[0]
        flash = [n for n in names if "flash" in n and not any(
            x in n for x in ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "8b"))]
        flash.sort(key=_version_key, reverse=True)
        self._fallback_models = self._fallback_models or flash[1:4]
        return flash[0] if flash else (names[0] if names else GEMINI_PREFERENCE[0])

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"

    # -- calling ---------------------------------------------------------
    def json(self, system: str, prompt: str, tag: str = "call", model: str | None = None,
             max_attempts: int = 6) -> dict:
        try:
            out = self._json(system, prompt, tag, model, max_attempts)
        except LLMError:
            self._dead_calls += 1
            raise
        self._dead_calls = 0
        return out

    def _json(self, system: str, prompt: str, tag: str, model: str | None, max_attempts: int) -> dict:
        """One JSON answer. Never sleeps > MAX_SLEEP, never runs past self.deadline,
        switches to a fallback model when a model is out of quota / overloaded / retired."""
        if self.provider == "baseline":
            raise LLMError("baseline provider has no model; pipeline uses rule fallbacks")
        follow_default = model is None
        model = model or self.model
        hit = self._cache_index().get(self._content_key(system, prompt))
        if hit and Path(hit).exists():
            data = json.loads(Path(hit).read_text(encoding="utf-8"))
            self.log(f"[llm] {tag}: reused saved answer ({data['meta'].get('model')})")
            self.calls.append({**data["meta"], "tag": tag, "cached": True, "cache_file": hit})
            return data["parsed"]

        if self._dead_calls >= 2:
            raise LLMError(f"{tag}: AI provider unavailable for this run (2 calls in a row failed); using rules")

        def switch(reason: str) -> bool:
            nonlocal model
            if not self._fallback_models:
                return False
            new = self._fallback_models.pop(0)
            self.log(f"[llm] {model} {reason}; switching to {new}")
            if follow_default or model == self.model:
                self.model = new
            model = new
            return True

        last_err: Exception | None = None
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            self._check_deadline(tag, 0)
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            t0 = self._last_call = time.time()
            try:
                text, usage = (self._gemini if self.provider == "gemini" else self._openai)(system, prompt, model)
                parsed = extract_json(text)
            except _RateLimited as e:
                last_err = e
                if e.daily_quota:
                    if switch("is out of free daily quota"):
                        continue
                    raise LLMError(f"{tag}: {model} is out of free daily quota and no fallback model is left")
                if e.overloaded and attempt >= 2 and switch("is overloaded"):
                    continue
                sleep_for = min(max(e.retry_after or 0, 5.0 * attempt), MAX_SLEEP)
                self._check_deadline(tag, sleep_for)
                self.log(f"[llm] {tag}: {'overloaded' if e.overloaded else 'rate limited'}, waiting {sleep_for:.0f}s")
                time.sleep(sleep_for)
                continue
            except ModelUnavailable as e:
                last_err = e
                if switch("is not available to this key"):
                    continue
                raise LLMError(f"{e}. Set LLM_MODEL to a model your key can use.")
            except requests.Timeout as e:
                last_err = e
                self.log(f"[llm] {tag}: no answer after {REQUEST_TIMEOUT}s")
                if switch("timed out"):
                    continue
            except (LLMError, requests.RequestException) as e:
                last_err = e
                self.log(f"[llm] {tag}: attempt {attempt} failed: {str(e)[:160]}")
                sleep_for = min(5.0 * attempt, MAX_SLEEP)
                self._check_deadline(tag, sleep_for)
                time.sleep(sleep_for)
                continue
            meta = {"model": model, "latency_s": round(time.time() - t0, 2), **usage}
            key = hashlib.sha256(f"{self.provider}|{model}|{system}|{prompt}".encode()).hexdigest()[:24]
            cpath = self.cache_dir / f"{key}.json"
            cpath.write_text(json.dumps({"meta": meta, "tag": tag, "system": system, "prompt": prompt,
                                         "raw": text, "parsed": parsed}, indent=1, ensure_ascii=False),
                             encoding="utf-8")
            self._cache_index()[self._content_key(system, prompt)] = str(cpath)
            self.calls.append({**meta, "tag": tag, "cached": False, "cache_file": str(cpath)})
            self.log(f"[llm] {tag}: done in {meta['latency_s']:.0f}s ({model})")
            return parsed
        raise LLMError(f"{tag} failed after {max_attempts} attempts: {str(last_err)[:300]}")

    def _check_deadline(self, tag: str, extra: float) -> None:
        if self.deadline and time.time() + extra > self.deadline:
            raise LLMError(f"{tag}: run time budget used up; falling back to rules for the rest")

    @staticmethod
    def _content_key(system: str, prompt: str) -> str:
        return hashlib.sha256(f"{system}|{prompt}".encode()).hexdigest()

    def _cache_index(self) -> dict:
        """Saved answers are reused for an identical prompt even if the model changed."""
        if getattr(self, "_index", None) is None:
            self._index = {}
            for f in self.cache_dir.glob("*.json"):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    self._index[self._content_key(d["system"], d["prompt"])] = str(f)
                except Exception:  # noqa: BLE001
                    continue
        return self._index

    def _thinking_config(self, model: str) -> dict | None:
        level = self.thinking
        if level in ("", "default") or self._no_thinking_cfg.get(model):
            return None
        m = re.search(r"gemini-(\d+)(?:\.(\d+))?", model)
        major = int(m.group(1)) if m else 0
        if major >= 3 or "latest" in model:
            return {"thinkingLevel": level}
        if major == 2:  # 2.x models use a token budget instead of a level
            return {"thinkingBudget": 0 if level in ("minimal", "off") else 1024}
        return None

    def _gemini(self, system: str, prompt: str, model: str):
        gen = {"temperature": self.temperature, "responseMimeType": "application/json"}
        tc = self._thinking_config(model)
        if tc:
            gen["thinkingConfig"] = tc
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": gen}
        r = requests.post(f"{self.base_url}/models/{model}:generateContent",
                          params={"key": self.api_key}, json=body, timeout=REQUEST_TIMEOUT)
        if r.status_code == 400 and tc and "thinking" in r.text.lower():
            # this model does not accept the thinking setting: remember and resend without it
            self._no_thinking_cfg[model] = True
            self.log(f"[llm] {model} does not accept thinking setting; using its default")
            return self._gemini(system, prompt, model)
        if r.status_code in (429, 500, 503):
            retry = None
            m = re.search(r'"retryDelay":\s*"(\d+)', r.text)
            if m:
                retry = float(m.group(1)) + 1
            daily = r.status_code == 429 and ("PerDay" in r.text or (retry or 0) > 120)
            raise _RateLimited(r.text[:2000], retry, overloaded=r.status_code in (500, 503), daily_quota=daily)
        if r.status_code == 404 or "no longer available" in r.text:
            raise ModelUnavailable(f"Gemini model '{model}' unavailable: {r.text[:300]}")
        if r.status_code >= 400:
            raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError):
            raise LLMError(f"Gemini empty response: {json.dumps(data)[:400]}")
        u = data.get("usageMetadata", {})
        return text, {"input_tokens": u.get("promptTokenCount", 0),
                      "output_tokens": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)}

    def _openai(self, system: str, prompt: str, model: str):
        body = {"model": model, "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}]}
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=300,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code in (429, 503, 500):
            ra = r.headers.get("retry-after")
            ra_f = float(ra) if ra and ra.replace(".", "").isdigit() else None
            raise _RateLimited(r.text[:2000], ra_f, overloaded=r.status_code in (500, 503),
                               daily_quota=r.status_code == 429 and ((ra_f or 0) > 120 or "per day" in r.text.lower()))
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()
        u = data.get("usage", {})
        return data["choices"][0]["message"]["content"], {
            "input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}

    def usage_summary(self) -> dict:
        live = [c for c in self.calls if not c.get("cached")]
        return {
            "model": self.describe(),
            "calls_total": len(self.calls),
            "calls_live": len(live),
            "input_tokens": sum(c.get("input_tokens", 0) for c in self.calls),
            "output_tokens": sum(c.get("output_tokens", 0) for c in self.calls),
            "latency_s": round(sum(c.get("latency_s", 0) for c in live), 1),
        }


class _RateLimited(Exception):
    def __init__(self, msg, retry_after=None, overloaded=False, daily_quota=False):
        super().__init__(msg)
        self.retry_after = retry_after
        self.overloaded = overloaded
        self.daily_quota = daily_quota
