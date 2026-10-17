"""Model layer: one seam, three implementations.

Every agent in this framework talks to a model only through
:meth:`LLMClient.complete`.  That single seam is what makes the harness
*agent-agnostic and provider-agnostic*: swapping the deterministic
:class:`MockLLM` for :class:`OllamaLLM` changes nothing in the agent code.

Why a ``MockLLM`` at all?  So the whole pipeline — prompt assembly, output
parsing, sandbox ACLs, injection handling, all six metric families, the
scorecard and the export — is runnable and *assertable* offline, on a laptop,
with no GPU and no network.  A mock is not a measurement of a real model, so
every report it produces is labelled ``mode=mock``.  Run the same benchmark
against Ollama or an OpenAI-compatible endpoint to get real numbers.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str = "stop"


def _approx_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~1 token per 4 chars)."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def model_family(model: str, transport: str = "custom") -> str:
    """Weight family, not transport — the judge guardrail needs *this*.

    ``qwen3.6:27b``, ``Qwen3.8-27B`` and ``unsloth/qwen3.6-27b-gguf`` all map to
    ``qwen``; an unparseable name falls back to the transport label so the
    guardrail degrades loudly instead of silently treating everything as distinct.
    """
    seg = (model or "").replace("\\", "/").rstrip("/").split("/")[-1].lower()
    seg = seg.strip(":._- /")
    m = re.match(r"[a-z]+", seg)
    head = m.group(0) if m else ""
    return head if len(head) >= 3 else (transport or "custom")


# ==========================================================================
class LLMClient(ABC):
    """Abstract model seam."""

    name: str = "abstract"
    family: str = "unknown"

    @abstractmethod
    def complete(self, prompt: str, *, stop: Optional[List[str]] = None,
                 meta: Optional[Dict[str, Any]] = None) -> Completion:
        ...

    def close(self) -> None:                                    # pragma: no cover
        pass


# ==========================================================================
@dataclass
class ScriptStep:
    """One deterministic turn of a scripted mock agent."""
    thought: str = ""
    action: Optional[str] = None
    action_input: Dict[str, Any] = field(default_factory=dict)
    final: Optional[str] = None

    def render(self) -> str:
        parts = []
        if self.thought:
            parts.append(f"Thought: {self.thought}")
        if self.final is not None:
            parts.append(f"Final Answer: {self.final}")
        elif self.action is not None:
            parts.append(f"Action: {self.action}")
            parts.append("Action Input: " + json.dumps(self.action_input, ensure_ascii=False))
        return "\n".join(parts)


_TOOL_LINE = re.compile(r"^-\s*(?P<name>[a-zA-Z0-9_.\-]+)\s*:\s*(?P<desc>[^\[\n]*)"
                        r"(?:\[(?P<params>[^\]]*)\])?", re.M)
_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "be", "it", "its", "this", "that", "as", "at", "by", "from", "you", "your", "we",
    "before", "after", "not", "no", "all", "any", "into", "then", "so", "if", "can",
}

#: The term extractor below is ASCII-only, so a Chinese instruction would
#: contribute no tokens, every tool would score zero, and the mock would echo the
#: question instead of reading anything.  This bridge maps wording an operator
#: uses onto the English words already in the tool names, descriptions and
#: parameter names.  Deliberately no entry for "why": asking for a cause is not
#: asking for the spec library.
ZH_TERMS: Dict[str, tuple] = {
    "告警": ("alarm", "log", "alarms"),
    "警报": ("alarm", "alarms"),
    "工单": ("alarm", "log"),
    "策略": ("alert", "policy", "a1"),
    "小区": ("cell", "kpi"),
    "基站": ("cell", "kpi"),
    "利用率": ("kpi", "cell"),
    "负荷": ("kpi", "cell"),
    "负载": ("kpi", "cell", "balancer", "load"),
    "拥塞": ("kpi", "cell"),
    "吞吐": ("kpi", "cell"),
    "速率": ("kpi", "cell"),
    "切换": ("kpi", "cell"),
    "干扰": ("kpi", "cell"),
    "信号": ("kpi", "cell"),
    "应用": ("xapp", "list"),
    "插件": ("xapp", "list"),
    "遥测": ("telemetry", "xapp"),
    "性能": ("telemetry", "xapp"),
    "文档": ("specs", "search"),
    "规范": ("specs", "search"),
    "标准": ("specs", "search"),
    "文件": ("file", "read"),
    "预算": ("cost", "budget", "spent"),
}
_INSTRUCTION = re.compile(r"Question/task:\s*(?P<body>.+?)"
                          r"(?:\nObservation:|\nAnswer with exactly|\n\n|$)",
                          re.S | re.I)


def _zh_terms(instr: str) -> set:
    """English tool-side tokens the operator's Chinese wording points at."""
    out = set()
    for cue, tokens in ZH_TERMS.items():
        if cue in instr:
            out.update(tokens)
    return out


def _stem(terms: set) -> set:
    """Fold plurals so "alarms" reaches ``read_alarm_log`` and "cells" ``read_cell_kpi``.

    The scorer below compares whole word tokens; without this an English
    operator's plural simply matched nothing, and the run took whatever tool
    shared a stray singular with the sentence.
    """
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in terms}


class MockLLM(LLMClient):
    """Deterministic stand-in model.

    Two behaviours, tried in order:

    1. **Scripted** — if ``meta["task_id"]`` has an entry in ``scripts``, replay
       it step by step.  Used to exercise exact trajectories (multi-step tool
       chains, injection-following behaviour, hallucinated tools).
    2. **Rule planner** — otherwise parse the tool block out of the prompt and
       pick the tool whose name/description overlaps the instruction most,
       using a hash of the *full prompt* as a deterministic pseudo-random tie
       breaker.  Perturbing the instruction therefore genuinely changes the
       plan, which is what makes the robustness curve meaningful without a GPU.
    """

    name = "mock"
    family = "mock"

    def __init__(self, scripts: Optional[Dict[str, List[Any]]] = None,
                 provider: Optional[Callable[[str], Optional[List[Any]]]] = None,
                 seed: int = 7, temperature_like: float = 0.0):
        self.scripts = {k: [self._as_step(s) for s in v] for k, v in (scripts or {}).items()}
        self._provider = provider
        self._seed = seed
        self._temp = float(temperature_like)
        self._cursor: Dict[str, int] = {}

    @staticmethod
    def _as_step(s: Any) -> ScriptStep:
        if isinstance(s, ScriptStep):
            return s
        return ScriptStep(thought=str(s.get("thought", "")),
                          action=s.get("action"),
                          action_input=dict(s.get("action_input", {}) or {}),
                          final=s.get("final"))

    # -- deterministic helpers -------------------------------------------
    def _hash01(self, text: str) -> float:
        h = hashlib.sha256(f"{self._seed}|{text}".encode("utf-8")).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def reset(self) -> None:
        self._cursor.clear()

    # -- main entry -------------------------------------------------------
    def complete(self, prompt: str, *, stop: Optional[List[str]] = None,
                 meta: Optional[Dict[str, Any]] = None) -> Completion:
        t0 = time.perf_counter()
        meta = meta or {}
        task_id = str(meta.get("task_id", ""))
        policy = str(meta.get("policy", "eager"))
        script_key = f"{task_id}#{policy}"
        if script_key in self.scripts:
            key = script_key
        elif task_id in self.scripts:
            key = task_id
        else:
            key = ""

        script = self.scripts.get(key)
        if script is None and self._provider is not None:
            provided = self._provider(task_id)
            if provided:
                script = [self._as_step(s) for s in provided]
                self.scripts[task_id] = script
                key = task_id

        if script and not isinstance(script[0], ScriptStep):
            script = [self._as_step(s) for s in script]
            self.scripts[key] = script

        if script:
            i = self._cursor.get(key, 0)
            step = script[min(i, len(script) - 1)]
            if i < len(script):
                self._cursor[key] = i + 1
            text = step.render()
        else:
            text = self._plan(prompt)

        comp = Completion(text=text, prompt_tokens=_approx_tokens(prompt),
                          completion_tokens=_approx_tokens(text), model=self.name,
                          latency_ms=(time.perf_counter() - t0) * 1000.0)
        return comp

    # -- rule planner ------------------------------------------------------
    def _plan(self, prompt: str) -> str:
        tools = {m.group("name"): (m.group("desc") or "", m.group("params") or "")
                 for m in _TOOL_LINE.finditer(prompt)}
        m = _INSTRUCTION.search(prompt)
        instr = (m.group("body") if m else prompt).strip()

        observations = re.findall(r"Observation:\s*(.+?)(?:\n(?:Thought|Action|Question)|$)",
                                  prompt, re.S)
        for line in reversed(observations):
            line = line.strip()
            if line and line not in ("", "None"):
                # A tool already returned something -> report it and stop.
                return f"Thought: I have a result from the last tool.\nFinal Answer: {line[:220]}"

        if not tools:
            return f"Thought: No tools available.\nFinal Answer: {instr[:200]}"

        scored = []
        instr_terms = _stem((set(re.findall(r"[a-z0-9]+", instr.lower())) - _STOP
                             | _zh_terms(instr)))
        for name, (desc, params) in tools.items():
            pnames = {p.split(":")[0].strip().lower()
                      for p in params.split(",") if p.strip()}
            pterms = _stem({t for p in pnames for t in re.findall(r"[a-z0-9]+", p)})
            dterms = _stem(set(re.findall(r"[a-z0-9]+", (name + " " + desc).lower()))
                           - _STOP)
            nterms = _stem(set(re.findall(r"[a-z0-9]+", name.lower())))
            args = self._fill_args(params, instr)
            # an argument the operator never named would go through as a
            # placeholder — 'Which cell raises most alarms?' must not turn into a
            # lookup of a cell called 'unknown', when a log of all of them exists
            holes = sum(1 for v in args.values() if str(v) == "unknown")
            overlap = (3 * len(instr_terms & nterms)
                       + 2 * len(instr_terms & pterms)
                       + 1 * len(instr_terms & dterms)
                       - 3 * holes)
            jitter = self._temp * (self._hash01(prompt + name) - 0.5)
            tie = self._hash01(prompt + "|" + name) * 0.001
            scored.append((overlap + jitter + tie, name, params, args))
        scored.sort(key=lambda s: -s[0])
        best_score, best_name, _best_params, best_args = scored[0]

        if best_score < 0.5:
            return ("Thought: Nothing matches the request well enough to act on.\n"
                    f"Final Answer: {instr[:200]}")

        return (f"Thought: Use {best_name} to answer this request.\n"
                f"Action: {best_name}\n"
                "Action Input: " + json.dumps(best_args, ensure_ascii=False))

    @staticmethod
    def _fill_args(params: str, instr: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        cands = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.\-/]*", instr)
        cnorm = [(c, re.sub(r"[^a-z0-9]", "", c.lower())) for c in cands]
        for key in [x.strip() for x in params.split(",") if x.strip()]:
            low = re.sub(r"[^a-z0-9]", "", key.lower())
            wants_value = any(t in low for t in ("id", "name", "path", "target"))
            hit = None
            for raw, norm in cnorm:
                if not norm:
                    continue
                shared = len(low[:4]) if low[:4] == norm[:4] else 0
                if not (shared or (len(low) >= 3 and low in norm)
                        or (len(norm) >= 4 and norm in low)):
                    continue
                if wants_value and not re.search(r"\d|[.\-/]", raw):
                    # "cell" names the dimension, it is not a cell's identifier —
                    # filling it would filter the sandbox down to nothing
                    continue
                hit = raw
                break
            if hit is None:
                nums = [c for c, n in cnorm if n.isdigit()]
                if "id" in low or "name" in low or "path" in low or "target" in low:
                    hit = "unknown"
                elif any(t in low for t in ("count", "num", "k", "limit", "amount", "budget")):
                    hit = nums[0] if nums else "10"
                else:
                    # nothing in the instruction names this field: leave it at the
                    # tool's default instead of filtering on a word that happens
                    # to lead the sentence, which would read back as an empty set
                    hit = ""
            out[key] = hit
        return out


# ==========================================================================
class OllamaLLM(LLMClient):
    """Local Ollama models (Qwen2.5 / Llama3.1 ...) over Ollama's REST API."""

    family = "ollama"                                # transport fallback

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434",
                 options: Optional[Dict[str, Any]] = None, timeout: float = 120.0,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        import urllib.request                              # lazy: offline-safe
        self._urlopen = urllib.request.urlopen
        self.model = model
        self.name = f"ollama:{model}"
        self.family = model_family(model, "ollama")
        self.host = host.rstrip("/")
        opts = dict(options or {})
        opts.setdefault("temperature", 0.0 if temperature is None else float(temperature))
        if max_tokens:
            opts.setdefault("num_predict", int(max_tokens))
        self.options = opts
        self.timeout = timeout

    def complete(self, prompt: str, *, stop: Optional[List[str]] = None,
                 meta: Optional[Dict[str, Any]] = None) -> Completion:
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": dict(self.options)}
        if stop:
            payload["stop"] = list(stop)
        req = self._urlopen(self.host + "/api/generate",
                            data=json.dumps(payload).encode("utf-8"),
                            timeout=self.timeout)
        with req as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data.get("response", "")
        return Completion(text=text,
                          prompt_tokens=int(data.get("prompt_eval_count", 0)) or _approx_tokens(prompt),
                          completion_tokens=int(data.get("eval_count", 0)) or _approx_tokens(text),
                          latency_ms=float(data.get("total_duration", 0)) / 1e6,
                          model=self.name)


# ==========================================================================
class OpenAICompatLLM(LLMClient):
    """Any OpenAI-compatible endpoint (vLLM, LM Studio, bionic, llama.cpp server, ...).

    Local GGUF runners differ in UI but agree on the wire: ``POST {base}/chat/completions``
    with an optional bearer token, so one client covers them all — only ``base_url``
    and ``model`` change.
    """

    family = "openai-compat"                         # transport fallback

    def __init__(self, model: str = "qwen2.5-7b-instruct", base_url: str = "http://localhost:8000/v1",
                 api_key: str = "", temperature: float = 0.0, max_tokens: int = 512,
                 timeout: float = 120.0):
        import urllib.request
        self._urlopen = urllib.request.urlopen
        self.model = model
        self.name = f"openai:{model}"
        self.family = model_family(model, "openai-compat")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, prompt: str, *, stop: Optional[List[str]] = None,
                 meta: Optional[Dict[str, Any]] = None) -> Completion:
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": self.temperature,
                   "max_tokens": self.max_tokens}
        if stop:
            payload["stop"] = list(stop)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = self._urlopen(self.base_url + "/chat/completions",
                            data=json.dumps(payload).encode("utf-8"),
                            headers=headers, timeout=self.timeout)
        with req as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content", "")
        usage = data.get("usage") or {}
        return Completion(text=text,
                          prompt_tokens=int(usage.get("prompt_tokens", 0)) or _approx_tokens(prompt),
                          completion_tokens=int(usage.get("completion_tokens", 0)) or _approx_tokens(text),
                          model=self.name)


# ==========================================================================
def make_llm(spec: str = "mock", **kw: Any) -> LLMClient:
    """Factory used by the runner and the UI. ``spec`` is ``mock|ollama|openai``."""
    spec = (spec or "mock").lower()
    if spec == "mock":
        return MockLLM(scripts=kw.get("scripts"), provider=kw.get("provider"),
                       seed=kw.get("seed", 7),
                       temperature_like=kw.get("temperature_like", 0.0))
    if spec.startswith("ollama"):
        return OllamaLLM(model=kw.get("model", "qwen2.5:7b"),
                         host=kw.get("host", "http://localhost:11434"),
                         options=kw.get("options"),
                         temperature=kw.get("temperature"),
                         max_tokens=kw.get("max_tokens"),
                         timeout=kw.get("timeout", 120.0))
    if spec.startswith("openai"):
        return OpenAICompatLLM(model=kw.get("model", "qwen2.5-7b-instruct"),
                               base_url=kw.get("base_url", "http://localhost:8000/v1"),
                               api_key=kw.get("api_key", ""),
                               temperature=float(kw.get("temperature") or 0.0),
                               max_tokens=int(kw.get("max_tokens") or 512),
                               timeout=kw.get("timeout", 120.0))
    raise ValueError(f"unknown llm spec: {spec}")


def list_models(spec: str = "openai", *, base_url: str = "", host: str = "",
                api_key: str = "", timeout: float = 8.0) -> List[str]:
    """Ask a running endpoint which models it serves.

    Lets the UI fill the model dropdown from the server's own inventory instead
    of trusting a hand-typed id — the one place where guessing a string is
    expensive (a typo costs a whole evaluation run of connection errors).
    """
    import json as _json
    import urllib.request

    spec = (spec or "openai").lower()
    if spec.startswith("ollama"):
        url = (host or "http://localhost:11434").rstrip("/") + "/api/tags"
        want = lambda d: [m.get("name") for m in (d.get("models") or []) if m.get("name")]
    else:
        base = (base_url or "http://localhost:8000/v1").rstrip("/")
        if base.endswith("/v1"):
            url = base + "/models"
        else:
            url = base + "/v1/models"
        want = lambda d: [m.get("id") or m.get("name") for m in (d.get("data") or [])
                          if m.get("id") or m.get("name")]
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"{url} is not answering ({type(e).__name__}: {e})") from e
    return [str(m) for m in want(payload) if m]
