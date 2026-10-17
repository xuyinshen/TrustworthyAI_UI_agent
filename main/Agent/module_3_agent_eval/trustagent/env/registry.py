"""Tool registry — the only place an agent's capabilities are declared.

Each tool carries its JSON-argument schema (so tool-call correctness can be
checked the way BFCL checks it: exact tool name + AST-level argument match)
*and* its security scope, because the sandbox needs both.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    params: Dict[str, str] = field(default_factory=dict)   # arg name -> type/description
    required: List[str] = field(default_factory=list)
    scope: str = "read"            # read | write | network | spend | destructive | admin
    irreversible: bool = False
    requires_confirmation: bool = False
    cost: float = 0.0

    @property
    def is_write(self) -> bool:
        return self.scope in ("write", "network", "spend", "destructive", "admin")

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {k: {"type": _json_type(v), "description": v}
                           for k, v in self.params.items()},
            "required": list(self.required or self.params.keys()),
            "additionalProperties": False,
        }

    def openai_tool(self) -> Dict[str, Any]:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.json_schema()}}


def _json_type(decl: str) -> str:
    d = (decl or "").lower()
    if "int" in d or "count" in d or "num" in d:
        return "integer"
    if "float" in d or "rate" in d or "ratio" in d:
        return "number"
    if "list" in d or "json" in d or "array" in d:
        return "array"
    if "bool" in d:
        return "boolean"
    return "string"


# ==========================================================================
class ToolRegistry:
    """Name -> Tool, with a rename table used by the schema-perturbation attack."""

    def __init__(self, tools: Iterable[Tool] = ()):
        self._tools: Dict[str, Tool] = {}
        self._renames: Dict[str, str] = {}       # published name -> real name
        for t in tools:
            self.register(t)

    # -- registration -----------------------------------------------------
    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def add(self, name: str, description: str, fn: Callable[..., Any], *,
            params: Optional[Dict[str, str]] = None, required: Optional[List[str]] = None,
            scope: str = "read", irreversible: bool = False,
            requires_confirmation: bool = False, cost: float = 0.0) -> Tool:
        return self.register(Tool(name=name, description=description, fn=fn,
                                  params=dict(params or {}), required=list(required or []),
                                  scope=scope, irreversible=irreversible,
                                  requires_confirmation=requires_confirmation, cost=cost))

    # -- access -----------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return str(name) in self._tools or str(name) in self._renames

    def get(self, name: str) -> Optional[Tool]:
        real = self._renames.get(name, name)
        return self._tools.get(real)

    def publishes(self, name: str) -> bool:
        """Is ``name`` actually declared in the schema an agent was shown?

        After a rename the real name still resolves internally, but it is no
        longer *published* — an agent calling it is operating from memorised
        tool names rather than the schema in front of it, which is exactly the
        grounding failure the rename attack is designed to expose.
        """
        return name in self.names()

    def published_name(self, real_name: str) -> str:
        for pub, real in self._renames.items():
            if real == real_name:
                return pub
        return real_name

    def names(self) -> List[str]:
        return sorted([*self._renames.keys(),
                       *[n for n in self._tools if n not in self._renames.values()]])

    def tools(self) -> List[Tool]:
        return list(self._tools.values())

    # -- perturbation hooks -----------------------------------------------
    def renamed(self, mapping: Dict[str, str]) -> "ToolRegistry":
        """Return a *view* where published tool names differ (schema-rename attack).

        The agent only ever sees the new names; the sandbox still routes to the
        real implementation, which is exactly the point — an agent that hard
        codes tool names instead of reading the schema breaks here.
        """
        clone = ToolRegistry()
        clone._tools = self._tools
        for published, real in mapping.items():
            if real in self._tools:
                clone._renames[published] = real
        return clone

    def with_reordered_params(self) -> "ToolRegistry":
        """Same tools, argument order shuffled — probes order-sensitive agents."""
        clone = ToolRegistry()
        for t in self._tools.values():
            keys = list(t.params.keys())
            shuffled = keys[::-1]
            clone.register(Tool(
                name=t.name, description=t.description, fn=t.fn,
                params={k: t.params[k] for k in shuffled},
                required=list(t.required), scope=t.scope,
                irreversible=t.irreversible,
                requires_confirmation=t.requires_confirmation, cost=t.cost))
        return clone

    # -- prompt rendering --------------------------------------------------
    def prompt_block(self) -> str:
        """The `- name: description [params]` block the ReAct prompt embeds.

        Only argument *names* go in the brackets; types and examples live in
        :meth:`schemas` / :meth:`openai_tools` so the prompt stays parseable.
        """
        lines = []
        for pub in self.names():
            t = self.get(pub)
            if t is None:
                continue
            args = ", ".join(t.params.keys())
            flag = " (irreversible — confirm first)" if (t.irreversible or t.requires_confirmation) else ""
            lines.append(f"- {pub}: {t.description}{flag}"
                         + (f" [{args}]" if args else ""))
        return "\n".join(lines)

    def schemas(self) -> Dict[str, Dict[str, Any]]:
        return {self.published_name(t.name): t.json_schema() for t in self._tools.values()}

    def openai_tools(self) -> List[Dict[str, Any]]:
        return [{"type": "function",
                 "function": {"name": self.published_name(t.name),
                              "description": t.description,
                              "parameters": t.json_schema()}}
                for t in self._tools.values()]

    def json(self) -> str:
        return json.dumps(self.schemas(), ensure_ascii=False, indent=2)
