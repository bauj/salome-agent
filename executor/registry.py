from typing import Callable


class ToolRegistry:
    """One module's tools, declared once instead of across five parallel dicts.

    shaper and smesh get their own instances of this class; nothing in here
    knows about geom.
    """

    def __init__(self, valid_categories: "set[str] | None" = None):
        self._valid = set(valid_categories or ())
        self._entries: dict[str, dict] = {}

    def tool(self, name: str, *, category: str, description: str, schema: dict,
             arg_types: "dict[str, Callable] | None" = None):
        if name in self._entries:
            raise ValueError(f"tool {name!r} already registered")
        if self._valid and category not in self._valid:
            raise ValueError(f"tool {name!r}: unknown category {category!r}")
        if not description:
            raise ValueError(f"tool {name!r}: empty description")

        def decorate(fn):
            self._entries[name] = {"fn": fn, "category": category,
                                   "description": description, "schema": schema,
                                   "arg_types": dict(arg_types or {})}
            return fn

        return decorate

    @property
    def names(self) -> list:
        return list(self._entries)

    def handlers(self) -> dict:
        return {n: e["fn"] for n, e in self._entries.items()}

    def categories(self) -> dict:
        return {n: e["category"] for n, e in self._entries.items()}

    def arg_types(self) -> dict:
        return {n: e["arg_types"] for n, e in self._entries.items()}

    def schemas(self) -> dict:
        return {n: e["schema"] for n, e in self._entries.items()}

    def descriptions(self) -> dict:
        return {n: e["description"] for n, e in self._entries.items()}

    def list_tools(self) -> list:
        return [{"name": n, "parameters": e["schema"] or {"type": "object", "properties": {}},
                 "description": e["description"], "category": e["category"]}
                for n, e in self._entries.items()]

    def coerce_args(self, name: str, args: dict) -> dict:
        types = self._entries.get(name, {}).get("arg_types")
        if not types:
            return args
        coerced = dict(args)
        for key, cast in types.items():
            if key in coerced:
                coerced[key] = cast(coerced[key])
        return coerced

    def call(self, executor, name: str, args: dict) -> dict:
        entry = self._entries.get(name)
        if entry is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        return entry["fn"](executor, self.coerce_args(name, args))
