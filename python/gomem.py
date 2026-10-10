"""gomem — Python client for the gomem bridge (ctypes over libgomem).

Build the library first::

    sh python/build.sh            # -> python/libgomem.{dylib,so} / gomem.dll

Quick tour::

    from gomem import Memory

    mem = Memory("/tmp/my-agent.db", provider="offline")
    mem.remember(kind="fact", title="...", body="...", tags=["db"])
    hits = mem.recall("speed up slow database queries", top_k=5)
    for h in hits:
        print(h["score"], h["title"])
    mem.stats(); mem.close()

Every method returns plain Python data (dicts/lists). Failures raise
:class:`GomemError` whose message is the Go-side error text.
"""
from __future__ import annotations

import asyncio
import ctypes
import functools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from ctypes import c_char_p, c_longlong
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Memory", "GomemError", "load_library", "TIER_HOT", "TIER_WARM", "TIER_COLD"]
__version__ = "0.1.0"

# Memory tiers: 2 = hot (new/active), 1 = warm, 0 = cold (decayed; first in
# line for eviction). ``tier=None`` (the default) leaves the choice to the
# store: a new item starts hot and updating an existing item keeps its
# stored tier. Pass a tier explicitly (e.g. ``TIER_COLD``) to set it.
TIER_HOT = 2
TIER_WARM = 1
TIER_COLD = 0

# Shared thread pool that runs the (blocking) ctypes bridge calls off
# the event loop for the async API. Created lazily on first use so merely
# importing this module has no side effects.
_SHARED_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _resolve_executor(custom: Optional[ThreadPoolExecutor]) -> ThreadPoolExecutor:
    """Return the pool to run blocking calls in: ``custom`` if given, else the shared one."""
    global _SHARED_EXECUTOR
    if custom is not None:
        return custom
    if _SHARED_EXECUTOR is None:
        _SHARED_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 4))
    return _SHARED_EXECUTOR


class GomemError(RuntimeError):
    """Raised when a bridge call reports an error."""


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _native_lib_name() -> str:
    # Matches the per-GOOS output name chosen by python/build.sh, so a
    # library built for another OS left in the tree is never picked up.
    if sys.platform == "darwin":
        return "libgomem.dylib"
    if sys.platform == "win32":
        return "gomem.dll"
    return "libgomem.so"


def load_library(path: Optional[str] = None) -> ctypes.CDLL:
    """Locate and load the bridge shared object.

    Resolution order: explicit ``path`` argument, ``GOMEM_LIB``
    environment variable, then the platform-native build output next
    to this file (``libgomem.dylib`` / ``libgomem.so`` / ``gomem.dll``).
    """
    if path:
        lib_path = path
    elif os.environ.get("GOMEM_LIB"):
        lib_path = os.environ["GOMEM_LIB"]
    else:
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _native_lib_name())
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"{lib_path} not found. Run `sh python/build.sh` from the "
                "repo root, or point GOMEM_LIB at the built library."
            )
    lib = ctypes.CDLL(lib_path)
    _wire_signatures(lib)
    return lib


def _wire_signatures(lib: ctypes.CDLL) -> None:
    # Results come back as raw pointers (c_void_p) so the exact
    # Go-malloc'd address survives for GomemFreeString; converting
    # to c_char_p up front would hand us a *copy* and orphan the
    # original allocation.
    lib.GomemFreeString.argtypes = [ctypes.c_void_p]
    lib.GomemFreeString.restype = None

    lib.GomemNew.argtypes = [c_char_p]
    lib.GomemNew.restype = ctypes.c_void_p

    lib.GomemRemember.argtypes = [c_longlong, c_char_p]
    lib.GomemRemember.restype = ctypes.c_void_p

    lib.GomemRememberMany.argtypes = [c_longlong, c_char_p]
    lib.GomemRememberMany.restype = ctypes.c_void_p

    lib.GomemGet.argtypes = [c_longlong, c_char_p]
    lib.GomemGet.restype = ctypes.c_void_p

    lib.GomemDelete.argtypes = [c_longlong, c_char_p]
    lib.GomemDelete.restype = ctypes.c_void_p

    lib.GomemTouch.argtypes = [c_longlong, c_char_p]
    lib.GomemTouch.restype = ctypes.c_void_p

    lib.GomemRecall.argtypes = [c_longlong, c_char_p]
    lib.GomemRecall.restype = ctypes.c_void_p

    lib.GomemReflect.argtypes = [c_longlong, c_char_p]
    lib.GomemReflect.restype = ctypes.c_void_p

    lib.GomemStats.argtypes = [c_longlong]
    lib.GomemStats.restype = ctypes.c_void_p

    lib.GomemClose.argtypes = [c_longlong]
    lib.GomemClose.restype = ctypes.c_void_p


def _invoke(lib: ctypes.CDLL, fn: Any, *args: Any) -> Dict[str, Any]:
    """Call a bridge function, decode its JSON payload, free the buffer."""
    ptr = fn(*args)
    try:
        doc = json.loads(ctypes.string_at(ptr).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface any decode surprise
        raise GomemError(f"undecodable bridge response: {exc!r}") from exc
    finally:
        lib.GomemFreeString(ptr)
    if isinstance(doc, dict) and "error" in doc and "handle" not in doc:
        raise GomemError(str(doc["error"]))
    return doc if isinstance(doc, dict) else {"value": doc}


def _enc(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def _item_doc(
    *,
    kind: str = "fact",
    title: str,
    body: str = "",
    payload: Any = None,
    tags: Optional[List[str]] = None,
    tier: Optional[int] = None,
    supersedes: str = "",
    item_id: str = "",
    source: str = "run",
    run_id: str = "",
    node_role: str = "",
    confidence: float = 0.8,
    evidence_refs: Optional[List[str]] = None,
    created_at: str = "",
) -> Dict[str, Any]:
    """Build the wire document for one item."""
    item: Dict[str, Any] = {
        "kind": kind,
        "title": title,
        "body": body,
    }
    if tier is not None:
        item["tier"] = tier
    if item_id:
        item["id"] = item_id
    if payload is not None:
        item["payload"] = payload
    if tags:
        item["tags"] = list(tags)
    if supersedes:
        item["supersedes"] = supersedes
    prov: Dict[str, Any] = {"source": source, "confidence": confidence}
    if run_id:
        prov["run_id"] = run_id
    if node_role:
        prov["node_role"] = node_role
    if evidence_refs:
        prov["evidence_refs"] = list(evidence_refs)
    if created_at:
        prov["created_at"] = created_at
    item["provenance"] = prov
    return item


# ---------------------------------------------------------------------------
# High-level handle
# ---------------------------------------------------------------------------

class Memory:
    """A live gomem handle. Use as a context manager for auto-close.

    ``recovered_from_corrupt`` is True when the store was rebuilt from a
    corrupt file (Go's ``New`` returns a usable handle with ``ErrCorrupt``).
    """

    def __init__(
        self,
        path: str = "gomem.db",
        *,
        lib: Optional[ctypes.CDLL] = None,
        # async: pool used to run blocking bridge calls off the event
        # loop; None (default) shares a module-level pool.
        executor: Optional[ThreadPoolExecutor] = None,
        # Every setting below except provider and log_level defaults to
        # None: it is then omitted, so Go's gomem.DefaultConfig() applies.
        # embedding
        provider: str = "offline",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dim: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        max_retries: Optional[int] = None,
        batch_size: Optional[int] = None,
        cache_dir: Optional[str] = None,
        send_dimensions: Optional[bool] = None,
        embed_length_threshold: Optional[int] = None,
        # "sqlite" scans vectors per query; "memory" keeps them in RAM
        dense_index: Optional[str] = None,
        # retrieval defaults
        top_k: Optional[int] = None,
        half_life_days: Optional[float] = None,
        dedup_jaccard: Optional[float] = None,
        # "legacy", "minmax" or "rrf": how leg scores are normalized
        fusion_mode: Optional[str] = None,
        # fusion weights
        alpha_dense: Optional[float] = None,
        beta_sparse: Optional[float] = None,
        gamma_salience: Optional[float] = None,
        # salience weights
        w_confidence: Optional[float] = None,
        w_frequency: Optional[float] = None,
        w_recency: Optional[float] = None,
        freq_cap: Optional[int] = None,
        # housekeeping / limits
        eviction_age_days: Optional[int] = None,
        max_items: Optional[int] = None,
        max_payload_kb: Optional[int] = None,
        log_level: str = "warn",
    ) -> None:
        self._lib = lib or load_library()
        settings: Dict[str, Any] = {
            "dim": dim,
            "api_url": api_url,
            "api_key": api_key,
            "model": model,
            "timeout_ms": timeout_ms,
            "max_retries": max_retries,
            "batch_size": batch_size,
            "cache_dir": cache_dir,
            "send_dimensions": send_dimensions,
            "embed_length_threshold": embed_length_threshold,
            "dense_index": dense_index,
            "top_k": top_k,
            "half_life_days": half_life_days,
            "dedup_jaccard": dedup_jaccard,
            "fusion_mode": fusion_mode,
            "alpha_dense": alpha_dense,
            "beta_sparse": beta_sparse,
            "gamma_salience": gamma_salience,
            "w_confidence": w_confidence,
            "w_frequency": w_frequency,
            "w_recency": w_recency,
            "freq_cap": freq_cap,
            "eviction_age_days": eviction_age_days,
            "max_items": max_items,
            "max_payload_kb": max_payload_kb,
        }
        cfg: Dict[str, Any] = {"path": path, "provider": provider, "log_level": log_level}
        cfg.update((k, v) for k, v in settings.items() if v is not None)
        doc = _invoke(self._lib, self._lib.GomemNew, _enc(cfg))
        if "handle" not in doc:
            raise GomemError(str(doc.get("error", "open returned no handle")))
        self._handle = int(doc["handle"])
        self.recovered_from_corrupt = "error" in doc
        self.path = path
        self._executor = executor

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        _invoke(self._lib, self._lib.GomemClose, self._handle)
        self._closed = True

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- core verbs -------------------------------------------------------

    def remember(
        self,
        *,
        kind: str = "fact",
        title: str,
        body: str = "",
        payload: Any = None,
        tags: Optional[List[str]] = None,
        tier: Optional[int] = None,
        supersedes: str = "",
        item_id: str = "",
        # provenance
        source: str = "run",
        run_id: str = "",
        node_role: str = "",
        confidence: float = 0.8,
        evidence_refs: Optional[List[str]] = None,
        created_at: str = "",
    ) -> str:
        """Persist an item; returns its ULID.

        ``tier`` is 0 (cold), 1 (warm) or 2 (hot). Left as None, a new item
        starts hot and an update keeps the stored tier; ``tier=0`` stores a
        cold item.
        """
        item = _item_doc(
            kind=kind, title=title, body=body, payload=payload, tags=tags, tier=tier,
            supersedes=supersedes, item_id=item_id, source=source, run_id=run_id,
            node_role=node_role, confidence=confidence, evidence_refs=evidence_refs,
            created_at=created_at,
        )
        doc = _invoke(self._lib, self._lib.GomemRemember, self._handle, _enc(item))
        return str(doc["id"])

    def remember_many(self, items: List[Dict[str, Any]]) -> List[str]:
        """Persist items atomically (all or none); returns their ULIDs in order.

        Each item is a dict of the keyword arguments :meth:`remember` accepts
        (``title`` is required; ``id`` may be used for ``item_id``; ``tier``
        is optional as in :meth:`remember`).
        """
        docs = []
        for it in items:
            kw = dict(it)
            if "id" in kw:
                kw["item_id"] = kw.pop("id")
            docs.append(_item_doc(**kw))
        doc = _invoke(self._lib, self._lib.GomemRememberMany, self._handle, _enc(docs))
        return [str(i) for i in doc["ids"]]

    def get(self, item_id: str) -> Dict[str, Any]:
        """Retrieve a single item by id; returns its document."""
        return _invoke(self._lib, self._lib.GomemGet, self._handle, _enc(item_id))

    def delete(self, item_id: str) -> None:
        """Tombstone (soft-delete) the item with the given id."""
        _invoke(self._lib, self._lib.GomemDelete, self._handle, _enc(item_id))

    def touch(self, item_id: str) -> None:
        """Record an access to the item: bumps its salience / anti-decay."""
        _invoke(self._lib, self._lib.GomemTouch, self._handle, _enc(item_id))

    def recall(
        self,
        goal: str,
        *,
        top_k: int = 5,
        sub_goal: str = "",
        kinds: Optional[List[str]] = None,
        min_tier: int = 0,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        budget_tokens: int = 0,
        recency_half_life_days: float = 0.0,
        include_superseded: bool = False,
        dedup_jaccard_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval; returns hits best-first.

        ``min_tier`` keeps only items with tier >= min_tier (0 = no filter).
        """
        q: Dict[str, Any] = {"goal": goal, "top_k": top_k}
        if sub_goal:
            q["sub_goal"] = sub_goal
        if kinds:
            q["kinds"] = list(kinds)
        if min_tier:
            q["min_tier"] = min_tier
        if tags:
            q["tags"] = list(tags)
        if min_confidence:
            q["min_confidence"] = min_confidence
        if budget_tokens:
            q["budget_tokens"] = budget_tokens
        if recency_half_life_days:
            q["recency_half_life_days"] = recency_half_life_days
        if include_superseded:
            q["include_superseded"] = True
        if dedup_jaccard_threshold:
            q["dedup_jaccard_threshold"] = dedup_jaccard_threshold
        doc = _invoke(self._lib, self._lib.GomemRecall, self._handle, _enc(q))
        return list(doc.get("hits", []))

    def reflect(self, trigger: str = "run_end") -> Dict[str, Any]:
        """Housekeeping pass; returns the report."""
        doc = _invoke(self._lib, self._lib.GomemReflect, self._handle, _enc({"trigger": trigger}))
        return doc

    def stats(self) -> Dict[str, Any]:
        """Metrics snapshot."""
        return _invoke(self._lib, self._lib.GomemStats, self._handle)

    # -- async core verbs ----------------------------------------------------
    # The async twins run the identical (blocking) bridge call in a
    # thread pool so the event loop is not stalled; they mirror the sync
    # methods one-for-one and share their semantics and error behaviour.

    async def _run_in_pool(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Await ``fn(*args, **kwargs)`` executed in the thread pool, off-loop."""
        loop = asyncio.get_running_loop()
        pool = _resolve_executor(self._executor)
        return await loop.run_in_executor(pool, functools.partial(fn, *args, **kwargs))

    async def aremember(
        self,
        *,
        kind: str = "fact",
        title: str,
        body: str = "",
        payload: Any = None,
        tags: Optional[List[str]] = None,
        tier: Optional[int] = None,
        supersedes: str = "",
        item_id: str = "",
        source: str = "run",
        run_id: str = "",
        node_role: str = "",
        confidence: float = 0.8,
        evidence_refs: Optional[List[str]] = None,
        created_at: str = "",
    ) -> str:
        """Async :meth:`remember`; returns the item's ULID without blocking the loop."""
        return await self._run_in_pool(
            self.remember,
            kind=kind, title=title, body=body, payload=payload, tags=tags,
            tier=tier, supersedes=supersedes, item_id=item_id, source=source,
            run_id=run_id, node_role=node_role, confidence=confidence,
            evidence_refs=evidence_refs, created_at=created_at,
        )

    async def aremember_many(self, items: List[Dict[str, Any]]) -> List[str]:
        """Async :meth:`remember_many`; returns the ULIDs without blocking the loop."""
        return await self._run_in_pool(self.remember_many, items)

    async def aget(self, item_id: str) -> Dict[str, Any]:
        """Async :meth:`get`; returns the item document without blocking the loop."""
        return await self._run_in_pool(self.get, item_id)

    async def adelete(self, item_id: str) -> None:
        """Async :meth:`delete`; tombstones the item without blocking the loop."""
        await self._run_in_pool(self.delete, item_id)

    async def atouch(self, item_id: str) -> None:
        """Async :meth:`touch`; records an access without blocking the loop."""
        await self._run_in_pool(self.touch, item_id)

    async def arecall(
        self,
        goal: str,
        *,
        top_k: int = 5,
        sub_goal: str = "",
        kinds: Optional[List[str]] = None,
        min_tier: int = 0,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        budget_tokens: int = 0,
        recency_half_life_days: float = 0.0,
        include_superseded: bool = False,
        dedup_jaccard_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Async :meth:`recall`; returns hits best-first without blocking the loop."""
        return await self._run_in_pool(
            self.recall, goal,
            top_k=top_k, sub_goal=sub_goal, kinds=kinds, min_tier=min_tier,
            tags=tags, min_confidence=min_confidence, budget_tokens=budget_tokens,
            recency_half_life_days=recency_half_life_days,
            include_superseded=include_superseded,
            dedup_jaccard_threshold=dedup_jaccard_threshold,
        )

    async def areflect(self, trigger: str = "run_end") -> Dict[str, Any]:
        """Async :meth:`reflect`; returns the housekeeping report."""
        return await self._run_in_pool(self.reflect, trigger)

    async def astats(self) -> Dict[str, Any]:
        """Async :meth:`stats`; returns the metrics snapshot."""
        return await self._run_in_pool(self.stats)

    async def aclose(self) -> None:
        """Async :meth:`close`; flushes the WAL and releases the handle."""
        await self._run_in_pool(self.close)

    # -- async context manager -------------------------------------------------

    async def __aenter__(self) -> "Memory":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()
