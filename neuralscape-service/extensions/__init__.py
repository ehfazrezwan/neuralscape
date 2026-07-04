"""Extension registry for NeuralScape — discovery, lifecycle, and event dispatch.

The ExtensionRegistry discovers, loads, and manages NeuralScape extensions.
Extensions are discovered from:
  1. Subdirectories of the extensions/ package (auto-discovery)
  2. Python import paths listed in the NEURALSCAPE_EXTENSIONS env var

Usage in main.py:
    registry = ExtensionRegistry()
    await registry.discover()
    await registry.startup_all()
    registry.mount_routes(app)
    ...
    await registry.shutdown_all()
"""

import importlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI

from extensions.base import ExtensionManifest, NeuralscapeExtension

logger = logging.getLogger(__name__)


@dataclass
class ExtensionEntry:
    """Internal record for a registered extension."""

    instance: NeuralscapeExtension
    manifest: ExtensionManifest
    status: str = "registered"  # registered | started | failed | stopped


@dataclass
class EmitResult:
    """Result of broadcasting an event to extensions."""

    responses: list[dict] = field(default_factory=list)
    notified_count: int = 0


class ExtensionRegistry:
    """Manages discovery, lifecycle, and event dispatch for NeuralScape extensions.

    Extension failures are logged and do not propagate — a single broken
    extension cannot take down the service.
    """

    def __init__(self) -> None:
        self._extensions: dict[str, ExtensionEntry] = {}

    @property
    def extensions(self) -> dict[str, ExtensionEntry]:
        """Read-only access to registered extensions."""
        return dict(self._extensions)

    async def discover(self) -> None:
        """Discover and register extensions from subdirectories and env var.

        Scans extensions/ subdirectories for modules exporting a class that
        implements NeuralscapeExtension, then loads any additional extensions
        specified in the NEURALSCAPE_EXTENSIONS environment variable.
        """
        self._discover_local()
        self._discover_env()
        logger.info(
            "Extension discovery complete",
            extra={"extensions": list(self._extensions.keys())},
        )

    def _discover_local(self) -> None:
        """Discover extensions from subdirectories of the extensions/ package."""
        extensions_dir = Path(__file__).parent
        for item in sorted(extensions_dir.iterdir()):
            if not item.is_dir() or item.name.startswith(("_", ".")):
                continue
            init_file = item / "__init__.py"
            if not init_file.exists():
                continue
            module_name = f"extensions.{item.name}"
            self._load_extension_module(module_name, source=f"local:{item.name}")

    def _discover_env(self) -> None:
        """Discover extensions from NEURALSCAPE_EXTENSIONS env var."""
        env_extensions = os.environ.get("NEURALSCAPE_EXTENSIONS", "").strip()
        if not env_extensions:
            return
        for import_path in env_extensions.split(","):
            import_path = import_path.strip()
            if not import_path:
                continue
            self._load_extension_module(import_path, source=f"env:{import_path}")

    def _load_extension_module(self, module_name: str, source: str) -> None:
        """Import a module and register the first NeuralscapeExtension class found."""
        try:
            module = importlib.import_module(module_name)
        except Exception:
            logger.exception(
                "Failed to import extension module %s (source=%s)",
                module_name,
                source,
            )
            return

        # Look for a class-level attribute or a factory
        ext_instance = None

        # Check for a module-level 'extension' attribute (pre-instantiated)
        if hasattr(module, "extension") and isinstance(module.extension, NeuralscapeExtension):
            ext_instance = module.extension

        # Otherwise, find the first class implementing the protocol
        if ext_instance is None:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and attr is not NeuralscapeExtension
                    and issubclass_safe(attr)
                ):
                    try:
                        ext_instance = attr()
                    except Exception:
                        logger.exception(
                            "Failed to instantiate extension class %s from %s",
                            attr_name,
                            module_name,
                        )
                        return
                    break

        if ext_instance is None:
            logger.warning(
                "No NeuralscapeExtension found in module %s (source=%s)",
                module_name,
                source,
            )
            return

        self.register(ext_instance)

    def register(self, extension: NeuralscapeExtension) -> None:
        """Register a single extension instance.

        Args:
            extension: An object implementing the NeuralscapeExtension protocol.

        Note:
            If an extension with the same name is already registered, logs a warning and skips the duplicate.
        """
        name = extension.manifest.name
        if name in self._extensions:
            logger.warning(
                "Extension already registered, skipping duplicate",
                extra={"extension": name},
            )
            return
        self._extensions[name] = ExtensionEntry(
            instance=extension,
            manifest=extension.manifest,
        )
        logger.info(
            "Extension registered",
            extra={
                "extension": name,
                "version": extension.manifest.version,
                "hooks": extension.manifest.hooks,
            },
        )

    async def startup_all(self) -> None:
        """Call startup() on all registered extensions.

        Failures are logged; other extensions continue starting.
        """
        for name, entry in self._extensions.items():
            try:
                await entry.instance.startup()
                entry.status = "started"
                logger.info("Extension started", extra={"extension": name})
            except Exception:
                entry.status = "failed"
                logger.exception(
                    "Extension startup failed",
                    extra={"extension": name},
                )

    async def shutdown_all(self) -> None:
        """Call shutdown() on all started extensions.

        Failures are logged; other extensions continue shutting down.
        """
        for name, entry in self._extensions.items():
            if entry.status not in ("started",):
                continue
            try:
                await entry.instance.shutdown()
                entry.status = "stopped"
                logger.info("Extension stopped", extra={"extension": name})
            except Exception:
                logger.exception(
                    "Extension shutdown failed",
                    extra={"extension": name},
                )

    def mount_routes(self, app: FastAPI) -> None:
        """Mount all extension routes onto the FastAPI app.

        Each extension's routes are mounted at /v1/extensions/<name>/.
        """
        for name, entry in self._extensions.items():
            try:
                router = entry.instance.get_routes()
                if router is not None:
                    app.include_router(
                        router,
                        prefix=f"/v1/extensions/{name}",
                        tags=[f"ext:{name}"],
                    )
                    logger.info(
                        "Extension routes mounted",
                        extra={"extension": name, "prefix": f"/v1/extensions/{name}"},
                    )
            except Exception:
                logger.exception(
                    "Failed to mount extension routes",
                    extra={"extension": name},
                )

    async def emit_event(self, event_type: str, payload: dict) -> EmitResult:
        """Broadcast an event to all extensions whose manifest.hooks includes the event type.

        Args:
            event_type: The event type string (e.g. 'memory_stored').
            payload: Event payload as a dictionary.

        Returns:
            EmitResult with non-None responses and the count of extensions notified.
        """
        # E1: mirror memory events onto the live SSE stream. Every worker
        # write path funnels its memory_stored emission through this method,
        # so one fire-and-forget hook covers them all. Channel routing
        # enforces visibility at publish time (see event_stream.channel_for).
        # Audit 27 #11: dispatched via the shared telemetry executor — the
        # per-fact fan-out used to do one sync Redis publish per memory on
        # the worker loop (a batch of 25 facts = 25 blocking round trips).
        if event_type == "memory_stored":
            try:
                from event_stream import publish_event_bg

                publish_event_bg(event_type, payload)
            except Exception:
                logger.debug("event-stream mirror failed (non-fatal)", exc_info=True)

        responses: list[dict] = []
        notified_count = 0
        for name, entry in self._extensions.items():
            if entry.status != "started":
                continue
            if event_type not in entry.manifest.hooks:
                continue
            notified_count += 1
            try:
                result = await entry.instance.on_event(event_type, payload)
                if result is not None:
                    responses.append(result)
            except Exception:
                logger.exception(
                    "Extension event handler failed",
                    extra={"extension": name, "event_type": event_type},
                )
        return EmitResult(responses=responses, notified_count=notified_count)

    def list_extensions(self) -> list[dict]:
        """Return a summary of all registered extensions.

        Returns:
            List of dicts with name, version, description, status, and hooks.
        """
        return [
            {
                "name": entry.manifest.name,
                "version": entry.manifest.version,
                "description": entry.manifest.description,
                "author": entry.manifest.author,
                "status": entry.status,
                "hooks": entry.manifest.hooks,
            }
            for entry in self._extensions.values()
        ]


def issubclass_safe(cls: type) -> bool:
    """Check if cls has the NeuralscapeExtension protocol attributes.

    Uses duck-typing check rather than issubclass() to avoid issues
    with Protocol runtime checking on classes that don't fully implement it.
    """
    required = ("manifest", "startup", "shutdown", "on_event", "get_routes")
    return all(hasattr(cls, attr) for attr in required)
