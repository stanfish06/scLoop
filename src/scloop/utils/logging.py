# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Literal

from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

LogLevel = Literal["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"]


class LogEntry(BaseModel):
    timestamp: float = Field(ge=0)
    level: LogLevel
    message: str
    icon: str = ""


class LogCache(BaseModel):
    maxlen: int = Field(default=20, ge=1)
    messages: deque[LogEntry] = Field(default_factory=lambda: deque(maxlen=20))
    total_counts: dict[str, int] = Field(default_factory=dict)
    _handler_id: int | None = None
    _update_callback: Callable[[], None] | None = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        del __context
        if len(self.messages) == 0:
            self.messages = deque(maxlen=self.maxlen)
        elif self.messages.maxlen != self.maxlen:
            new_deque = deque(self.messages, maxlen=self.maxlen)
            self.messages = new_deque

    def append(self, message: str) -> None:
        try:
            record = json.loads(message)
            level = record["record"]["level"]["name"]
            icon = record["record"]["level"].get("icon", "")
            msg = record["record"]["message"]
        except (json.JSONDecodeError, KeyError):
            level = "INFO"
            icon = "ℹ️"
            msg = message.strip()

        entry = LogEntry(
            timestamp=time.time(), level=level, message=msg, icon=icon.strip()
        )
        self.messages.append(entry)

        # Track total counts
        self.total_counts[level] = self.total_counts.get(level, 0) + 1

        if self._update_callback:
            self._update_callback()

    def get_messages(self) -> list[LogEntry]:
        return list(self.messages)

    def get_total_counts(self) -> dict[str, int]:
        return self.total_counts.copy()

    def clear(self) -> None:
        self.messages.clear()
        self.total_counts.clear()


class LogDisplay(BaseModel):
    maxlen: int = Field(default=20, ge=1, le=1000)
    refresh_per_second: int = Field(default=10, ge=1, le=60)
    cache: LogCache = Field(default_factory=lambda: LogCache())
    console: Console = Field(default_factory=Console)
    progress: Progress | None = None
    _live: Live | None = None
    _handler_id: int | None = None
    _layout: Layout | None = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        del __context
        self.cache = LogCache(maxlen=self.maxlen)

    def _create_table(self) -> Panel:
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Time", style="dim", width=8)
        table.add_column("Level", width=7)
        table.add_column("Message")

        messages = self.cache.get_messages()
        for entry in messages:
            dt = datetime.fromtimestamp(entry.timestamp)
            time_str = dt.strftime("%H:%M:%S")

            level_styles = {
                "DEBUG": "dim",
                "INFO": "blue",
                "SUCCESS": "green",
                "WARNING": "yellow",
                "ERROR": "red",
            }
            level_style = level_styles.get(entry.level, "white")

            level_display = f"[{level_style}]{entry.level}[/{level_style}]"

            table.add_row(time_str, level_display, entry.message)

        return Panel(table, title="[bold blue]Recent Logs", border_style="blue")

    def _create_layout(self) -> Layout:
        if self.progress is not None:
            layout = Layout()
            log_height = self.maxlen + 5
            layout.split_column(
                Layout(name="progress", size=4), Layout(name="logs", size=log_height)
            )
            layout["progress"].update(
                Panel(self.progress, title="[bold green]Progress", border_style="green")
            )
            layout["logs"].update(self._create_table())
            return layout
        else:
            return Layout(self._create_table())

    def __enter__(self) -> LogCache:
        logger.remove()
        self._handler_id = logger.add(
            self.cache.append,
            serialize=True,
        )

        self.cache._update_callback = self.update

        self._layout = self._create_layout()

        try:
            from IPython.core.getipython import get_ipython

            ip = get_ipython()
            config = getattr(ip, "config", None) if ip is not None else None
            in_jupyter = (
                ip is not None and config is not None and "IPKernelApp" in config
            )
        except (ImportError, AttributeError):
            in_jupyter = False

        self._live = Live(
            self._layout,
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            transient=in_jupyter,
        )
        self._live.__enter__()
        self._in_jupyter = in_jupyter

        return self.cache

    def __exit__(self, *args: Any) -> None:
        if self._live:
            self._live.__exit__(*args)

        if (
            hasattr(self, "_in_jupyter")
            and self._in_jupyter
            and len(self.cache.total_counts) > 0
        ):
            from rich.table import Table

            summary_table = Table(
                show_header=True, header_style="bold magenta", title="Log Summary"
            )
            summary_table.add_column("Level", width=7)
            summary_table.add_column("Count")

            level_counts = self.cache.get_total_counts()

            level_styles = {
                "DEBUG": "dim",
                "INFO": "blue",
                "SUCCESS": "green",
                "WARNING": "yellow",
                "ERROR": "red",
            }

            for level, count in sorted(level_counts.items()):
                level_style = level_styles.get(level, "white")
                summary_table.add_row(
                    f"[{level_style}]{level}[/{level_style}]", str(count)
                )

            self.console.print(summary_table)

        if self._handler_id is not None:
            logger.remove(self._handler_id)

    def update(self) -> None:
        if self._live and self._layout:
            if self.progress is not None:
                self._layout["logs"].update(self._create_table())
            else:
                self._live.update(self._create_table())
