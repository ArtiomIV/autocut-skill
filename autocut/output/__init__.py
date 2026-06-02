"""Output writers — turn a ranked clip plan into files on disk."""

from autocut.output.base import OutputWriter, WrittenClip
from autocut.output.dispatcher import (
    PLAN_FILENAME,
    DispatchResult,
    PlanReadError,
    dispatch_outputs,
    read_plan_json,
    write_plan_json,
)
from autocut.output.merged import MergedWriter
from autocut.output.separate import SeparateWriter

__all__ = [
    "PLAN_FILENAME",
    "DispatchResult",
    "MergedWriter",
    "OutputWriter",
    "PlanReadError",
    "SeparateWriter",
    "WrittenClip",
    "dispatch_outputs",
    "read_plan_json",
    "write_plan_json",
]
