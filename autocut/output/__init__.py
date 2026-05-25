"""Output writers — turn a ranked clip plan into files on disk."""

from autocut.output.base import OutputWriter, WrittenClip
from autocut.output.dispatcher import DispatchResult, dispatch_outputs
from autocut.output.merged import MergedWriter
from autocut.output.separate import SeparateWriter

__all__ = [
    "DispatchResult",
    "MergedWriter",
    "OutputWriter",
    "SeparateWriter",
    "WrittenClip",
    "dispatch_outputs",
]
