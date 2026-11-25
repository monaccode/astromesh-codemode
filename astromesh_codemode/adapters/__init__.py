"""Framework adapters: each module exposes ``as_tools(cm) -> list[<native tool>]``.

Import the module of the framework you use (``astromesh_codemode.adapters.langchain``...);
each one needs its extra (``pip install 'astromesh-codemode[langchain]'``).
"""

from astromesh_codemode.adapters._common import MissingExtraError, run_sync

__all__ = ["MissingExtraError", "run_sync"]
