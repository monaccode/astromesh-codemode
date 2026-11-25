from astromesh_codemode import CodeMode
from tests.fixtures import make_tools

CODE = "len(await get_rows(n=3))"
NAMES = ["codemode_search", "codemode_run"]


def make_cm() -> CodeMode:
    specs, _ = make_tools()
    return CodeMode(specs)
