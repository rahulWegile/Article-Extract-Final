import shutil
import uuid

from backend.core.settings import TEMP_DIR


class WorkspaceManager:
    """
    Temporary per-run scratch workspace.

    Every run gets its OWN uniquely named directory underneath
    TEMP_DIR.

    A single shared directory used to be reused for every run, and
    because create() wipes that directory up front, two uploads
    running at the same time deleted each other's rendered pages
    mid-pipeline -- surfacing as a FileNotFoundError on a
    page_NNN.png that had definitely been rendered moments earlier.
    cleanup() has the same hazard, so it now removes only the
    directory belonging to this instance.

    Pass an explicit `workspace` to adopt an existing directory
    (mainly useful in tests); otherwise a fresh unique one is used.
    """

    def __init__(self, workspace=None):

        if workspace is not None:
            self.workspace = workspace
        else:
            self.workspace = (
                TEMP_DIR / f"run_{uuid.uuid4().hex[:12]}"
            )

    def create(self):

        if self.workspace.exists():
            shutil.rmtree(self.workspace)

        self.workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

        (self.workspace / "pages").mkdir()
        (self.workspace / "page_json").mkdir()
        (self.workspace / "gemini").mkdir()
        (self.workspace / "final").mkdir()

        return self.workspace

    def cleanup(self):

        if self.workspace.exists():
            shutil.rmtree(self.workspace)
