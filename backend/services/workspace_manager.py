import shutil

from backend.core.settings import TEMP_DIR


class WorkspaceManager:

    def __init__(self):

        self.workspace = TEMP_DIR

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