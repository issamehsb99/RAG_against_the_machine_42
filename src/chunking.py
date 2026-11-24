from pathlib import Path
from models import Source
from ast import get_source_segment, parse
import ast


class chunking:
    def __init__(self, max_size: int):
        self.max_size: int = max_size
        self.id: int = 1
        self.dic_chunks: dict = {"python": list(), "text": list()}

    def save_chunks(self, start, end, file, content, mode="python"):
        self.dic_chunks[mode].append(
            Source(
                file_path=file,
                first_character_index=start,
                last_character_index=end,
                content=content,
                chunk_id=self.id,
            )
        )
        self.id += 1

    def find_files(self):
        """
        this function separate files of code python and other docs that should
        i chunk and make it in a dict

        return value is dict of file with two keys "py" and "docs"
        """
        path_files = dict()
        path_files["py"] = []
        path_files["docs"] = []
        i = 1
        repo = Path("data/raw/vllm-0.10.1")
        for file in repo.rglob("*"):
            if str(file).endswith(".py"):
                path_files["py"].append(str(file))
            elif str(file).endswith(".md"):
                path_files["docs"].append(str(file))
            elif str(file).endswith(".txt"):
                path_files["docs"].append(str(file))
        # print(path_files)
        return path_files

    def chunk_docs(self, files: list[str]):
        """
        this function chunk docs files and should return first and last index
        and file path

        """
        overlap = int(self.max_size * 0.15)
        size = self.max_size
        start = 0
        for file in files:
            with open(file, "r") as f:
                content = f.read()
            content_len = len(content)
            start = 0
            if content_len <= size:
                self.save_chunks(
                    start=start,
                    end=content_len,
                    content=content,
                    file=file,
                    mode="text",
                )
                continue
            step = size - overlap
            for s in range(0, max(len(content) - overlap, 1), step):
                k = min(s + size, len(content))
                self.save_chunks(s, k, file, content[s:k], "text")

    @staticmethod
    def _line_starts(code):
        """ls[i] = char offset of line i+1; ls[n] = offset right after line n."""
        ls = [0]
        pos = code.find("\n")
        while pos != -1:
            ls.append(pos + 1)
            pos = code.find("\n", pos + 1)
        if ls[-1] != len(code):  # no trailing newline
            ls.append(len(code))
        return ls

    def _split_by_size(self, code, a, b, file):
        """fixed windows, snapped back to a newline so lines aren't cut."""
        size = self.max_size
        while a < b:
            k = min(a + size, b)
            if k < b:
                nl = code.rfind("\n", a, k)
                if nl > a:
                    k = nl + 1
            self.save_chunks(a, k, file, code[a:k])
            a = k

    def chunking_long_node(self, body, code, ls, prev, stop, file):
        """
        merge consecutive nodes of `body` into chunks <= max_size.
        big node: flush pending, recurse if class, else split by size.
        chunks are contiguous: each starts where the previous node ended,
        so comments / blank lines / decorators / class header are kept.
        """
        size = self.max_size
        start = None
        for node in body:
            end = max(ls[node.end_lineno], prev)
            if end - prev >= size:
                if start is not None:
                    self.save_chunks(start, prev, file, code[start:prev])
                    start = None
                if isinstance(node, ast.ClassDef):
                    self.chunking_long_node(node.body, code, ls, prev, end, file)
                else:
                    self._split_by_size(code, prev, end, file)
            elif start is None:
                start = prev
            elif end - start > size:
                self.save_chunks(start, prev, file, code[start:prev])
                start = prev
            prev = end
        if start is None:
            start = prev
        if stop > start and code[start:stop].strip():
            self.save_chunks(start, stop, file, code[start:stop])

    def chunking_py(self, paths):
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            n = len(code)
            if n <= self.max_size:
                self.save_chunks(0, n, path, code)
                continue
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError):
                # unparsable: fallback to plain windows instead of skipping
                self._split_by_size(code, 0, n, path)
                continue
            ls = self._line_starts(code)
            self.chunking_long_node(tree.body, code, ls, 0, n, path)