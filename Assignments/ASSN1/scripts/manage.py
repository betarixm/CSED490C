import csv
import glob
import io
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

from jinja2 import Environment, FileSystemLoader

PROJECT_DIR = Path(__file__).parent.parent


Data = Tuple[str, Path, Path, Path]
Dataset = List[Data]


@dataclass
class Log:
    kind: str
    description: str
    at: str
    elapsed_time_in_ms: float


@dataclass
class Result:
    key: str
    is_correct: bool
    logs: List[Log]


@dataclass
class GpuInfo:
    name: str
    memory_in_mib: int


class Executable:
    def __init__(self, executable_path: Path) -> None:
        self._executable_path = executable_path

    def __call__(
        self,
        args: List[str | Path] = [],
        capture_output: bool = False,
        cwd: Path | None = None,
    ):
        if cwd is None:
            cwd = self._executable_path.parent

        return subprocess.run(
            [self._executable_path, *args], cwd=cwd, capture_output=capture_output
        )

    def _use_default_cwd(self, cwd: Path | None = None) -> Path:
        if cwd is None:
            return self._executable_path.parent

        return cwd


class ExecutableWithColonResults(Executable):
    def query(self) -> List[Tuple[str, str]]:
        result = self(capture_output=True).stdout.decode().strip()

        return [
            (entries[0].strip(), entries[1].strip())
            for entries in [line.split(":") for line in result.split("\n")]
        ]


class DatasetGenerator(Executable):
    def generate(self, cwd: Path | None = None) -> Dataset:
        cwd = self._use_default_cwd(cwd)

        self(cwd=cwd)

        return [
            (
                path.split("/")[-1],
                Path(path) / "input0.raw",
                Path(path) / "input1.raw",
                Path(path) / "output.raw",
            )
            for path in glob.glob(f"{cwd}/VectorAdd/Dataset/*")
        ]


class Target(Executable):
    def run(self, data: Data, cwd: Path | None = None):
        cwd = self._use_default_cwd(cwd)

        _, input0, input1, output = data

        result = self(
            ["-e", output, "-i", f"{input0},{input1}", "-t", "vector"],
            cwd=cwd,
            capture_output=True,
        )

        return result.stdout.decode()


class NvidiaSmi(Executable):
    def __init__(self, executable_path: Path = Path("/usr/bin/nvidia-smi")) -> None:
        super().__init__(executable_path)

    def query(
        self, params: List[str], format_: Literal["csv"] = "csv"
    ) -> List[Tuple[str, str]]:
        result = (
            self(
                ["--query-gpu", ",".join(params), "--format", format_],
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )

        for row in csv.DictReader(io.StringIO(result), delimiter=","):
            return [(str(k).strip(), str(v).strip()) for k, v in row.items()]

        raise ValueError()


class LsCpu(ExecutableWithColonResults):
    def __init__(self, executable_path: Path = Path("/usr/bin/lscpu")) -> None:
        super().__init__(executable_path)


class HostnameCtl(ExecutableWithColonResults):
    def __init__(self, executable_path: Path = Path("/usr/bin/hostnamectl")) -> None:
        super().__init__(executable_path)


class Project:
    def __init__(
        self,
        project_dir: Path,
        source_dir: str = "sources",
        build_dir: str = "build",
        c_compiler_path: Path = Path("/usr/bin/gcc"),
        cxx_compiler_path: Path = Path("/usr/bin/g++"),
    ) -> None:
        self._project_dir: Path = project_dir
        self._source_dir: Path = project_dir / source_dir
        self._build_dir: Path = project_dir / build_dir
        self._c_compiler_path: Path = c_compiler_path
        self._cxx_compiler_path: Path = cxx_compiler_path

    def build_cmake_project(self):
        subprocess.run(
            [
                "cmake",
                "--no-warn-unused-cli",
                "-D",
                "CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE",
                "-D",
                f"CMAKE_C_COMPILER:FILEPATH={self._c_compiler_path}",
                "-D",
                f"CMAKE_CXX_COMPILER:FILEPATH={self._cxx_compiler_path}",
                "-S",
                self._project_dir,
                "-B",
                self._build_dir,
            ]
        )
        subprocess.run(["make"], cwd=self._build_dir)

    def make_dataset_generator(
        self, suppress_compile: bool = False
    ) -> DatasetGenerator:
        if not suppress_compile:
            subprocess.run(["make", "dataset_generator"], cwd=self._source_dir)

        return DatasetGenerator(self._source_dir / "dataset_generator")

    def make_target(self, suppress_compile: bool = False) -> Target:
        if not suppress_compile:
            subprocess.run(["make", "template"], cwd=self._source_dir)

        return Target(self._source_dir / "VectorAdd_template")

    def make_clean(self):
        subprocess.run(["make", "clean"], cwd=self._source_dir)


def _parse_line_of_elapsed_time(line: str):
    matched = re.search(
        r"\[TIME\]\[(.*)\]\[(.*)\]\[(.*)\] Elapsed time: (.*) ms",
        line,
    )

    if matched is None:
        return None

    kind, description, at, elapsed_time_in_ms = matched.groups()

    return Log(
        kind=kind,
        description=description,
        at=at,
        elapsed_time_in_ms=float(elapsed_time_in_ms),
    )


def _parse_result(key: str, out: str):
    lines = out.strip().split("\n")

    return Result(
        key=key,
        is_correct=lines[-1].endswith("is correct"),
        logs=[
            parsed
            for parsed in [
                _parse_line_of_elapsed_time(line)
                for line in lines
                if line.startswith("[TIME]")
            ]
            if parsed is not None
        ],
    )


def _evaluate(project: Project, cwd: Path):
    ls_cpu, nvidia_smi, hostname_ctl = (
        LsCpu(),
        NvidiaSmi(),
        HostnameCtl(),
    )

    dataset_generator = project.make_dataset_generator(suppress_compile=True)
    target = project.make_target(suppress_compile=True)

    dataset = dataset_generator.generate(cwd)

    results = [_parse_result(data[0], target.run(data, cwd)) for data in dataset]
    results = sorted(results, key=lambda x: x.key)

    jinja2_env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates"))
    )
    template = jinja2_env.get_template("evaluation.md")

    return template.render(
        cpu_info=ls_cpu.query(),
        gpu_info=nvidia_smi.query(["name", "memory.total"], "csv"),
        host_info=hostname_ctl.query(),
        fields=[
            "Key",
            "Correctness",
            *[log.description for log in results[0].logs],
        ],
        results=[
            (
                result.key,
                result.is_correct,
                *(str(log.elapsed_time_in_ms) + " ms" for log in result.logs),
            )
            for result in results
        ],
    )


def _compile_project(project: Project):
    project.make_dataset_generator()
    project.make_target()


if __name__ == "__main__":
    import sys

    project = Project(PROJECT_DIR)
    option = sys.argv[1]

    if option == "compile":
        _compile_project(project)
    elif option == "evaluate":
        tempdir = tempfile.TemporaryDirectory()
        cwd = Path(tempdir.name)

        evaluation = _evaluate(project, cwd)

        open(str(PROJECT_DIR / "docs" / "evaluation.md"), "w").write(evaluation)

        print(evaluation)
