import csv
import glob
import io
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple
import os
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
        self.executable_path = executable_path

    def __call__(
        self,
        args: List[str | Path] = [],
        capture_output: bool = False,
        cwd: Path | None = None,
    ):
        if cwd is None:
            cwd = self.executable_path.parent

        return subprocess.run(
            [self.executable_path, *args], cwd=cwd, capture_output=capture_output
        )

    def _use_default_cwd(self, cwd: Path | None = None) -> Path:
        if cwd is None:
            return self.executable_path.parent

        return cwd


class ExecutableWithColonResults(Executable):
    def query(self) -> List[Tuple[str, str]]:
        result = self(capture_output=True).stdout.decode().strip()

        return [
            (entries[0].strip(), entries[1].strip())
            for entries in [line.split(":") for line in result.split("\n")]
        ]


class DatasetGenerator(Executable):
    def generated_dataset(self, cwd: Path | None = None) -> Dataset:
        cwd = self._use_default_cwd(cwd)
        results = [
            (
                path.split("/")[-1],
                Path(path) / "input0.raw",
                Path(path) / "input1.raw",
                Path(path) / "output.raw",
            )
            for path in glob.glob(f"{cwd}/TiledMatrixMultiplication/Dataset/*")
        ]

        return sorted(results, key=lambda x: x[0])

    def generate_dataset(self, cwd: Path | None = None) -> Dataset:
        cwd = self._use_default_cwd(cwd)

        self(cwd=cwd)

        return self.generated_dataset(cwd)


class Target(Executable):
    def run(self, data: Data, cwd: Path | None = None):
        cwd = self._use_default_cwd(cwd)

        _, input0, input1, output = data

        result = self(
            ["-e", output, "-i", f"{input0},{input1}", "-t", "matrix"],
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

    def make_target(
        self, tile_width: int = 16, suppress_compile: bool = False
    ) -> Target:
        target_path = self.get_target_path_by_tile_width(tile_width)

        if not suppress_compile:
            template = open(self._source_dir / "template.cu", "r").read()

            source = template.replace(
                "#define TILE_WIDTH 16", f"#define TILE_WIDTH {tile_width}"
            )

            try:
                writer = open(self._source_dir / "template.cu", "w")
                writer.write(source)
                writer.flush()

                subprocess.run(["make", "template"], cwd=self._source_dir)

                os.rename(
                    self._source_dir / "TiledGEMM_template",
                    target_path,
                )
            finally:
                writer = open(self._source_dir / "template.cu", "w")
                writer.write(template)
                writer.flush()

        return Target(target_path)

    def make_clean(self):
        subprocess.run(["make", "clean"], cwd=self._source_dir)

    def get_target_pathes(self):
        return [
            Path(path)
            for path in glob.glob(str(self._source_dir / "TiledGEMM_template*"))
        ]

    def get_targets(self):
        return [Target(target_path) for target_path in self.get_target_pathes()]

    def get_target_path_by_tile_width(self, tile_width: int):
        return Path(self._source_dir / f"TiledGEMM_template_{tile_width}")

    def get_target_by_tile_width(self, tile_width: int):
        return Target(self.get_target_path_by_tile_width(tile_width))


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


def _run_target_with_dataset(target: Target, dataset: Dataset, cwd: Path | None = None):
    results = [_parse_result(data[0], target.run(data, cwd)) for data in dataset]

    return sorted(results, key=lambda x: x.key)


def _run_targets_with_data(targets: List[Target], data: Data, cwd: Path | None = None):
    results = [
        _parse_result(target.executable_path.stem, target.run(data, cwd))
        for target in targets
    ]

    return sorted(results, key=lambda x: x.key)


def _evaluate(project: Project, cwd: Path | None = None):
    ls_cpu, nvidia_smi, hostname_ctl = (
        LsCpu(),
        NvidiaSmi(),
        HostnameCtl(),
    )

    dataset_generator = project.make_dataset_generator(suppress_compile=True)

    dataset = dataset_generator.generated_dataset(cwd)

    results_by_data = _run_target_with_dataset(
        project.get_target_by_tile_width(16), dataset
    )

    results_by_tile_width = _run_targets_with_data(project.get_targets(), dataset[-1])

    results_by_tile_width = [
        Result(
            key=re.findall(r"(\d+)", result.key)[0],
            is_correct=result.is_correct,
            logs=result.logs,
        )
        for result in results_by_tile_width
    ]

    results_by_tile_width = sorted(results_by_tile_width, key=lambda x: int(x.key))

    jinja2_env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates"))
    )

    template = jinja2_env.get_template("evaluation.md")

    return template.render(
        cpu_info=ls_cpu.query(),
        gpu_info=nvidia_smi.query(["name", "memory.total"], "csv"),
        host_info=hostname_ctl.query(),
        results_by_data_fields=[
            "Data",
            "Correctness",
            *[log.description for log in results_by_data[0].logs],
        ],
        results_by_data_values=[
            (
                result.key,
                result.is_correct,
                *(str(log.elapsed_time_in_ms) + " ms" for log in result.logs),
            )
            for result in results_by_data
        ],
        results_by_tile_width_fields=[
            "Tile Width",
            "Correctness",
            *[log.description for log in results_by_tile_width[0].logs],
        ],
        results_by_tile_width_values=[
            (
                result.key,
                result.is_correct,
                *(str(log.elapsed_time_in_ms) + " ms" for log in result.logs),
            )
            for result in results_by_tile_width
        ],
    )


def _compile_project(project: Project):
    project.make_dataset_generator()

    for tile_width in [2, 4, 8, 12, 16, 24, 32]:
        project.make_target(tile_width=tile_width)


if __name__ == "__main__":
    import sys

    project = Project(PROJECT_DIR)

    command = sys.argv[1]

    if command == "compile":
        _compile_project(project)
    elif command == "generate-dataset":
        project.make_dataset_generator(suppress_compile=True).generate_dataset()
    elif command == "evaluate":
        evaluation = _evaluate(project)

        open(PROJECT_DIR / "docs" / "evaluation.md", "w").write(evaluation)

        print(evaluation)
