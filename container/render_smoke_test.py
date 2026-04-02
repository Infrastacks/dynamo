#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


CONTAINER_DIR = Path(__file__).parent
DOCKER_INSTRUCTIONS = {
    "ARG",
    "CMD",
    "COPY",
    "ENTRYPOINT",
    "ENV",
    "EXPOSE",
    "FROM",
    "HEALTHCHECK",
    "LABEL",
    "ONBUILD",
    "RUN",
    "SHELL",
    "STOPSIGNAL",
    "USER",
    "VOLUME",
    "WORKDIR",
}
RENDER_CASES = [
    ("vllm", "xpu", "runtime"),
    ("vllm", "cuda", "dev"),
    ("vllm", "cuda", "wheel_builder"),
]


def validate_rendered_dockerfile(path: Path) -> None:
    content = path.read_text()
    if "{{" in content or "{%" in content:
        raise AssertionError(f"unresolved template marker in {path}")

    previous_continued = False
    for line_no, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            previous_continued = stripped.endswith("\\") if stripped else False
            continue

        instruction = stripped.split(maxsplit=1)[0].upper()
        if previous_continued and instruction in DOCKER_INSTRUCTIONS:
            raise AssertionError(
                f"Dockerfile instruction {instruction} appears inside a continued instruction at {path}:{line_no}"
            )

        previous_continued = stripped.endswith("\\")


def render_and_validate(framework: str, device: str, target: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "render.py",
            "--framework",
            framework,
            "--device",
            device,
            "--target",
            target,
            "--output-short-filename",
        ],
        cwd=CONTAINER_DIR,
        check=True,
    )
    validate_rendered_dockerfile(CONTAINER_DIR / "rendered.Dockerfile")


def main() -> None:
    for framework, device, target in RENDER_CASES:
        render_and_validate(framework, device, target)
    print("container render smoke test passed")


if __name__ == "__main__":
    main()
