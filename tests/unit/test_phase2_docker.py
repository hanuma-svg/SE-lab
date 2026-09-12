from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

IMAGE_TAG = "se-lab-worker:test"
SEC_COMP_PROFILE = str(pathlib.Path("docker/seccomp.json").resolve())


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _docker_available() -> bool:
    return _run(["docker", "--version"]).returncode == 0


def _build_image() -> None:
    build = _run(["docker", "build", "-t", IMAGE_TAG, "-f", "docker/worker.Dockerfile", "."])
    assert build.returncode == 0, build.stderr


def _remove_image() -> None:
    _run(["docker", "rmi", IMAGE_TAG])


def test_docker_is_available() -> None:
    assert _docker_available()


def test_docker_build_smoke() -> None:
    if not _docker_available():
        pytest.skip("docker is unavailable")

    try:
        _build_image()
    finally:
        _remove_image()


def test_docker_runtime_smoke(tmp_path: pathlib.Path) -> None:
    if not _docker_available():
        pytest.skip("docker is unavailable")

    workspace = tmp_path / "workspace"
    repo = workspace / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "README.md").write_text("runtime smoke", encoding="utf-8")
    workspace.chmod(0o777)
    repo.chmod(0o777)
    (repo / "README.md").chmod(0o666)

    container_name = "se-lab-worker-smoke"

    try:
        _build_image()

        docker_run = _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--network",
                "none",
                "--cpus",
                "0.5",
                "--memory",
                "128m",
                "-v",
                f"{workspace}:/workspace:rw",
                IMAGE_TAG,
                "sleep",
                "10",
            ]
        )
        assert docker_run.returncode == 0, docker_run.stderr
        container_id = docker_run.stdout.strip()
        assert container_id

        inspect_result = _run(["docker", "inspect", container_id])
        assert inspect_result.returncode == 0, inspect_result.stderr

        details = json.loads(inspect_result.stdout)[0]
        assert details["HostConfig"]["NetworkMode"] == "none"
        assert details["HostConfig"]["Memory"] == 128 * 1024 * 1024
        assert details["HostConfig"]["NanoCpus"] == 500_000_000
        assert details["Config"]["User"] == "app"
        assert details["Config"]["WorkingDir"] == "/workspace"

        exec_script = (
            "from pathlib import Path; "
            "import os, pwd; "
            "assert Path('/workspace/repo/README.md').read_text(encoding='utf-8') == 'runtime smoke'; "
            "assert Path('/workspace').exists(); "
            "assert pwd.getpwuid(os.geteuid()).pw_name == 'app'; "
            "assert os.environ.get('HOME') == '/home/app'; "
            "assert not Path('/var/run/docker.sock').exists(); "
            "print('runtime-clean')"
        )

        exec_result = _run(["docker", "exec", container_id, "python", "-c", exec_script])
        assert exec_result.returncode == 0, exec_result.stderr
        assert exec_result.stdout.strip() == "runtime-clean"
    finally:
        _run(["docker", "rm", "-f", container_name])
        _remove_image()


def test_docker_seccomp_profile_is_host_limited() -> None:
    if not _docker_available():
        pytest.skip("docker is unavailable")

    profile_path = pathlib.Path(SEC_COMP_PROFILE)
    assert profile_path.exists()

    try:
        _build_image()

        seccomp_run = _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--security-opt",
                f"seccomp={SEC_COMP_PROFILE}",
                IMAGE_TAG,
                "python",
                "-c",
                "print('should-not-start')",
            ]
        )
    finally:
        _remove_image()

    assert seccomp_run.returncode != 0
    assert (
        "operation not permitted" in seccomp_run.stderr.lower()
        or "unable to apply bounding set" in seccomp_run.stderr.lower()
        or "seccomp" in seccomp_run.stderr.lower()
    )
