"""Tests for container-based vLLM lifecycle management."""

from unittest.mock import Mock, patch

from scicode_lint.vllm.container import (
    _CONTAINER_NAME,
    _container_exists,
    _container_running,
    _detect_container_runtime,
    _find_container_on_port,
    _identify_port_holder,
    _port_available,
    container_logs,
    container_monitor,
    container_status,
    remove_container,
    start_container,
    stop_container,
)


class TestDetectContainerRuntime:
    """Tests for _detect_container_runtime."""

    def test_prefers_podman(self) -> None:
        """Should prefer podman when both are available."""
        with patch("shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            assert _detect_container_runtime() == "podman"

    def test_fallback_to_docker(self) -> None:
        """Should fall back to docker when podman is not available."""

        def which(name: str) -> str | None:
            return "/usr/bin/docker" if name == "docker" else None

        with patch("shutil.which", side_effect=which):
            assert _detect_container_runtime() == "docker"

    def test_none_when_nothing_available(self) -> None:
        """Should return None when neither podman nor docker is found."""
        with patch("shutil.which", return_value=None):
            assert _detect_container_runtime() is None


class TestContainerHelpers:
    """Tests for container helper functions."""

    def test_container_exists_true(self) -> None:
        """Should return True when container exists."""
        mock_result = Mock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert _container_exists("podman") is True

    def test_container_exists_false(self) -> None:
        """Should return False when container does not exist."""
        mock_result = Mock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            assert _container_exists("podman") is False

    def test_container_running_true(self) -> None:
        """Should return True when container is running."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "true"
        with patch("subprocess.run", return_value=mock_result):
            assert _container_running("podman") is True

    def test_container_running_false(self) -> None:
        """Should return False when container is not running."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "false"
        with patch("subprocess.run", return_value=mock_result):
            assert _container_running("podman") is False

    def test_port_available_free(self) -> None:
        """Should return True for a free port."""
        # Use a high-numbered port unlikely to be in use
        assert _port_available("127.0.0.1", 59999) is True

    def test_identify_port_holder_found(self) -> None:
        """Should identify container holding a port."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "scicode-lint-vllm\t0.0.0.0:5001->8000/tcp"
        with patch("subprocess.run", return_value=mock_result):
            result = _identify_port_holder("podman", 5001)
            assert "scicode-lint-vllm" in result

    def test_identify_port_holder_not_found(self) -> None:
        """Should return empty string when no container holds the port."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert _identify_port_holder("podman", 5001) == ""

    def test_identify_port_holder_no_runtime(self) -> None:
        """Should return empty string when no runtime provided."""
        assert _identify_port_holder(None, 5001) == ""

    def test_find_container_on_port_returns_name(self) -> None:
        """Should return the container name holding the port."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "other-vllm\t0.0.0.0:5001->8000/tcp"
        with patch("subprocess.run", return_value=mock_result):
            assert _find_container_on_port("podman", 5001) == "other-vllm"

    def test_find_container_on_port_not_found(self) -> None:
        """Should return None when no container holds the port."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            assert _find_container_on_port("podman", 5001) is None

    def test_find_container_on_port_no_runtime(self) -> None:
        """Should return None when no runtime provided."""
        assert _find_container_on_port(None, 5001) is None


class TestStartContainer:
    """Tests for start_container."""

    def test_no_runtime_returns_error(self) -> None:
        """Should return 1 when no container runtime found."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value=None,
        ):
            assert start_container() == 1

    def test_already_running_returns_0(self) -> None:
        """Should return 0 if container is already running."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_running", return_value=True):
                assert start_container() == 0

    def test_restart_stopped_container(self) -> None:
        """Should restart a stopped container (fast path)."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_running", return_value=False):
                with patch("scicode_lint.vllm.container._container_exists", return_value=True):
                    mock_result = Mock()
                    mock_result.returncode = 0
                    with patch("subprocess.run", return_value=mock_result):
                        assert start_container() == 0

    def test_start_success(self) -> None:
        """Should start container successfully."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_running", return_value=False):
                with patch(
                    "scicode_lint.vllm.container._container_exists",
                    return_value=False,
                ):
                    with patch(
                        "scicode_lint.vllm.container._port_available",
                        return_value=True,
                    ):
                        mock_result = Mock()
                        mock_result.returncode = 0
                        mock_result.stdout = "abc123def456"
                        mock_result.stderr = ""
                        with patch("subprocess.run", return_value=mock_result):
                            assert start_container() == 0

    def test_cdi_error_shows_instructions(self) -> None:
        """Should show CDI fix instructions when GPU passthrough fails."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_running", return_value=False):
                with patch(
                    "scicode_lint.vllm.container._container_exists",
                    return_value=False,
                ):
                    with patch(
                        "scicode_lint.vllm.container._port_available",
                        return_value=True,
                    ):
                        mock_result = Mock()
                        mock_result.returncode = 125
                        mock_result.stdout = ""
                        mock_result.stderr = (
                            "Error: CDI specification references an unknown device: nvidia.com/gpu"
                        )
                        with patch("subprocess.run", return_value=mock_result):
                            assert start_container() == 1

    def test_port_conflict_returns_error(self) -> None:
        """Should return error when port is in use."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_running", return_value=False):
                with patch(
                    "scicode_lint.vllm.container._container_exists",
                    return_value=False,
                ):
                    with patch(
                        "scicode_lint.vllm.container._port_available",
                        return_value=False,
                    ):
                        with patch(
                            "scicode_lint.vllm.container._identify_port_holder",
                            return_value="",
                        ):
                            assert start_container() == 1


class TestStopContainer:
    """Tests for stop_container."""

    def test_stop_running(self) -> None:
        """Should stop a running container."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=True):
                with patch(
                    "scicode_lint.vllm.container._container_running",
                    return_value=True,
                ):
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stderr = ""
                    with patch("subprocess.run", return_value=mock_result):
                        assert stop_container() == 0

    def test_stop_not_running(self) -> None:
        """Should return 0 when container exists but is not running."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=True):
                with patch(
                    "scicode_lint.vllm.container._container_running",
                    return_value=False,
                ):
                    assert stop_container() == 0

    def test_stop_not_exists(self) -> None:
        """Should return 0 when container does not exist."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=False):
                assert stop_container() == 0

    def test_stop_no_runtime(self) -> None:
        """Should return 1 when no container runtime found."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value=None,
        ):
            assert stop_container() == 1


class TestRemoveContainer:
    """Tests for remove_container."""

    def test_remove_stopped(self) -> None:
        """Should remove a stopped container."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=True):
                with patch(
                    "scicode_lint.vllm.container._container_running",
                    return_value=False,
                ):
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stderr = ""
                    with patch("subprocess.run", return_value=mock_result):
                        assert remove_container() == 0

    def test_remove_running_requires_force(self) -> None:
        """Should refuse to remove running container without force."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=True):
                with patch(
                    "scicode_lint.vllm.container._container_running",
                    return_value=True,
                ):
                    assert remove_container(force=False) == 1

    def test_remove_running_with_force(self) -> None:
        """Should remove running container with force."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=True):
                with patch(
                    "scicode_lint.vllm.container._container_running",
                    return_value=True,
                ):
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stderr = ""
                    with patch("subprocess.run", return_value=mock_result) as mock_run:
                        assert remove_container(force=True) == 0
                        # Verify -f flag was passed
                        cmd = mock_run.call_args[0][0]
                        assert "-f" in cmd

    def test_remove_not_exists(self) -> None:
        """Should return 0 when container does not exist."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=False):
                assert remove_container() == 0


class TestContainerStatus:
    """Tests for container_status."""

    def test_status_not_exists(self) -> None:
        """Should report container does not exist."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=False):
                assert container_status() == 0

    def test_status_running(self) -> None:
        """Should show status when container is running."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=True):
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "running"
                with patch("subprocess.run", return_value=mock_result):
                    with patch("scicode_lint.vllm.get_server_info") as mock_info:
                        mock_info.return_value = Mock(
                            is_running=True,
                            base_url="http://localhost:5001",
                            model="qwen3-8b-fp8",
                            max_model_len=40960,
                        )
                        assert container_status() == 0


class TestContainerLogs:
    """Tests for container_logs."""

    def test_logs_not_exists(self) -> None:
        """Should return error when container does not exist."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=False):
                assert container_logs() == 1

    def test_logs_default(self) -> None:
        """Should show last 50 lines by default."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=True):
                mock_proc = Mock()
                mock_proc.wait.return_value = 0
                with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                    assert container_logs() == 0
                    cmd = mock_popen.call_args[0][0]
                    assert "--tail" in cmd
                    assert "50" in cmd
                    assert _CONTAINER_NAME in cmd

    def test_logs_follow(self) -> None:
        """Should pass -f flag when follow=True."""
        with patch(
            "scicode_lint.vllm.container._detect_container_runtime",
            return_value="podman",
        ):
            with patch("scicode_lint.vllm.container._container_exists", return_value=True):
                mock_proc = Mock()
                mock_proc.wait.return_value = 0
                with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                    assert container_logs(follow=True) == 0
                    cmd = mock_popen.call_args[0][0]
                    assert "-f" in cmd


# Prometheus metrics parsing coverage lives in tests/test_vllm_metrics.py
# since the parser now lives in scicode_lint.vllm.metrics.


class TestContainerMonitor:
    """Tests for container_monitor.

    Note: the monitor deliberately does NOT exit when the container is
    stopped or the API is loading — it shows those states in the live
    display until Ctrl+C. The only early-exit path is no container runtime
    installed on the host.
    """

    def test_monitor_returns_1_when_no_runtime(self) -> None:
        """Should return 1 when neither podman nor docker is installed."""
        with patch("scicode_lint.vllm.container._detect_container_runtime", return_value=None):
            assert container_monitor() == 1
