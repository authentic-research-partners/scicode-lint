"""Tests for vLLM server utilities."""

import warnings
from unittest.mock import Mock, patch

import httpx
import pytest

from scicode_lint.vllm import (
    GPUInfo,
    ServerInfo,
    VLLMServer,
    _get_default_model,
    get_gpu_info,
    get_server_info,
    is_running,
    wait_for_ready,
)

# Use the same default model as production code (DRY)
DEFAULT_MODEL = _get_default_model()


class TestIsRunning:
    """Tests for is_running function."""

    def test_is_running_when_server_responds(self) -> None:
        """Should return True when server health endpoint returns 200."""
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            assert is_running("http://localhost:5001") is True
            mock_get.assert_called_once_with("http://localhost:5001/health", timeout=2)

    def test_is_running_when_server_down(self) -> None:
        """Should return False when server is not responding."""
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection refused")

            assert is_running("http://localhost:5001") is False

    def test_is_running_with_non_200_status(self) -> None:
        """Should return False when server returns non-200 status."""
        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_get.return_value = mock_response

            assert is_running("http://localhost:5001") is False


class TestWaitForReady:
    """Tests for wait_for_ready function."""

    def test_wait_for_ready_immediate(self) -> None:
        """Should return True immediately if server is ready."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            assert wait_for_ready(timeout=5) is True

    def test_wait_for_ready_after_delay(self) -> None:
        """Should return True after server becomes ready."""
        call_count = [0]

        def mock_is_running(_: str) -> bool:
            call_count[0] += 1
            return call_count[0] >= 3  # Ready on third call

        with patch("scicode_lint.vllm.is_running", side_effect=mock_is_running):
            with patch("time.sleep"):  # Speed up test
                assert wait_for_ready(timeout=10, check_interval=0.1) is True

    def test_wait_for_ready_timeout(self) -> None:
        """Should return False if timeout reached."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("time.sleep"):  # Speed up test
                assert wait_for_ready(timeout=1, check_interval=0.1) is False


class TestVLLMServerContextManager:
    """Tests for VLLMServer context manager."""

    def test_local_server_not_running_starts_container(self) -> None:
        """Should start container if not running, stop on exit."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm.container.start_container", return_value=0) as mock_start:
                with patch("scicode_lint.vllm.wait_for_ready", return_value=True):
                    with patch(
                        "scicode_lint.vllm.container.stop_container", return_value=0
                    ) as mock_stop:
                        with VLLMServer():
                            pass

                        mock_start.assert_called_once()
                        mock_stop.assert_called_once()

    def test_local_server_already_running_reuses(self) -> None:
        """Should reuse local server if already running, not stop on exit."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("scicode_lint.vllm.httpx.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {"data": [{"id": DEFAULT_MODEL}]}

                with patch("scicode_lint.vllm.container.start_container") as mock_start:
                    with patch("scicode_lint.vllm.container.stop_container") as mock_stop:
                        with VLLMServer():
                            pass

                        mock_start.assert_not_called()
                        mock_stop.assert_not_called()

    def test_local_server_wrong_model_warns(self) -> None:
        """Should warn if local server running with different model."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("scicode_lint.vllm.httpx.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {"data": [{"id": "different-model"}]}

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    with VLLMServer(model=DEFAULT_MODEL):
                        pass

                    assert len(w) == 1
                    assert "running with model 'different-model'" in str(w[0].message)
                    assert issubclass(w[0].category, RuntimeWarning)

    def test_remote_server_running_reuses(self) -> None:
        """Should reuse remote server if running."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("scicode_lint.vllm.container.start_container") as mock_start:
                with patch("scicode_lint.vllm.container.stop_container") as mock_stop:
                    with VLLMServer(base_url="http://remote:5001"):
                        pass

                    mock_start.assert_not_called()
                    mock_stop.assert_not_called()

    def test_remote_server_not_running_raises(self) -> None:
        """Should raise error if remote server not reachable."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with pytest.raises(RuntimeError, match="Remote vLLM server not reachable"):
                with VLLMServer(base_url="http://remote:5001"):
                    pass

    def test_container_start_failure_raises(self) -> None:
        """Should raise RuntimeError if container start fails."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm.container.start_container", return_value=1):
                with pytest.raises(RuntimeError, match="Failed to start vLLM container"):
                    with VLLMServer():
                        pass

    def test_container_timeout_raises(self) -> None:
        """Should raise TimeoutError if server not ready in time."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm.container.start_container", return_value=0):
                with patch("scicode_lint.vllm.wait_for_ready", return_value=False):
                    with patch("scicode_lint.vllm.container.stop_container"):
                        with pytest.raises(TimeoutError, match="not ready within"):
                            with VLLMServer(wait_timeout=1):
                                pass

    def test_exception_in_context_still_stops_container(self) -> None:
        """Should stop container even if exception raised in context."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            with patch("scicode_lint.vllm.container.start_container", return_value=0):
                with patch("scicode_lint.vllm.wait_for_ready", return_value=True):
                    with patch("scicode_lint.vllm.container.stop_container") as mock_stop:
                        with pytest.raises(ValueError):
                            with VLLMServer():
                                raise ValueError("Test error")

                        mock_stop.assert_called_once()


class TestGetGPUInfo:
    """Tests for get_gpu_info function."""

    def test_get_gpu_info_success(self) -> None:
        """Should parse nvidia-smi output correctly."""
        mock_result = Mock()
        mock_result.stdout = "NVIDIA RTX 4000 Ada, 20480, 4096, 16384, 15"

        mock_cuda = Mock()
        mock_cuda.stdout = "535.183.01"

        with patch("subprocess.run", side_effect=[mock_result, mock_cuda]):
            info = get_gpu_info()

            assert info is not None
            assert info.name == "NVIDIA RTX 4000 Ada"
            assert info.total_memory_mb == 20480
            assert info.used_memory_mb == 4096
            assert info.free_memory_mb == 16384
            assert info.utilization_percent == 15
            assert info.cuda_version == "535.183.01"

    def test_get_gpu_info_nvidia_smi_not_found(self) -> None:
        """Should return None if nvidia-smi not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            info = get_gpu_info()
            assert info is None

    def test_get_gpu_info_parse_error(self) -> None:
        """Should return None if parsing fails."""
        mock_result = Mock()
        mock_result.stdout = "invalid output"

        with patch("subprocess.run", return_value=mock_result):
            info = get_gpu_info()
            assert info is None


class TestGetServerInfo:
    """Tests for get_server_info function."""

    def test_get_server_info_running(self) -> None:
        """Should return server info when running."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("httpx.get") as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                model_id = "RedHatAI/Qwen3-8B-FP8-dynamic"
                mock_response.json.return_value = {"data": [{"id": model_id}]}
                mock_get.return_value = mock_response

                info = get_server_info()

                assert info.is_running is True
                assert info.model == DEFAULT_MODEL
                assert info.base_url == "http://localhost:5001"

    def test_get_server_info_not_running(self) -> None:
        """Should return not running status."""
        with patch("scicode_lint.vllm.is_running", return_value=False):
            info = get_server_info()

            assert info.is_running is False
            assert info.model is None

    def test_get_server_info_model_fetch_fails(self) -> None:
        """Should handle model fetch failure gracefully."""
        with patch("scicode_lint.vllm.is_running", return_value=True):
            with patch("httpx.get", side_effect=httpx.HTTPError("Connection refused")):
                info = get_server_info()

                assert info.is_running is True
                assert info.model is None


class TestDataClasses:
    """Tests for dataclasses."""

    def test_gpu_info_dataclass(self) -> None:
        """Should create GPUInfo correctly."""
        info = GPUInfo(
            name="Test GPU",
            total_memory_mb=16000,
            used_memory_mb=8000,
            free_memory_mb=8000,
            utilization_percent=50,
            cuda_version="12.0",
        )

        assert info.name == "Test GPU"
        assert info.total_memory_mb == 16000
        assert info.used_memory_mb == 8000
        assert info.free_memory_mb == 8000
        assert info.utilization_percent == 50
        assert info.cuda_version == "12.0"

    def test_server_info_dataclass(self) -> None:
        """Should create ServerInfo correctly."""
        info = ServerInfo(model="test-model", is_running=True, base_url="http://localhost:5001")

        assert info.model == "test-model"
        assert info.is_running is True
        assert info.base_url == "http://localhost:5001"
