"""Tests for llmstack.docker.manager.DockerManager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import docker
import pytest
from docker.errors import APIError, NotFound

from llmstack.docker.manager import DockerManager


def _make_container(name: str, service: str, status: str = "running", ports: dict | None = None):
    container = MagicMock()
    container.name = name
    container.short_id = name[:8]
    container.status = status
    container.ports = ports or {}
    container.labels = {
        DockerManager.LABEL_MANAGED: "true",
        DockerManager.LABEL_SERVICE: service,
    }
    return container


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.ping.return_value = True
    with patch("llmstack.docker.manager.docker.from_env", return_value=client):
        yield client


@pytest.fixture
def manager(mock_client):
    return DockerManager(network_name="test_net")


def test_init_raises_systemexit_when_docker_unavailable():
    with patch(
        "llmstack.docker.manager.docker.from_env",
        side_effect=docker.errors.DockerException("boom"),
    ):
        with pytest.raises(SystemExit, match="Cannot connect to Docker daemon"):
            DockerManager()


def test_init_success(mock_client):
    mgr = DockerManager(network_name="my_net")
    assert mgr.network_name == "my_net"
    mock_client.ping.assert_called_once()


def test_managed_container_count(manager, mock_client):
    mock_client.containers.list.return_value = [
        _make_container("a", "ollama"),
        _make_container("b", "redis"),
    ]
    assert manager.managed_container_count == 2


def test_is_service_running_true(manager, mock_client):
    mock_client.containers.list.return_value = [_make_container("a", "ollama")]
    assert manager.is_service_running("ollama") is True


def test_is_service_running_false(manager, mock_client):
    mock_client.containers.list.return_value = [_make_container("a", "ollama")]
    assert manager.is_service_running("redis") is False


def test_ensure_network_creates_when_missing(manager, mock_client):
    mock_client.networks.get.side_effect = NotFound("missing")
    manager.ensure_network()
    mock_client.networks.create.assert_called_once_with("test_net", driver="bridge")


def test_ensure_network_noop_when_exists(manager, mock_client):
    mock_client.networks.get.return_value = MagicMock()
    manager.ensure_network()
    mock_client.networks.create.assert_not_called()


def test_build_image(manager, mock_client):
    tag = manager.build_image(path="/ctx", dockerfile="/ctx/Dockerfile", tag="img:local")
    assert tag == "img:local"
    mock_client.images.build.assert_called_once_with(
        path="/ctx", dockerfile="/ctx/Dockerfile", tag="img:local", rm=True
    )


def _make_service(name="ollama", category="inference", ports=None):
    svc = MagicMock()
    svc.name = name
    svc.category = category
    svc.container_spec.return_value = {
        "image": "img:local",
        "name": f"llmstack-{name}",
        "ports": {"11434/tcp": 11434} if ports is None else ports,
        "environment": {},
    }
    return svc


def test_run_service_removes_existing_container(manager, mock_client):
    existing = MagicMock()
    mock_client.containers.get.return_value = existing
    svc = _make_service()

    manager.run_service(svc)

    existing.stop.assert_called_once_with(timeout=10)
    existing.remove.assert_called_once_with(force=True)
    mock_client.containers.run.assert_called_once()
    _, kwargs = mock_client.containers.run.call_args
    assert kwargs["name"] == "llmstack-ollama"
    assert kwargs["network"] == "test_net"
    assert kwargs["labels"] == {
        DockerManager.LABEL_MANAGED: "true",
        DockerManager.LABEL_SERVICE: "ollama",
    }


def test_run_service_no_existing_container(manager, mock_client):
    mock_client.containers.get.side_effect = NotFound("nope")
    svc = _make_service()

    manager.run_service(svc)

    mock_client.containers.run.assert_called_once()


def test_run_service_port_conflict_raises_systemexit(manager, mock_client):
    mock_client.containers.get.side_effect = NotFound("nope")
    mock_client.containers.run.side_effect = APIError("port is already allocated")
    svc = _make_service()

    with pytest.raises(SystemExit, match="already in use"):
        manager.run_service(svc)


def test_run_service_port_conflict_no_ports(manager, mock_client):
    mock_client.containers.get.side_effect = NotFound("nope")
    mock_client.containers.run.side_effect = APIError("address already in use")
    svc = _make_service(ports={})

    with pytest.raises(SystemExit, match="unknown"):
        manager.run_service(svc)


def test_run_service_other_api_error_reraised(manager, mock_client):
    mock_client.containers.get.side_effect = NotFound("nope")
    mock_client.containers.run.side_effect = APIError("something else broke")
    svc = _make_service()

    with pytest.raises(APIError):
        manager.run_service(svc)


def test_stop_service_found(manager, mock_client):
    container = _make_container("llmstack-ollama", "ollama")
    mock_client.containers.list.return_value = [container]

    manager.stop_service("ollama")

    container.stop.assert_called_once_with(timeout=10)
    container.remove.assert_called_once_with(force=True)


def test_stop_service_not_found_is_noop(manager, mock_client):
    mock_client.containers.list.return_value = []
    manager.stop_service("ollama")  # should not raise


def test_stop_all(manager, mock_client):
    containers = [
        _make_container("llmstack-ollama", "ollama"),
        _make_container("llmstack-redis", "redis"),
    ]
    mock_client.containers.list.return_value = containers
    net = MagicMock()
    mock_client.networks.get.return_value = net

    stopped = manager.stop_all()

    assert stopped == ["llmstack-ollama", "llmstack-redis"]
    for c in containers:
        c.stop.assert_called_once_with(timeout=10)
        c.remove.assert_called_once_with(force=True)
    net.remove.assert_called_once()


def test_stop_all_with_volumes(manager, mock_client):
    mock_client.containers.list.return_value = []
    vol_a = MagicMock(name="llmstack_data")
    vol_a.name = "llmstack_data"
    vol_b = MagicMock(name="other_data")
    vol_b.name = "other_data"
    mock_client.volumes.list.return_value = [vol_a, vol_b]
    mock_client.networks.get.return_value = MagicMock()

    manager.stop_all(remove_volumes=True)

    vol_a.remove.assert_called_once_with(force=True)
    vol_b.remove.assert_not_called()


def test_stop_all_volume_remove_api_error_ignored(manager, mock_client):
    mock_client.containers.list.return_value = []
    vol = MagicMock()
    vol.name = "llmstack_data"
    vol.remove.side_effect = APIError("busy")
    mock_client.volumes.list.return_value = [vol]
    mock_client.networks.get.return_value = MagicMock()

    manager.stop_all(remove_volumes=True)  # should not raise


def test_stop_all_network_missing_is_ignored(manager, mock_client):
    mock_client.containers.list.return_value = []
    mock_client.networks.get.side_effect = NotFound("gone")

    manager.stop_all()  # should not raise


def test_stop_all_network_api_error_is_ignored(manager, mock_client):
    mock_client.containers.list.return_value = []
    mock_client.networks.get.side_effect = APIError("denied")

    manager.stop_all()  # should not raise


def test_get_container_found(manager, mock_client):
    container = _make_container("llmstack-ollama", "ollama")
    mock_client.containers.list.return_value = [container]

    assert manager.get_container("ollama") is container


def test_get_container_not_found(manager, mock_client):
    mock_client.containers.list.return_value = []
    assert manager.get_container("ollama") is None


def test_stream_logs_raises_when_no_container(manager, mock_client):
    mock_client.containers.list.return_value = []
    with pytest.raises(ValueError, match="No running container"):
        list(manager.stream_logs("ollama"))


def test_stream_logs_yields_decoded_lines(manager, mock_client):
    container = _make_container("llmstack-ollama", "ollama")
    container.logs.return_value = [b"line one\n", b"line two\n"]
    mock_client.containers.list.return_value = [container]

    lines = list(manager.stream_logs("ollama", follow=False, tail=10))

    assert lines == ["line one\n", "line two\n"]
    container.logs.assert_called_once_with(stream=True, follow=False, tail=10)


def test_list_services(manager, mock_client):
    container = _make_container(
        "llmstack-ollama", "ollama", status="running", ports={"11434/tcp": [{"HostPort": "11434"}]}
    )
    mock_client.containers.list.return_value = [container]

    result = manager.list_services()

    assert result == [
        {
            "name": "ollama",
            "container_name": "llmstack-ollama",
            "container_id": container.short_id,
            "status": "running",
            "ports": {"11434/tcp": [{"HostPort": "11434"}]},
        }
    ]
    container.reload.assert_called_once()


def test_managed_containers_filters_by_label(manager, mock_client):
    manager._managed_containers()
    mock_client.containers.list.assert_called_once_with(
        all=True, filters={"label": "llmstack.managed=true"}
    )
