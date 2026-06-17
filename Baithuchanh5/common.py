import json
import socket
from typing import Any, Dict, Optional

_BUFFERS: Dict[int, str] = {}


def create_server(host: str, port: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    return server


def connect_tcp(host: str, port: int) -> socket.socket:
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))
    return client


def send_json(connection: socket.socket, payload: Dict[str, Any]) -> None:
    message = (json.dumps(payload) + "\n").encode("utf-8")
    connection.sendall(message)


def recv_json(connection: socket.socket) -> Optional[Dict[str, Any]]:
    key = id(connection)
    buffer = _BUFFERS.get(key, "")

    while "\n" not in buffer:
        chunk = connection.recv(65536)
        if not chunk:
            _BUFFERS.pop(key, None)
            if not buffer.strip():
                return None
            break
        buffer += chunk.decode("utf-8")

    line, _, remaining = buffer.partition("\n")
    if remaining:
        _BUFFERS[key] = remaining
    else:
        _BUFFERS.pop(key, None)

    if not line.strip():
        return None
    return json.loads(line)


def close_quietly(connection: Optional[socket.socket]) -> None:
    if connection is not None:
        _BUFFERS.pop(id(connection), None)
        try:
            connection.close()
        except OSError:
            pass
