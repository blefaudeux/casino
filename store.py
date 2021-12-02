# example from torch distributed

# >>> server_store = dist.TCPStore("127.0.0.1", 1234, 2, True, timedelta(seconds=30))
# >>> # Run on process 2 (client)
# >>> client_store = dist.TCPStore("127.0.0.1", 1234, 2, False)
# >>> # Use any of the store methods from either the client or server after initialization
# >>> server_store.set("first_key", "first_value")
# >>> client_store.get("first_key")

import torch
from datetime import timedelta

IP = "127.0.0.1"
PORT = 1234

# FIXME: should probably be set to false in a truly adversarial environment
WAIT_FOR_WORKER = True
TIMEOUT = timedelta(seconds=30)


def get_server_store(ip: str = IP, port: int = PORT, world_size: int = -1):
    return torch.distributed.TCPStore(ip, port, world_size, WAIT_FOR_WORKER, TIMEOUT)


def get_client_store(ip: str = IP, port: int = PORT, world_size: int = -1):
    return torch.distributed.TCPStore(ip, port, world_size, WAIT_FOR_WORKER, TIMEOUT)
