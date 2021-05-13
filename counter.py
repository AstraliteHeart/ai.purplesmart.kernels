import ray
from pydantic import BaseModel
from typing import Dict


class ReplicaUsage(BaseModel):
    """Class for keeping track of an item in inventory."""

    used: int
    available: int
    device: str


BackendUsage = Dict[str, Dict[str, ReplicaUsage]]


@ray.remote
class Counter:
    def __init__(self):
        self.backend_data = {}

    def register(
        self, name: str, actor_id: str, max_concurrent_queries: int, device: str
    ):
        """Associate a new replica usage entry with an actor.

        Args:
            name (str): [description]
            actor_id (str): [description]
            max_concurrent_queries (int): [description]
            device (str): [description]

        Returns:
            str: [description]
        """
        replicas = self.backend_data.setdefault(name, {})
        replicas[actor_id] = ReplicaUsage(
            used=0, available=max_concurrent_queries, device=device
        )

    def set(self, name: str, replica_id, count: int) -> None:
        """[summary]

        Args:
            name (str): [description]
            replica_id ([type]): [description]
            count (int): [description]
        """
        name = "/".join(name.split(".")[1:-2]).lower()
        self.backend_data[name][replica_id].used = count

    def read(self) -> BackendUsage:
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.backend_data

    def increment(self, name: str, replica_id: str) -> None:
        """[summary]

        Args:
            name (str): [description]
            replica_id (str): [description]
        """
        self.backend_data[name][replica_id].used += 1

    def decrement(self, name: str, replica_id: str) -> None:
        """[summary]

        Args:
            name (str): [description]
            replica_id (str): [description]
        """
        self.backend_data[name][replica_id].used -= 1


def create_counter():
    return Counter.options(name="CounterActor", lifetime="detached").remote()
