import logging
import os
from abc import ABC, abstractmethod
from timeit import default_timer as timer
import starlette
from typing import Optional, Any

import ray

from pydantic import BaseModel


class Runner(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def func(self):
        pass

    def __init__(self, counter, name, max_concurrent_queries) -> None:
        self.log = logging.getLogger()
        self.counter = counter
        self.name = name

        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kernel_data")

        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_devices = list(map(str.strip, devices.split(",")))

        device_id = ray.get_gpu_ids()
        logging.debug("GPU: %s %s", visible_devices, device_id)
        if device_id:
            device_id_str = f"gpu:{device_id[0]}"
            device_id = visible_devices.index(device_id[0])
        else:
            device_id = None
            device_id_str = "cpu"

        self.device_id = device_id

        self.actor_id = ray.get_runtime_context().actor_id.hex()
        self.counter.register.remote(
            self.name, self.actor_id, max_concurrent_queries, device_id_str
        )
        logging.info(
            "Registered replica %s as %s [%s]", self.name, self.actor_id, device_id_str
        )

    def normalize_input(self, input) -> str:
        if isinstance(input, str):
            return input
        else:
            return input.decode("utf-8")

    async def __call__(self, request, **kwargs):
        start = timer()
        self.log.debug("Starting task %s", self.name)
        result = None
        error = None

        if isinstance(request, starlette.requests.Request):
            query = (await request.body()).decode("utf-8")
            kwargs = request.query_params
        else:
            query = request
            kwargs = kwargs

        print('%s %s' % (query, kwargs))
        try:
            result = (await self.func(query, **kwargs))
        except Exception as e:
            error = Error(message=str(e), type="runner_error")
            logging.exception("Runner error")

        gen_seconds = timer() - start
        self.log.debug("Finished task %s in %.2fs", self.name, gen_seconds)
        ray.get(self.counter.decrement.remote(self.name, self.actor_id))
        return Response(output=result, timing=gen_seconds, error=error).dict()


class Error(BaseModel):
    type: str
    message: str


class Response(BaseModel):
    error: Optional[Error]
    output: Any
    timing: float
