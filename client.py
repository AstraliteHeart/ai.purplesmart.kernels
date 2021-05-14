import argparse
import asyncio
import functools
import importlib
import logging
import os
import socket
import threading
import time
import uuid
from typing import List

import py3nvml
import ray
import toml
from aiohttp import web
from autobahn.asyncio.component import Component, run
from dotenv import load_dotenv
from furl import furl
from ray import serve
from redis import client

from tui import TUI

load_dotenv()
WS_ENDPOINT = os.getenv("WS_ENDPOINT")
NODE_ID = uuid.getnode()
HOSTNAME = socket.gethostname()


def init(
    backend,
    num_replicas: int,
    max_concurrent_queries: int,
    resources,
    counter,
    conda_env,
):
    backend_path = backend.split(".")[1:-2]
    route = "/".join(backend_path).lower()

    backend_module, backend_class = backend.rsplit(".", 1)
    backend_module = importlib.import_module(backend_module)
    backend_class = getattr(backend_module, backend_class)

    ray_actor_options = {}
    if "cpu" in resources:
        ray_actor_options["num_cpus"] = resources["cpu"]

    if "gpu" in resources:
        ray_actor_options["num_gpus"] = resources["gpu"]

    if conda_env:
        conda_env = ray.serve.CondaEnv(conda_env)

    backend_class.options(
        name=route,
        route_prefix="/" + route,
        ray_actor_options=ray_actor_options,
        num_replicas=num_replicas,
        max_concurrent_queries=max_concurrent_queries,
    ).deploy(
        counter,
        route,
        max_concurrent_queries,
    )


def run_in_executor(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return inner


def _try_assign_replica(self, query):
    raise ValueError("_try_assign_replica")


# serve.router.ReplicaSet._try_assign_replica = _try_assign_replica

component = Component(
    transports=[
        {
            "type": "websocket",
            "url": WS_ENDPOINT,
            "options": {
                "auto_ping_interval": 10000,
                "auto_ping_timeout": 5000,
                "auto_ping_size": 4,
            },
        },
    ],
    authentication={
        "ticket": {
            "authid": "drone",
            "ticket": "pone_drone",
        }
    },
    realm="gpt2",
)


def override():
    def print_to_stdstream(data):
        logging.info("\n".join(data["lines"]))

    ray.worker.print_to_stdstream = print_to_stdstream


class Client:
    def __init__(self, user: str, visibility: str) -> None:
        self.client = None
        self.tui = None
        self.user = user
        self.visibility = visibility

    """
    async def handle_request(self, request):
        # Offload the computation to our Ray Serve backend.
        my_handle = self.client.get_handle("score/v1")
        result = await my_handle.remote("dummy input")
        return web.Response(text=result)
    """

    def init_client(self, gpus):
        gpus = [int(gpu.strip()) for gpu in gpus.split(",")]
        if gpus:
            py3nvml.grab_gpus(len(gpus), gpu_fraction=0, gpu_select=gpus)

        ray.init(num_gpus=len(gpus), configure_logging=False, include_dashboard=False)

        import counter

        self.counter = counter.create_counter()
        self.client = serve.start(detached=True)

    def init_modules(self, modules: List[str]) -> None:
        logging.info(f"Loading configs")
        for module in modules:
            logging.info(f"Loading config: {module}")

            config = toml.load(os.path.join("configs", f"{module}.toml"))

            for kernel in config["kernels"]["kernel"]:
                kernel_name = kernel["name"]
                logging.info(f"Creating kernel {kernel_name}")
                init(
                    "kernels." + kernel_name,
                    kernel.get("num_replicas", 1),
                    kernel.get("max_concurrent_queries", 8),
                    kernel.get("resources"),
                    self.counter,
                    kernel.get("conda_env"),
                )

    def fetch_resources(self):
        if not self.client:
            return {}

        try:
            configs = serve.list_endpoints().items()
        except AttributeError:
            return {}

        backend_to_endpoints = {}
        for endpoint_name, endpoint_config in configs:
            backend = list(endpoint_config["traffic"].keys())[0]
            backend_to_endpoints.setdefault(backend, []).append(endpoint_name)

        if not len(backend_to_endpoints.keys()):
            return {}

        resources = ray.get(self.counter.read.remote())

        if self.tui:
            self.tui.resources_by_endpoint = resources
        return resources

    def callback(self, tui):
        self.fetch_resources()

    def start_tui(self):
        self.tui = TUI(NODE_ID, {})
        override()

        self.x = threading.Thread(
            target=lambda: self.tui.live(self.callback), daemon=True
        )
        self.x.start()

    def stop_tui(self):
        self.tui.stop()
        self.x.join()

    @run_in_executor
    def exec_remote(self, method, query, params):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ref = serve.get_deployment(method).get_handle().remote(query, **params)
        result = ray.get(ref)
        return result

    def init_ws(self):
        @component.register(
            f"com.purplesmart.{NODE_ID}.api",
        )
        async def api(wamp_request):
            method = wamp_request["method"]
            query = wamp_request["query"]
            params = wamp_request.get("params") or {}
            logging.critical(
                "api, method: %s, query: %s, params: %s",
                wamp_request["method"],
                wamp_request["query"],
                params,
            )
            try:
                outputs = await self.exec_remote(method, query, params)
                return outputs
            except:
                logging.exception("Api call failure")
                return {"outputs": "error"}

        @component.on_disconnect
        async def on_disconnect():
            logging.info("on_disconnect")

        @component.on_connectfailure
        async def on_connectfailure():
            logging.info("on_connectfailure")

        @component.on_join
        async def joined(session, details):
            logging.info("Joined")
            while True:
                resources_by_endpoint = self.fetch_resources()

                heartbeat_data = {}
                for endpoint, data in resources_by_endpoint.items():
                    endpoint_data = []
                    for agent, agent_data in data.items():
                        endpoint_data.append(agent_data.dict())
                    heartbeat_data[endpoint] = endpoint_data
                logging.debug("Heartbeat: %s", resources_by_endpoint)

                session.publish(
                    "com.purplesmart.heartbeat",
                    {
                        "node": NODE_ID,
                        "hostname": HOSTNAME,
                        "resources": heartbeat_data,
                        "visibility": self.visibility,
                        "user": self.user,
                        "heartbeat": 10.0
                    },
                )

                await asyncio.sleep(10.0)

        def run_main():
            run([component], log_level=None)
            component.start()

        run_main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PurpleSmart.ai client.")
    parser.add_argument("--crossbar_server", default=True, action="store_true")
    parser.add_argument(
        "--no-crossbar_server", dest="crossbar-server", action="store_false"
    )
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--gpu", dest="gpu", required=False)
    parser.add_argument(
        "--disable-tui", dest="disable_tui", action="store_true", default=False
    )
    parser.add_argument("--app-url", dest="app_url")

    parser.add_argument(
        "--visibility",
        dest="visibility",
        choices=["public", "private", "unlisted"],
        default="unlisted",
    )

    parser.add_argument("--user", dest="user")

    args = parser.parse_args()

    client = Client(args.user, args.visibility)

    if not args.disable_tui:
        client.start_tui()
    try:
        client.init_client(args.gpu)

        if args.app_url:
            app_url = furl(args.app_url).add({"kernel": NODE_ID})
            print(f"Application url: {app_url}")

        client.init_modules(args.config)
        if args.crossbar_server:
            client.init_ws()
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt as e:
        if not args.disable_tui:
            client.stop_tui()
        raise e
