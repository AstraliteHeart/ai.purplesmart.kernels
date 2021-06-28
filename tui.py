import collections
import datetime
import io
import logging
import shutil
from time import sleep

import asciichartpy
import humanize
import psutil
from cpuinfo import get_cpu_info
from py3nvml import py3nvml
from pyfiglet import figlet_format
from rich import box
from rich.align import Align
from rich.ansi import AnsiDecoder
from rich.columns import Columns
from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.measure import Measurement, measure_renderables
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

COLORS = [asciichartpy.green, asciichartpy.magenta, asciichartpy.red]
RICH_COLORS = ["green", "magenta", "red"]

CPU_NAME = get_cpu_info()
CPU_NAME = CPU_NAME.get("brand", CPU_NAME.get("brand_raw"))


BACKEND_COLORS = [
    asciichartpy.lightcyan,
    asciichartpy.lightmagenta,
    asciichartpy.lightblue,
    asciichartpy.lightyellow,
    asciichartpy.lightgreen,
]
RICH_BACKEND_COLORS = [
    "bright_cyan",
    "bright_magenta",
    "bright_blue",
    "bright_yellow",
    "bright_green",
]


class LogTail:
    def __init__(self, console) -> None:
        self.console = console
        self.decoder = AnsiDecoder()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        height = (options.height or options.size.height) - 2

        self.console.file.seek(0)
        last_lines = list(self.console.file.readlines()[-height:])
        self.console.file.seek(0, io.SEEK_END)
        lines = self.decoder.decode("".join(last_lines))
        yield Panel(Text("\n").join(lines))

        # yield Text('\n'.join(list(self.console.file.readlines())))


class AsciiGraph:
    def __init__(self, series, max, colors=COLORS) -> None:
        self.series = series
        self.max = max
        self.colors = colors
        self.decoder = AnsiDecoder()

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        def _slice(items, max_items):
            items = list(items)
            items_len = min(len(items), max_items)
            return items[-items_len:]

        height = (options.height or options.size.height) - 1
        graph = asciichartpy.plot(
            series=[_slice(s, options.max_width - 10) for s in self.series],
            cfg={
                "min": 0,
                "max": self.max,
                "height": height,
                "colors": self.colors,
            },
        )
        yield Text("\n").join(self.decoder.decode(graph))


class EndpointMonitor:
    def __init__(self, endpoint) -> None:
        self.endpoint = endpoint

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:

        items = []
        color_id = 0
        for endpoint_name, endpoint_data in self.endpoint.items():
            color = RICH_BACKEND_COLORS[color_id]
            color_id += 1
            item = Panel.fit(
                Endpoint(endpoint_data),
                title=f"[{color}]{endpoint_name}[/]",
                # box=box.SIMPLE,
            )
            items.append(item)

        yield Align.center(Columns(items), vertical="middle")


def emojify_device(s):
    return s.replace("cpu", "🐢").replace("gpu:", "🦄")


class Endpoint:
    def __init__(self, backends) -> None:
        self.backends = backends

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Columns(
            [
                Panel.fit(
                    Backend(backend.used, backend.available),
                    box=box.SQUARE,
                    title=emojify_device(backend.device),
                )
                for backend in self.backends.values()
            ],
        )

    def __rich_measure__(self, console: Console, max_width: int) -> Measurement:
        width = 0
        for backend in self.backends.values():
            panel = Panel.fit(
                Backend(backend.used, backend.available),
                box=box.SQUARE,
                title=emojify_device(backend.device),
            )
            size = measure_renderables(console, console.options, [panel])
            width += size.minimum

        width += len(self.backends) - 1
        return Measurement(width, width)


class Backend:
    def __init__(self, used, total) -> None:
        self.total = total
        self.used = used

    def __rich_console__(self, console: Console, options: ConsoleOptions):
        text = ""
        for current in range(self.total):
            if current < self.used:
                text += "[medium_purple1]⬛[/]"
            else:
                text += "[medium_purple4]⬛[/]"
        yield text

    def __rich_measure__(self, console: Console, max_width: int) -> Measurement:
        return measure_renderables(console, console.options, [Text("⬛" * self.total)])


class TUI:
    def __init__(self, node_id, resources_by_endpoint) -> None:
        self.node_id = node_id
        self.resources_by_endpoint = resources_by_endpoint
        self.start_time = datetime.datetime.now()

        self.layout = Layout(name="root")
        self.layout.split(
            Layout(name="header", size=6),
            Layout(name="gpu", size=15),
            Layout(name="cpu", size=15),
            Layout(name="endpoints", size=19),
            Layout(name="console", ratio=1),
        )

        self.layout["header"].split_row(
            Layout(name="logo", size=70),
            Layout(name="info", ratio=1),
        )

        self.gpu_layout = Layout(name="gpu")
        self.gpu_layout.split_row(
            Layout(name="utilization", ratio=1),
            Layout(name="memory", ratio=1),
        )

        self.cpu_layout = Layout(name="cpu")
        self.cpu_layout.split_row(
            Layout(name="utilization", ratio=1),
            Layout(name="memory", ratio=1),
        )

        self.endpoints_layout = Layout(name="cpu")
        self.endpoints_layout.split_row(
            Layout(name="data", ratio=1),
            Layout(name="graph", ratio=1),
        )

        py3nvml.nvmlInit()

        self.gpu_mem_usage = [
            collections.deque(maxlen=150),
            collections.deque(maxlen=150),
        ]
        self.gpu_usage = [collections.deque(maxlen=150), collections.deque(maxlen=150)]
        self.cpu_usage = [collections.deque(maxlen=150)]
        self.ram_usage = [collections.deque(maxlen=150)]

        self.layout["header"]["logo"].update(
            Text(figlet_format("PurpleSmart", font="slant"), style="magenta")
        )

        width, _ = shutil.get_terminal_size()
        self.console = Console(
            file=io.StringIO(),
            force_terminal=True,
            color_system="truecolor",
            width=width - 4,
        )
        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, console=self.console)],
        )

        self.tail = LogTail(self.console)

        self.gpu_usage_graph = AsciiGraph(self.gpu_usage, 100)
        max_mem = []
        for i in range(py3nvml.nvmlDeviceGetCount()):
            handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
            max_mem.append(meminfo.total)
        max_mem = int(round(max(max_mem) / 1024 ** 2))
        self.gpu_mem_usage_graph = AsciiGraph(self.gpu_mem_usage, max_mem)

        self.cpu_usage_graph = AsciiGraph(self.cpu_usage, 100)
        self.ram_usage_graph = AsciiGraph(
            self.ram_usage, int(round(psutil.virtual_memory().total / 1024 ** 2))
        )

        self.endpoints_past_values = {}
        self.stop_flag = False

    def live(self, callback):
        try:
            with Live(
                self.layout,
                refresh_per_second=2,
                screen=False,
                redirect_stderr=False,
                redirect_stdout=False,
            ) as live:
                while True:
                    if self.stop_flag:
                        break

                    if callback:
                        callback(self)

                    if not self.resources_by_endpoint:
                        self.layout["endpoints"].update(
                            Panel(
                                Align.center(
                                    Text("Waiting for endpoints to come alive"),
                                    vertical="middle",
                                )
                            )
                        )
                    else:
                        self.endpoints_layout["data"].update(
                            EndpointMonitor(self.resources_by_endpoint)
                        )

                        self.endpoints_values = []
                        updated_keys = set()
                        max_value = 0
                        for (
                            endpoint_name,
                            endpoint_data,
                        ) in self.resources_by_endpoint.items():
                            updated_keys.add(endpoint_name)  # todo: finish

                            total_used = 0
                            total_max = 0
                            for entry in endpoint_data.values():
                                total_used += entry.used
                                total_max += entry.available

                            max_value = max(max_value, total_max)

                            # self.endpoints_values.append(data)
                            past_entries = self.endpoints_past_values.setdefault(
                                endpoint_name, []
                            )
                            past_entries.append(total_used)

                            self.endpoints_values.append(past_entries)

                        self.endpoints_graph = AsciiGraph(
                            self.endpoints_values, max_value, BACKEND_COLORS
                        )
                        self.endpoints_layout["graph"].update(self.endpoints_graph)

                        self.layout["endpoints"].update(
                            Panel(self.endpoints_layout, title="Endpoints")
                        )

                    uptime = datetime.datetime.now() - self.start_time
                    self.layout["header"]["info"].update(
                        Align.right(
                            Text(
                                f"""Node ID: {self.node_id}
                                Uptime: {humanize.naturaldelta(uptime)}
                                https://discord.gg/94KqBcE"""
                            ),
                            vertical="middle",
                        )
                    )

                    titles = []

                    table = Table.grid()
                    table.add_column(style="green")
                    table.add_column(no_wrap=True)

                    self.cpu_usage[0].append(psutil.cpu_percent(interval=None))
                    self.ram_usage[0].append(
                        int(round(psutil.virtual_memory().used / 1024 ** 2))
                    )

                    total_gpus_actual = py3nvml.nvmlDeviceGetCount()
                    for i in range(total_gpus_actual):
                        handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
                        meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                        utilization_info = py3nvml.nvmlDeviceGetUtilizationRates(handle)

                        table.add_row(
                            py3nvml.nvmlDeviceGetName(handle),
                            str(round(meminfo.used / 1024 ** 2)),
                        )
                        self.gpu_mem_usage[i].append(round(meminfo.used / 1024 ** 2))
                        self.gpu_usage[i].append(utilization_info.gpu)

                        color = RICH_COLORS[i]
                        titles.append(
                            f"[{color}]"
                            + py3nvml.nvmlDeviceGetName(handle)
                            + f" {utilization_info.gpu}%, {humanize.naturalsize(meminfo.used)}/{humanize.naturalsize(meminfo.total)}"
                            + "[/]"
                        )

                    self.gpu_layout["utilization"].update(
                        self.gpu_usage_graph,
                    )
                    self.gpu_layout["memory"].update(self.gpu_mem_usage_graph)

                    self.layout["gpu"].update(
                        Panel(self.gpu_layout, title=" ".join(titles))
                    )

                    self.cpu_layout["utilization"].update(Panel(self.cpu_usage_graph))
                    self.cpu_layout["memory"].update(Panel(self.ram_usage_graph))

                    self.cpu_layout["utilization"].update(
                        self.cpu_usage_graph,
                    )
                    self.cpu_layout["memory"].update(self.ram_usage_graph)

                    self.layout["cpu"].update(Panel(self.cpu_layout, title=CPU_NAME))

                    self.layout["console"].update(self.tail)

                    sleep(1.0)
        except KeyboardInterrupt as e:
            py3nvml.nvmlShutdown()
            raise e

    def stop(self) -> None:
        self.stop_flag = True


if __name__ == "__main__":
    import random

    def callback(tui):
        tui.resources_by_endpoint = {
            "emoji/v1": [
                {
                    "used": random.randint(0, 8),
                    "available": 8,
                    "active": True,
                    "device": "cpu",
                },
                {
                    "used": random.randint(0, 8),
                    "available": 8,
                    "active": True,
                    "device": "cpu",
                },
            ],
            "generate/v1": [
                {
                    "used": random.randint(0, 8),
                    "available": 8,
                    "active": True,
                    "device": "gpu:0",
                },
                {
                    "used": random.randint(0, 8),
                    "available": 8,
                    "active": True,
                    "device": "gpu:0",
                },
            ],
            "summarize/v1": [
                {
                    "used": random.randint(0, 8),
                    "available": 8,
                    "active": True,
                    "device": "gpu:1",
                },
                {
                    "used": random.randint(0, 8),
                    "available": 8,
                    "active": True,
                    "device": "gpu:1",
                },
            ],
            "score/v1": [
                {
                    "used": random.randint(0, 8),
                    "available": 8,
                    "active": True,
                    "device": "gpu:1",
                },
                {
                    "used": random.randint(0, 8),
                    "available": 8,
                    "active": True,
                    "device": "gpu:1",
                },
            ],
            "tts/v1": [
                {
                    "used": random.randint(0, 4),
                    "available": 4,
                    "active": True,
                    "device": "gpu:1",
                },
            ],
        }

    tui = TUI(
        "12345",
        {},
    )
    tui.live(callback)
