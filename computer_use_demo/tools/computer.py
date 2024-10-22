import asyncio
import base64
import os
import shlex
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from .base import BaseAnthropicTool, ComputerToolOptions, ToolError, ToolResult
from .run import run

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name = "computer"
    api_type = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(ScalingSource.COMPUTER,
                                               self.width, self.height)
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def __init__(self):
        super().__init__()

        self.width = int(os.getenv("WIDTH") or 0)
        self.height = int(os.getenv("HEIGHT") or 0)
        assert self.width and self.height, "WIDTH, HEIGHT must be set"
        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
            self._display_prefix = f"DISPLAY=:{self.display_num} "
        else:
            self.display_num = None
            self._display_prefix = ""

        self.xdotool = f"{self._display_prefix}xdotool"

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        r = Repl()
        return await r(action=action, text=text, coordinate=coordinate)


class Repl:
    _screenshot_delay = 2.0

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ) -> ToolResult:
        match action:
            case "key":
                return await self.key(text)
            case "type":
                return await self.type(text)
            case "mouse_move":
                return await self.mouse_move(coordinate)
            case "left_click":
                return await self.left_click()
            case "left_click_drag":
                return await self.left_click_drag(coordinate)
            case "right_click":
                return await self.right_click()
            case "middle_click":
                return await self.middle_click()
            case "double_click":
                return await self.double_click()
            case "screenshot":
                return await self.screenshot()
            case "cursor_position":
                return await self.cursor_position()
            case _:
                return ToolResult(error=f"unknown action: {action}")

    async def raw_screenshot(self) -> str:
        path = Path(OUTPUT_DIR) / f"screenshot_{uuid4().hex}.png"
        await run(f"mkdir -p {OUTPUT_DIR}")
        result = await self.shell(f"gnome-screenshot -f {path} -p",
                                  take_screenshot=False)
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode()
        raise Exception(f"screenshot: {result.error}")

    async def shell(self, command: str, take_screenshot=True):
        _, stdout, stderr = await run(command)
        base64_image = None

        if take_screenshot:
            await asyncio.sleep(self._screenshot_delay)
            base64_image = await self.raw_screenshot()

        return ToolResult(output=stdout,
                          error=stderr,
                          base64_image=base64_image)

    async def screenshot(self) -> ToolResult:
        screenshot = await self.raw_screenshot()
        return ToolResult(base64_image=screenshot)

    async def mouse_move(self,
                         coordinate: tuple[int, int] | None) -> ToolResult:
        if not coordinate:
            return ToolResult(
                error="for 'mouse_move' action, coordinate is required")
        return await self.shell("xdotool mousemove --sync %d %d" %
                                (coordinate[0], coordinate[1]))

    async def cursor_position(self) -> ToolResult:
        return await self.shell(
            "xdotool getmouselocation --shell | head -n2",
            take_screenshot=False,
        )

    async def left_click(self) -> ToolResult:
        return await self.shell("xdotool click 1")

    async def left_click_drag(
            self, coordinate: tuple[int, int] | None) -> ToolResult:
        if not coordinate:
            raise ToolError(
                "for 'left_click_drag' action, coordinate is required")
        return await self.shell("xdotool mousedown 1 mousemove --sync "
                                f"{coordinate[0]} {coordinate[1]} mouseup 1")

    async def middle_click(self) -> ToolResult:
        return await self.shell("xdotool click 2")

    async def right_click(self) -> ToolResult:
        return await self.shell("xdotool click 3")

    async def double_click(self) -> ToolResult:
        return await self.shell("xdotool click --repeat 2 --delay 500 1")

    async def key(self, text: str | None) -> ToolResult:
        if not text:
            raise ToolError("for 'key' action, text is required")
        return await self.shell(f"xdotool key -- {shlex.quote(text)}")

    async def type(self, text: str | None) -> ToolResult:
        if not text:
            raise ToolError("for 'type' action, text is required")
        result: ToolResult = ToolResult()
        for chunk in chunks(text, TYPING_GROUP_SIZE):
            result = await self.shell(
                f"xdotool type --delay {TYPING_DELAY_MS} -- {shlex.quote(chunk)}",
                take_screenshot=False)
            if result.error:
                raise Exception(f"type: {result.error}")
        return result
