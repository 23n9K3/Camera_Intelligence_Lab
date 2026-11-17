from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Bayer = Literal["RGGB", "BGGR", "GRBG", "GBRG"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RawSpec(Model):
    width: int = Field(ge=2, le=16384)
    height: int = Field(ge=2, le=16384)
    bit_depth: Literal[8, 10, 12, 14, 16]
    storage_format: Literal["unpacked8", "unpacked16", "mipi10", "mipi12", "mipi14"]
    bayer_pattern: Bayer
    black_level: float = Field(ge=0)
    white_level: float
    endianness: Literal["little", "big"] = "little"

    @model_validator(mode="after")
    def check_layout(self):
        if self.width * self.height > 32_000_000:
            raise ValueError("RAW exceeds 32 million pixels")
        if not self.black_level < self.white_level <= (1 << self.bit_depth) - 1:
            raise ValueError("Invalid black or white level")
        if self.storage_format == "unpacked8" and self.bit_depth != 8:
            raise ValueError("8-bit container requires 8-bit data")
        if self.storage_format.startswith("mipi"):
            depth = int(self.storage_format[4:])
            group = 2 if depth == 12 else 4
            if self.bit_depth != depth or self.width % group or self.endianness != "little":
                raise ValueError("Invalid MIPI bit depth, row width or endianness")
        return self


class Inputs(Model):
    raw: str | None = None
    reference: str | None = None
    dark: list[str] = Field(default_factory=list)
    flat: list[str] = Field(default_factory=list)
    ptc: list[list[str]] = Field(default_factory=list)
    exposures: list[str] = Field(default_factory=list)
    exposure_times: list[float] = Field(default_factory=list)
    burst: list[str] = Field(default_factory=list)
    edge: str | None = None
    uniform_roi: tuple[int, int, int, int] | None = None

