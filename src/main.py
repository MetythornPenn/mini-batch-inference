import asyncio
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel, Field


def select_best_device() -> str:
    """
    Pick the GPU with the most free memory; fall back to CPU if CUDA is unavailable.
    If the DEVICE env var is set, it takes precedence (e.g., DEVICE=cuda:0 or cpu).
    """
    env_device = os.getenv("DEVICE")
    if env_device:
        return env_device

    if torch.cuda.is_available():
        best_idx = None
        best_free_mem = -1
        for idx in range(torch.cuda.device_count()):
            try:
                free_mem, _ = torch.cuda.mem_get_info(idx)
            except RuntimeError:
                continue
            if free_mem > best_free_mem:
                best_free_mem = free_mem
                best_idx = idx
        if best_idx is not None:
            return f"cuda:{best_idx}"

    return "cpu"


class MLP(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 16, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PredictRequest(BaseModel):
    x: List[float] = Field(
        ..., description="Input features", example=[1.0, 2.0, 4.0]
    )

class PredictResponse(BaseModel):
    y: float
    batch_size_used: int

@dataclass
class _Item:
    x: torch.Tensor
    fut: asyncio.Future


class MicroBatcher:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda:1",
        max_batch_size: int = 1024,
        max_wait_ms: int = 100,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._queue: asyncio.Queue[_Item] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

        # For CPU: limit intra-op threads if you want more predictable latency
        # torch.set_num_threads(1)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task:
            await self._task

    async def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enqueue one request. Returns the per-request output tensor.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_Item(x=x, fut=fut))
        return await fut  # resolved by batcher

    async def _run_loop(self) -> None:
        """
        Background loop that collects items and runs batched inference.
        """
        while not self._stopped.is_set():
            try:
                # Wait for at least one item
                first = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            items = [first]
            start_t = time.perf_counter()

            # Collect more until batch full or max_wait reached
            while len(items) < self.max_batch_size:
                remaining = (self.max_wait_ms / 1000.0) - (time.perf_counter() - start_t)
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    items.append(nxt)
                except asyncio.TimeoutError:
                    break

            # Run model in a batch
            try:
                with torch.inference_mode():
                    batch_x = torch.stack([it.x for it in items], dim=0).to(self.device)
                    batch_y = self.model(batch_x)  # shape [B, 1]
                    batch_y = batch_y.detach().cpu()

                # Resolve each request's future
                for i, it in enumerate(items):
                    if not it.fut.cancelled():
                        it.fut.set_result((batch_y[i], len(items)))
            except Exception as e:
                # Fail all pending items in this batch
                for it in items:
                    if not it.fut.cancelled():
                        it.fut.set_exception(e)


app = FastAPI()

DEVICE = select_best_device()
model = MLP(in_dim=3)
batcher = MicroBatcher(model=model, device=DEVICE, max_batch_size=2048, max_wait_ms=100)


@app.on_event("startup")
async def _startup():
    await batcher.start()


@app.on_event("shutdown")
async def _shutdown():
    await batcher.stop()


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    x = torch.tensor(req.x, dtype=torch.float32)  # shape [3]
    y_tensor, used_bs = await batcher.predict(x)

    # y_tensor shape [1]; make it float
    y = float(y_tensor.item())
    return PredictResponse(y=y, batch_size_used=used_bs)


@app.post("/predict_direct", response_model=PredictResponse)
async def predict_direct(req: PredictRequest):
    """Baseline inference endpoint that skips the micro-batcher."""

    x = torch.tensor(req.x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        y_tensor = model(x)

    y = float(y_tensor.squeeze(0).item())
    return PredictResponse(y=y, batch_size_used=1)
