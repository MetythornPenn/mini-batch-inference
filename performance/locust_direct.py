"""Locust file for load testing the /predict_direct endpoint."""

from __future__ import annotations

import random
from typing import List

from locust import HttpUser, between, task


def _random_vector(dim: int = 3) -> List[float]:
    return [random.uniform(-2.0, 2.0) for _ in range(dim)]


class PredictDirectUser(HttpUser):
    """Simulates direct (non-batched) inference requests."""

    wait_time = between(0.1, 0.5)
    vector_dim = 3

    @task
    def predict_direct(self) -> None:
        payload = {"x": _random_vector(self.vector_dim)}
        with self.client.post(
            "/predict_direct", json=payload, catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Unexpected status {resp.status_code}")
                return

            data = resp.json()
            if "y" not in data:
                resp.failure("Missing 'y' in response")
                return

            resp.success()
