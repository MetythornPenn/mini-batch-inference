from __future__ import annotations

import random
from typing import List

from locust import HttpUser, between, task


def _random_vector(dim: int = 3) -> List[float]:
    return [random.uniform(-2.0, 2.0) for _ in range(dim)]


class PredictUser(HttpUser):
    """Simulates clients sending inference requests."""

    wait_time = between(0.1, 0.5)
    vector_dim = 3

    @task
    def predict(self) -> None:
        payload = {"x": _random_vector(self.vector_dim)}
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Unexpected status {resp.status_code}")
                return

            data = resp.json()
            if "y" not in data:
                resp.failure("Missing 'y' in response")
                return

            resp.success()
