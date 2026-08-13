"""Health endpoint integration tests."""

import concurrent.futures
import time

import requests


class TestHealth:
    def test_health_returns_200(self, server_url):
        r = requests.get(f"{server_url}/health", timeout=5)
        assert r.status_code == 200

    def test_health_under_concurrent_load(self, server_url):
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
            futures = [ex.submit(requests.get, f"{server_url}/health", timeout=5)
                       for _ in range(50)]
            results = [f.result() for f in futures]
        dt = time.perf_counter() - t0
        assert all(r.status_code == 200 for r in results)
        assert dt < 3.0, f"50 parallel /health took {dt:.2f}s (ceiling 3.0s)"

    def test_health_wrong_method(self, server_url):
        r = requests.post(f"{server_url}/health", timeout=5)
        assert r.status_code in (404, 405)
