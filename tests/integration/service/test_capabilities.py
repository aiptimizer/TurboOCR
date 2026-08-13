"""/capabilities endpoint + a dynamic guarantee that every advertised
HTTP endpoint is actually reachable (registered), not just documented."""

import requests


class TestCapabilities:
    def test_capabilities_returns_200_json(self, server_url):
        r = requests.get(f"{server_url}/capabilities", timeout=5)
        assert r.status_code == 200
        assert "application/json" in r.headers.get("Content-Type", "")
        cap = r.json()
        # Stable top-level shape.
        for key in ("build", "features", "pdf", "limits", "endpoints"):
            assert key in cap, f"/capabilities missing '{key}': {cap}"

    def test_capabilities_schema(self, server_url):
        cap = requests.get(f"{server_url}/capabilities", timeout=5).json()
        assert cap["build"] in ("gpu", "cpu")
        feats = cap["features"]
        for k in ("layout", "autorotate", "profile_endpoint"):
            assert isinstance(feats[k], bool)
        assert feats["grpc_response_mode"] in ("json_bytes", "structured")
        # profile_endpoint advertises the one cross-build HTTP divergence.
        assert feats["profile_endpoint"] == (cap["build"] == "cpu")
        pdf = cap["pdf"]
        assert isinstance(pdf["modes"], list) and "ocr" in pdf["modes"]
        # auto_verified is GPU-only.
        if "auto_verified" in pdf["modes"]:
            assert cap["build"] == "gpu"
        assert isinstance(pdf["default_dpi"], int) and 50 <= pdf["default_dpi"] <= 600
        for k in ("max_body_mb", "max_image_dim", "max_batch_images"):
            assert isinstance(cap["limits"][k], int) and cap["limits"][k] > 0
        assert isinstance(cap["endpoints"], list) and "/capabilities" in cap["endpoints"]

    def test_every_advertised_endpoint_is_reachable(self, server_url):
        """The 'literally covers all endpoints' guarantee: every path the
        server advertises in /capabilities must be REGISTERED — a GET must not
        404. POST-only endpoints answer GET with 405 (still registered), so we
        assert simply: not 404 and not a connection error."""
        cap = requests.get(f"{server_url}/capabilities", timeout=5).json()
        unreachable = []
        for path in cap["endpoints"]:
            try:
                r = requests.get(f"{server_url}{path}", timeout=10)
            except requests.RequestException as e:  # noqa: BLE001
                unreachable.append(f"{path}: {e}")
                continue
            if r.status_code == 404:
                unreachable.append(f"{path}: 404 (advertised but not registered)")
        assert not unreachable, "advertised-but-unreachable endpoints: " + "; ".join(unreachable)

    def test_profile_endpoint_matches_capabilities(self, server_url):
        """/profile exists only on the CPU build (PROFILE_STAGES). The
        capabilities flag must agree with reality."""
        cap = requests.get(f"{server_url}/capabilities", timeout=5).json()
        r = requests.get(f"{server_url}/profile", timeout=5)
        if cap["features"]["profile_endpoint"]:
            assert r.status_code != 404
        else:
            assert r.status_code == 404
