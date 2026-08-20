"""P-3 — `env_hash_v2` 가 **조건에만** 의존하는가.

구 정의는 `env` 전체를 해싱해서, 조건이 같아도 다시 돌리면 값이 바뀌었다.
이 캠페인에서 실제로 겪었다 (메모리 클럭 기록을 빠뜨려 재실행 -> 해시 변경).
"""
from __future__ import annotations

import copy

import pytest

from kerneltab.core.env_hash import (
    ENV_HASH_KEYS_V2,
    EXCLUDED_WITH_REASON,
    env_hash_v2,
    hash_inputs,
)


@pytest.fixture
def env():
    return {
        "hardware": {"name": "GPU", "sm_count": 84, "peak_tflops_f16": 116.1},
        "nvcc_arch_flag": "sm_86",
        "protocol": {"target_ms": 20.0, "min_reps_floor": 5},
        "soak": {"enabled": False},
        "segments": {"kernels": 500, "warmup_seconds": 20},
        "clock_locked": True, "locked_mhz": 1350,
        "mem_clock_locked": True, "locked_mem_mhz": 7601,
        "peak_tflops_f16_effective": 116.1,
        "bandwidth_gbps_effective": 729.7,
        "shuffle_seed": 12345,
        "cutlass": {"commit": "abc123", "dir": "/home/someone/cutlass"},
        "cuda": {"nvcc_version": "12.4.99", "nvcc_path": "/usr/local/cuda/bin/nvcc"},
        # 아래는 해시에 들어가면 안 된다
        "created_utc": "2026-08-19T00:00:00Z",
        "host": {"hostname": "box-a", "ram_available_gb": 31.2},
        "launch_overhead_ms": 0.00321,
        "launch_overhead": {"median_ms": 0.00321},
        "device_index": 3,
        "manifest_hash": "deadbeef",
        "env_hash": "무시돼야 한다",
    }


class TestStability:
    def test_same_condition_same_hash(self, env):
        assert env_hash_v2(env) == env_hash_v2(copy.deepcopy(env))

    @pytest.mark.parametrize("field,value", [
        ("created_utc", "2027-01-01T00:00:00Z"),
        ("host", {"hostname": "다른-호스트", "ram_available_gb": 7.7}),
        ("launch_overhead_ms", 0.00999),
        ("launch_overhead", {"median_ms": 0.00999}),
        ("device_index", 0),
        ("manifest_hash", "다른-값"),
        ("env_hash", "다른-값"),
    ])
    def test_volatile_fields_do_not_change_hash(self, env, field, value):
        """**이 테스트가 P-3 의 요점이다.** 실행마다 변하는 값이 해시를
        바꾸면 조건이 같은데도 재개가 끊긴다."""
        h = env_hash_v2(env)
        env[field] = value
        assert env_hash_v2(env) == h, f"{field} 가 해시를 바꿨다"

    def test_nested_volatile_paths_ignored(self, env):
        h = env_hash_v2(env)
        env["cutlass"]["dir"] = "/다른/경로"
        env["cuda"]["nvcc_path"] = "/다른/nvcc"
        assert env_hash_v2(env) == h


class TestSensitivity:
    @pytest.mark.parametrize("field,value", [
        ("locked_mhz", 1800),
        ("mem_clock_locked", False),
        ("peak_tflops_f16_effective", 154.8),
        ("bandwidth_gbps_effective", 768.0),
        ("shuffle_seed", 999),
        ("nvcc_arch_flag", "sm_89"),
    ])
    def test_condition_fields_change_hash(self, env, field, value):
        h = env_hash_v2(env)
        env[field] = value
        assert env_hash_v2(env) != h, f"{field} 가 바뀌었는데 해시가 같다"

    @pytest.mark.parametrize("path,value", [
        (("protocol", "target_ms"), 50.0),
        (("soak", "enabled"), True),
        (("segments", "kernels"), 300),
        (("segments", "warmup_seconds"), 60),
        (("cutlass", "commit"), "다른커밋"),
        (("cuda", "nvcc_version"), "12.8.0"),
        (("hardware", "sm_count"), 128),
    ])
    def test_nested_condition_fields_change_hash(self, env, path, value):
        h = env_hash_v2(env)
        env[path[0]][path[1]] = value
        assert env_hash_v2(env) != h, f"{'.'.join(path)} 가 해시를 안 바꿨다"

    def test_segments_and_soak_are_included(self):
        """계획서 작성 뒤에 추가된 조건이다. 빠뜨리면 드리프트 대책이
        다른 데이터가 같은 해시로 섞인다."""
        assert "segments" in ENV_HASH_KEYS_V2
        assert "soak" in ENV_HASH_KEYS_V2


class TestDocumentation:
    def test_exclusions_have_reasons(self):
        """왜 뺐는지 남아 있어야 한다 — 나중에 다시 넣으려 할 때 근거가 된다."""
        for k, why in EXCLUDED_WITH_REASON.items():
            assert why and len(why) > 5, f"{k} 의 제외 이유가 비어 있다"

    def test_hash_inputs_is_explainable(self, env):
        inp = hash_inputs(env)
        assert set(inp) == set(ENV_HASH_KEYS_V2)
        assert inp["locked_mhz"] == 1350
        assert inp["cutlass.commit"] == "abc123"
