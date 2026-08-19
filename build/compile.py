"""커널 생성 -> 컴파일 -> ptxas/SASS 정적 분석.

커널 하나당 산출물:
  build/artifacts/src/<kernel_id>.cu     생성된 CUTLASS 인스턴스화
  build/artifacts/lib/<kernel_id>.so     dlopen 가능한 커널
  results/kernels.jsonl 의 한 줄          자원 사용량 + 정적 분석 결과

정적 분석은 전부 "실행 불필요" 한 것들이다 (ncu 같은 프로파일러를 쓰지 않는다).
kt_info() 만 CUDA 컨텍스트가 필요해서 별도 패스로 돌린다.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from build import paths

__all__ = ["BuildEnv", "KtInfo", "build_kernel", "introspect"]


# ---------------------------------------------------------------------------
# 빌드 환경 (모든 워커가 공유하는 불변 값)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuildEnv:
    nvcc: str
    arch_flag: str
    cutlass_dir: Path
    src_dir: Path
    lib_dir: Path
    include_args: tuple[str, ...]

    @staticmethod
    def from_env_json(env: dict) -> BuildEnv:
        cutlass = Path(env["cutlass"]["dir"])
        src = paths.ARTIFACT_DIR / "src"
        lib = paths.ARTIFACT_DIR / "lib"
        src.mkdir(parents=True, exist_ok=True)
        lib.mkdir(parents=True, exist_ok=True)
        inc = (*tuple(paths.cutlass_includes(cutlass)), f"-I{paths.REPO_ROOT / 'measure'}")
        return BuildEnv(
            nvcc=env["cuda"]["nvcc_path"],
            arch_flag=env["nvcc_arch_flag"],
            cutlass_dir=cutlass,
            src_dir=src,
            lib_dir=lib,
            include_args=inc,
        )


# ---------------------------------------------------------------------------
# ptxas -v 파싱
# ---------------------------------------------------------------------------
_RE_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
_RE_PROPS = re.compile(r"Function properties for (\S+)")
_RE_STACK = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)
_RE_USED = re.compile(r"Used (\d+) registers")
_RE_SMEM = re.compile(r"(\d+) bytes smem")
_RE_CMEM = re.compile(r"(\d+) bytes cmem\[0\]")
_RE_SPILL_WARN = re.compile(
    r"Registers are spilled to local memory in function '([^']+)', "
    r"(\d+) bytes spill stores, (\d+) bytes spill loads"
)


def parse_ptxas(text: str) -> dict:
    """entry 별 자원 사용량. GEMM 커널은 보통 유일한 entry 다."""
    entries: dict[str, dict] = {}
    cur = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _RE_ENTRY.search(line)
        if m:
            cur = m.group(1)
            entries.setdefault(cur, {"name": cur, "target": m.group(2)})
            continue
        m = _RE_PROPS.search(line)
        if m:
            cur = m.group(1)
            entries.setdefault(cur, {"name": cur})
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            ms = _RE_STACK.search(nxt)
            if ms:
                entries[cur].update(
                    stack_frame=int(ms.group(1)),
                    spill_stores=int(ms.group(2)),
                    spill_loads=int(ms.group(3)),
                )
            continue
        m = _RE_USED.search(line)
        if m and cur:
            entries[cur]["regs_per_thread"] = int(m.group(1))
            ms = _RE_SMEM.search(line)
            entries[cur]["smem_static_bytes"] = int(ms.group(1)) if ms else 0
            mc = _RE_CMEM.search(line)
            if mc:
                entries[cur]["cmem_bytes"] = int(mc.group(1))
    for m in _RE_SPILL_WARN.finditer(text):
        e = entries.setdefault(m.group(1), {"name": m.group(1)})
        e["spill_stores"] = max(e.get("spill_stores", 0), int(m.group(2)))
        e["spill_loads"] = max(e.get("spill_loads", 0), int(m.group(3)))
    return entries


def pick_gemm_entry(entries: dict) -> dict:
    """GEMM 커널 entry 를 고른다. 이름에 GemmUniversal 이 있는 것 우선."""
    if not entries:
        return {}
    named = [e for n, e in entries.items() if "GemmUniversal" in n or "Kernel2" in n]
    pool = named or list(entries.values())
    return max(pool, key=lambda e: e.get("regs_per_thread", 0))


# ---------------------------------------------------------------------------
# SASS 분석 (cuobjdump 로 cubin 추출 -> nvdisasm -c)
# ---------------------------------------------------------------------------
_RE_INST = re.compile(r"/\*[0-9a-fA-F]{4,}\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9._]*)")


def analyze_sass(so_path: Path) -> dict:
    """SASS 명령어 카운트. 실행하지 않는 정적 분석이다."""
    out = {
        "hmma_count": None,
        "lds_count": None,
        "sts_count": None,
        "ldsm_count": None,
        "ldg_count": None,
        "cpasync_count": None,
        "inst_total": None,
        "sass_error": None,
    }
    try:
        cuobjdump = str(paths.cuda_bin("cuobjdump"))
        nvdisasm = str(paths.cuda_bin("nvdisasm"))
    except Exception as e:  # pragma: no cover
        out["sass_error"] = repr(e)
        return out

    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [cuobjdump, "-xelf", "all", str(so_path.resolve())],
            capture_output=True, text=True, cwd=td,
        )
        cubins = sorted(Path(td).glob("*.cubin"))
        if r.returncode != 0 or not cubins:
            out["sass_error"] = f"cuobjdump: rc={r.returncode} {r.stderr[-300:]}"
            return out
        counts: dict[str, int] = {}
        total = 0
        for cb in cubins:
            d = subprocess.run(
                [nvdisasm, "-c", str(cb)], capture_output=True, text=True
            )
            if d.returncode != 0:
                out["sass_error"] = f"nvdisasm: {d.stderr[-300:]}"
                return out
            for line in d.stdout.splitlines():
                m = _RE_INST.search(line)
                if not m:
                    continue
                op = m.group(1).split(".")[0]
                counts[op] = counts.get(op, 0) + 1
                total += 1

    out["hmma_count"] = counts.get("HMMA", 0)
    out["lds_count"] = counts.get("LDS", 0)
    out["sts_count"] = counts.get("STS", 0)
    out["ldsm_count"] = counts.get("LDSM", 0)
    out["ldg_count"] = counts.get("LDG", 0)
    out["cpasync_count"] = counts.get("LDGSTS", 0)
    out["inst_total"] = total
    out["sass_top_ops"] = dict(
        sorted(counts.items(), key=lambda kv: -kv[1])[:12]
    )
    return out


def res_usage(so_path: Path) -> dict:
    """cuobjdump -res-usage 로 교차 검증."""
    try:
        r = subprocess.run(
            [str(paths.cuda_bin("cuobjdump")), "-res-usage", str(so_path)],
            capture_output=True, text=True,
        )
    except Exception as e:  # pragma: no cover
        return {"res_usage_error": repr(e)}
    if r.returncode != 0:
        return {"res_usage_error": r.stderr[-300:]}
    txt = r.stdout
    out = {}
    m = re.search(r"REG:(\d+)", txt)
    if m:
        out["res_regs"] = int(m.group(1))
    m = re.search(r"SHARED:(\d+)", txt)
    if m:
        out["res_shared"] = int(m.group(1))
    m = re.search(r"LOCAL:(\d+)", txt)
    if m:
        out["res_local"] = int(m.group(1))
    return out


# ---------------------------------------------------------------------------
# 빌드
# ---------------------------------------------------------------------------
def build_kernel(cfg, backend, be: BuildEnv, force: bool = False) -> dict:
    kid = backend.kernel_id(cfg)
    src = be.src_dir / f"{kid}.cu"
    so = be.lib_dir / f"{kid}.so"

    src.write_text(backend.emit_cpp(cfg))

    # 캐싱은 kernels.jsonl 수준에서 한다 (이미 기록된 kernel_id 는 애초에
    # 여기까지 오지 않는다). .so 만 있고 기록이 없으면 ptxas 정보를 얻을 수
    # 없으므로 반쪽짜리 행을 남기지 말고 다시 컴파일한다.
    # ⛔ so_path 는 기록하지 않는다 (P-1). 빌드한 기계의 **절대 경로**라
    #    컨테이너 안이나 다른 저장소 위치에서는 존재하지 않는다.
    #    읽는 쪽은 kernel_id 에서 `paths.kernel_so()` 로 조립한다.
    row: dict = {"kernel_id": kid}
    cmd = [
        be.nvcc, "-std=c++17", f"-arch={be.arch_flag}", "-O3",
        "-shared", "-Xcompiler", "-fPIC",
        "-Xptxas", "-v", "-Xptxas", "-warn-spills",
        "--expt-relaxed-constexpr",
        *be.include_args,
        str(src), "-o", str(so),
    ]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    row["build_seconds"] = round(time.time() - t0, 2)

    if p.returncode != 0:
        row["build_status"] = "build_fail"
        row["build_error"] = _first_error(p.stderr)
        row["build_error_full"] = p.stderr[-3000:]
        return row

    row["build_status"] = "ok"
    entry = pick_gemm_entry(parse_ptxas(p.stderr))
    row["mangled_name"] = entry.get("name")
    row["regs_per_thread"] = entry.get("regs_per_thread")
    row["smem_static_bytes"] = entry.get("smem_static_bytes", 0)
    row["stack_frame"] = entry.get("stack_frame", 0)
    row["spill_stores"] = entry.get("spill_stores", 0)
    row["spill_loads"] = entry.get("spill_loads", 0)
    row["cmem_bytes"] = entry.get("cmem_bytes")
    row.update(res_usage(so))
    row.update(analyze_sass(so))
    # cudaFuncGetAttributes / occupancy — 유일하게 CUDA 컨텍스트가 필요한 부분.
    try:
        row.update(introspect(so))
    except Exception as e:
        row["info_error"] = repr(e)
    return row


_RE_STATIC_ASSERT = re.compile(r'static assertion failed with "([^"]*)"')


def _first_error(stderr: str) -> str:
    """실패 원인을 그룹핑 가능한 짧은 문자열로 정규화한다."""
    m = _RE_STATIC_ASSERT.search(stderr)
    if m:
        return f"static_assert: {m.group(1)}"
    for line in stderr.splitlines():
        if ": error:" in line:
            # 경로/줄번호를 떼어 같은 원인끼리 묶이게 한다.
            return "error: " + line.split(": error:", 1)[1].strip()[:160]
        if "Error" in line and "ptxas" in line:
            return line.strip()[:160]
    return stderr.strip().splitlines()[-1][:160] if stderr.strip() else "unknown"


# ---------------------------------------------------------------------------
# kt_info() — 유일하게 CUDA 컨텍스트가 필요한 부분
# ---------------------------------------------------------------------------
class KtInfo(ctypes.Structure):
    _fields_ = [
        ("kernel_id", ctypes.c_char * 128),
        ("num_regs", ctypes.c_int),
        ("smem_static", ctypes.c_size_t),
        ("smem_dynamic", ctypes.c_size_t),
        ("local_bytes", ctypes.c_size_t),
        ("const_bytes", ctypes.c_size_t),
        ("max_threads_per_block", ctypes.c_int),
        ("threads", ctypes.c_int),
        ("max_blocks_per_sm", ctypes.c_int),
        ("cutlass_max_blocks", ctypes.c_int),
    ]


def introspect(so_path: str | Path) -> dict:
    """dlopen -> kt_info() -> dlclose."""
    lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_LOCAL)
    try:
        info = KtInfo()
        lib.kt_info.argtypes = [ctypes.POINTER(KtInfo)]
        lib.kt_info.restype = ctypes.c_int
        rc = lib.kt_info(ctypes.byref(info))
        if rc != 0:
            return {"info_error": f"kt_info rc={rc}"}
        return {
            "num_regs": info.num_regs,
            "smem_static": int(info.smem_static),
            "smem_dynamic": int(info.smem_dynamic),
            "local_bytes": int(info.local_bytes),
            "const_bytes": int(info.const_bytes),
            "func_max_threads_per_block": info.max_threads_per_block,
            "threads": info.threads,
            "max_blocks_per_sm": info.max_blocks_per_sm,
            "cutlass_max_blocks": info.cutlass_max_blocks,
        }
    finally:
        _dlclose(lib._handle)


_libdl = None


def _dlclose(handle) -> None:
    """dlclose. argtypes 를 지정하지 않으면 핸들이 32비트로 잘려 segfault 난다."""
    global _libdl
    try:
        if _libdl is None:
            _libdl = ctypes.CDLL("libc.so.6")
            _libdl.dlclose.argtypes = [ctypes.c_void_p]
            _libdl.dlclose.restype = ctypes.c_int
        _libdl.dlclose(ctypes.c_void_p(handle))
    except (OSError, AttributeError):
        # dlclose 실패는 무해하다 — 이미 introspect 를 끝냈고 프로세스가
        # 곧 끝난다. (예전에 argtypes 를 빠뜨려 핸들이 32비트로 잘려
        # segfault 가 났었다. 그건 여기서 삼킬 문제가 아니라 argtypes 로
        # 고쳤다 — docs/decisions.md 참조.)
        pass


def build_ctx_so(env: dict, force: bool = False) -> Path:
    """libkt_ctx.so 를 한 번 빌드한다 (측정 프로토콜 + cuBLAS + 리덕션)."""
    be = BuildEnv.from_env_json(env)
    src = paths.REPO_ROOT / "measure" / "kt_ctx.cu"
    so = paths.ARTIFACT_DIR / "libkt_ctx.so"
    if so.exists() and not force and so.stat().st_mtime > src.stat().st_mtime:
        return so
    cmd = [
        be.nvcc, "-std=c++17", f"-arch={be.arch_flag}", "-O3",
        "-shared", "-Xcompiler", "-fPIC",
        f"-I{paths.REPO_ROOT / 'measure'}",
        str(src), "-o", str(so), "-lcublas",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"libkt_ctx.so 빌드 실패:\n{p.stderr[-4000:]}")
    return so
