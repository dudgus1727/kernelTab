# 측정 완료 직후 체크리스트

전수 측정이 끝난 뒤 **판단하지 말고 이 순서대로 따라가면 되도록** 정리한 것.
각 단계에 실패 시 대응을 함께 적었다.

완료 예상: **2026-08-17(월) 15:00~16:00 UTC** (진행률 기준 추정).

```bash
# 완료 확인
pgrep -f "rehearse.py --all" || echo "측정 종료됨"
tail -30 <phase3 로그>
```

---

## 0. 완료 여부 확인 — 정말 끝났는가

측정 프로세스가 사라졌다고 완료가 아니다. **자발적 중단**(요청 O) 일 수 있다.

```bash
grep -E "!! 중단|드리프트 .* 연속|sw_power_cap" <phase3 로그> | tail
python3 scripts/rehearse.py --all --dry-run | grep "남은 작업"
```

| 상황 | 대응 |
|---|---|
| `남은 작업: 0` | 정상 완료. 1 단계로. |
| `!! 중단: 드리프트 …` | **조건이 변했다.** 원인(클럭 해제? 다른 프로세스?)을 먼저 찾는다. 고친 뒤 같은 명령으로 재개하면 이어서 진행한다. |
| 남은 작업 > 0, 중단 메시지 없음 | 프로세스가 비정상 종료(OOM/커널 패닉/세션 종료). `dmesg \| tail`, `free -g` 확인 후 재개. |

**재개 명령** (조건이 그대로일 때만):
```bash
cd /home/piai/workspace/kernelTab
nohup python3 -u scripts/rehearse.py --all >> <phase3 로그> 2>&1 &
```
재개 시 20 초 워밍업이 다시 들어간다. 정상이다.

---

## 1. 백업 — 다른 무엇보다 먼저

40 시간짜리 데이터다. 이후 단계에서 무엇이 잘못되든 되돌릴 수 있어야 한다.

```bash
cd /home/piai/workspace
tar -czf ~/kerneltab-results-$(date +%Y%m%d-%H%M).tar.gz kernelTab/results
ls -lh ~/kerneltab-results-*.tar.gz
```

`build/artifacts/lib` (7.4 GB) 는 **백업하지 않아도 된다** — 재빌드 40 분이면
복원된다. `results/` 는 재현 불가능하다.

---

## 2. 클럭 해제 — 다른 사용자를 위해

측정이 끝났으면 즉시 푼다. 고정된 채로 두면 그 GPU 를 쓰는 사람이 원인 모를
성능 저하를 겪는다.

```bash
sudo nvidia-smi -i 3 -rgc
sudo nvidia-smi -i 3 -rmc
nvidia-smi -i 3 --query-gpu=clocks.sm,clocks.mem,pstate --format=csv
```

> **순서 주의**: 1(백업) 다음, 3(검증) **전**에 한다. 이후 단계는 GPU 를
> 쓰지 않으므로 클럭을 풀어도 결과가 달라지지 않는다.
>
> 단 **3 번에서 재측정이 필요하다고 판정되면 다시 고정해야 한다.** 그때는
> `phase0_env.py` 를 다시 돌리지 말고(→ `env_hash` 가 바뀐다) 클럭만 원래
> 값으로 되돌린 뒤 재개한다:
> ```bash
> sudo nvidia-smi -i 3 -pm 1
> sudo nvidia-smi -i 3 -lgc 1350,1350
> sudo nvidia-smi -i 3 -lmc 8001      # 부하 중 실측은 7601
> python3 scripts/verify_clock_lock.py --minutes 5   # 유지되는지 확인
> ```

---

## 3. 표 무결성 검사 — **통과해야 다음으로**

```bash
python3 scripts/validate_table.py --expect full
echo "exit=$?"
```

`exit=0` 이어야 한다. 실패 유형별 대응:

### 3-a. 누락 (`4. 누락`)

```bash
python3 scripts/validate_table.py --expect full --list-missing 200 \
    > /tmp/missing.txt
grep -A20 "형상별 누락 상위" /tmp/missing.txt
```

| 누락 비율 | 판단 | 대응 |
|---|---|---|
| 0 | 정상 | 다음으로 |
| < 0.1% | 경고만 (스크립트가 warn 처리) | 어떤 형상인지 확인. 특정 형상에 몰려 있지 않다면 진행 가능 |
| 0.1 ~ 5% | **재측정** | 그대로 `rehearse.py --all` 재실행하면 누락분만 잰다 (resume 이 이미 잰 것을 건너뛴다). 클럭을 먼저 다시 고정할 것 |
| > 5% | **원인 조사 우선** | 중단 지점이 있었는지, 특정 형상/커널에 몰려 있는지 확인. 열거기가 바뀌었을 가능성도 본다 |

**재측정 범위를 좁힐 필요는 없다.** resume 이 조합 단위로 동작하므로
`rehearse.py --all` 을 그냥 다시 돌리면 빠진 것만 잰다. 범위를 손으로
좁히려다 실수하는 편이 더 위험하다.

### 3-b. 중복 (`2. 중복 측정`)

같은 `(env_hash, kernel_id, 형상, 런타임)` 이 두 번 이상 있다. resume 로직
버그이거나 프로세스가 두 개 돌았다는 뜻이다.

```bash
# 프로세스가 둘이었는지 확인
grep -c "\[Phase 3 전수\]" <phase3 로그>     # 재개 횟수
```
데이터를 지우지 말 것 (append-only). `export.py` 단계에서 중복 제거 규칙
(같은 키면 **마지막 줄 채택**)을 정하고 문서화한다.

### 3-c. SOL 위반 (`6. SOL`)

물리적으로 불가능한 값이다. 목록을 뽑아 해당 (형상, 커널) 을 확인한다.

```bash
python3 scripts/validate_table.py --expect full --list-bad 50 | grep -A50 "SOL"
```
* 몇 건뿐이고 특정 커널에 몰려 있다 → 그 커널의 계산이 틀렸을 수 있다.
  `scripts/check_correctness.py` 로 그 커널만 확인.
* 광범위하다 → **실효 피크(`peak_tflops_f16_effective`) 추정이 틀렸다.**
  `env.json` 의 클럭 보정을 재검토한다. 데이터 문제가 아니다.

### 3-d. 커버리지 0 인 형상 (`5. 커버리지`)

그 형상은 표에서 쓸 수 없다. 원인 확인:

```bash
python3 - <<'EOF'
import json
from collections import Counter
tgt = (1024, 4096, 4097)     # 문제의 형상
c = Counter()
for l in open('results/results.jsonl'):
    d = json.loads(l)
    p = d.get('problem', {})
    if (p.get('M'), p.get('N'), p.get('K')) == tgt:
        c[d.get('status')] += 1
print(c)
EOF
```
* 전부 `numerical_fail` / `runtime_fail` → 그 형상은 **측정 불가**가 결론이다.
  리포트에 명시하고 표에서 제외한다.
* 측정 자체가 0 건 → 열거 문제. `alignments_for` 와 빌드된 alignment 조합을
  확인 (`a118` 은 445 개뿐이라는 점을 기억할 것).

### 3-e. `max_rel_error` 이상 (`7.`)

`parallel` 인데 중앙값이 크거나 상한을 넘는 그룹이 있으면 **리덕션 경로에
문제가 있다.** `scripts/smoke_splitk.py` 로 재확인한다 (GPU 필요 → 클럭을
다시 고정한 뒤).

---

## 4. parquet 생성

```bash
python3 scripts/export.py
python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('results/table.parquet')
print(t.num_rows, '행', t.num_columns, '열')"
```

`export.py` 는 `smem_computed` 를 **현재 공식으로 재계산**한다
(`kernels.jsonl` 의 값은 빌드 시점 것이다). 재계산 결과가 `smem_dynamic` 과
다르면 `smem_matches=False` 로 표시된다 — 0 이어야 한다.

## 5. 리포트

```bash
python3 scripts/report_phase3.py
ls results/report/
```

`results/report/report_<hash>.md` 와 PNG 8 종이 나온다. **요청 Q 의 7 개
항목**이 여기 들어 있다. 검토 후 사용자에게 보고한다.

특히 볼 것:
* `4. split-K 가설` 판정 — 미지지면 그것이 결론이다
* `5. waves > 40 에서 split_k=16` — 표본 편향이었는지 실제 현상인지
* `6. 축별 성능 분포` 의 각 축 판정 — "선택 축이 아니다" 로 나온 축은
  Phase 4 이후 열거에서 줄일 수 있다

## 6. 소비 인터페이스 확인

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from core.table import load_for_ranking, load_for_scoring, assert_no_answers
import json
env = json.load(open('results/env.json'))['env_hash'][:8]
X = load_for_ranking('results/table.parquet', env_hash=env)
y = load_for_scoring('results/table.parquet', env_hash=env)
assert_no_answers(X)
print(f'X {X.shape}  y {y.shape}  누출 없음')"
```

---

## 7. 마이그레이션 (`docs/migration_plan.md`)

여기서부터는 **코드를 고친다.** 3~6 이 전부 통과한 뒤에만 시작한다.

```
P-3 (env_hash 재정의 + env_registry)  →  P-1 (so_path)  →  P-2 (GPU 선택)
  →  나머지 8 건
```

각 단계 사이에 커밋하고, 단계마다 `validate_table.py` 를 다시 돌린다.
**마이그레이션이 데이터 해석을 바꾸면 안 된다.**

## 8. 패키지화 (`docs/packaging.md`)

P-1 이 끝난 뒤에만. 디렉토리 이동이 포함되므로 마지막에 한다.

## 9. 컨테이너 (`docker/Dockerfile.draft`)

마이그레이션 1~4 번이 끝나면 초안의 "빌드하기 전에 반드시 할 일" 이 전부
충족된다. 그때 베이스 다이제스트를 채우고 빌드·검증한다.

---

## 요약 — 한 화면

```bash
# 0) 완료 확인
pgrep -f "rehearse.py --all" || echo done
python3 scripts/rehearse.py --all --dry-run | grep "남은 작업"

# 1) 백업
tar -C .. -czf ~/kerneltab-results-$(date +%Y%m%d-%H%M).tar.gz kernelTab/results

# 2) 클럭 해제
sudo nvidia-smi -i 3 -rgc && sudo nvidia-smi -i 3 -rmc

# 3) 검증 (실패하면 여기서 멈춘다)
python3 scripts/validate_table.py --expect full || echo "STOP"

# 4~6) 산출물
python3 scripts/export.py
python3 scripts/report_phase3.py
python3 -c "import sys;sys.path.insert(0,'.');from core.table import *;print('ok')"

# 7~9) 코드 변경은 그 다음
```
