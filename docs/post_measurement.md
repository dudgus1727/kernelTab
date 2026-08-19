# 측정 완료 직후 체크리스트

전수 측정이 끝난 뒤 **판단하지 말고 이 순서대로 따라가면 되도록** 정리한 것.
각 단계에 실패 시 대응을 함께 적었다.

완료 예상: **2026-08-17(월) 15:00~16:00 UTC** (진행률 기준 추정).

```bash
# 완료 확인 — pgrep 을 쓰지 마라. 명령줄에 "rehearse.py --all" 이 들어간
# 감시 셸 자신에 매칭되어 "아직 도는 중" 으로 오판한다 (13 시간을 그렇게
# 날렸다). heartbeat.json + /proc 를 보는 watch.py 를 쓴다.
python3 scripts/watch.py; echo "exit=$?"     # 0=진행중 3=끝남 5=죽음
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

> **보존 순서 (C-5).** 아래가 전체 순서다. 이 문서의 절 번호와 대응한다.
>
> | # | 할 일 | 절 |
> |---|---|---|
> | 1 | `results/` 원본 tar 백업 | 1 |
> | 2 | `validate_table.py --expect full` 통과 확인 | 3 |
> | 3 | `export.py` → `table.parquet` | 4 |
> | 4 | `bundle.py --archive --archive-raw` → 번들 (검증 실패 시 **거부됨**) | 4-b |
> | 5 | **다른 물리 디스크**로 복사 | 4-b |
> | 6 | 오프사이트 1 부 | 4-b |
> | 7 | 체크섬 대조 | 4-b |
>
> **5~6 번(다른 물리 디스크 + 오프사이트)을 마치기 전에는 `results/` 안의
> 무엇도 수정하지 마라.** 같은 디스크 안의 tar 는 백업이 아니다 — 디스크가
> 죽으면 원본과 사본이 같이 죽는다. 1 번은 "실수로 지웠을 때" 용이고,
> 5~6 번이 "디스크가 죽었을 때" 용이다. 둘은 대체재가 아니다.

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

## 4-b. 번들 + 오프사이트 — **여기까지 끝나야 수정 허용**

```bash
python3 scripts/bundle.py --archive --archive-raw
ls -lh datasets/*/
```

`bundle.py` 는 안에서 `validate_table.py --expect full` 을 먼저 돌리고
**통과하지 못하면 번들을 만들지 않는다.** 검증 안 된 데이터가 배포되면 안
되기 때문이다. `--archive-raw` 는 `results.jsonl` 원본을
`results-raw-{env_hash8}.jsonl.zst` 로 따로 압축한다 — `table.parquet` 은
파생물이라 파생 지표 계산식이 바뀌면 원본에서 다시 만들어야 한다.

그 다음 **물리적으로 다른 곳에 두 부**를 만든다.

```bash
BID=$(ls datasets | head -1)

# 5. 다른 물리 디스크 (같은 디스크 안의 사본은 백업이 아니다)
lsblk -o NAME,SIZE,MOUNTPOINT           # 마운트된 다른 디스크 확인
cp -a datasets/$BID  /mnt/<other-disk>/kerneltab/
cp -a datasets/$BID.tar.zst* ~/kerneltab-results-*.tar.gz  /mnt/<other-disk>/kerneltab/

# 6. 오프사이트 1 부 (rclone / scp / GitHub Release 중 하나)
rclone copy datasets/$BID.tar.zst  remote:kerneltab/

# 7. 체크섬 대조 — 복사가 조용히 깨졌을 수 있다
sha256sum -c datasets/$BID.tar.zst.sha256
(cd /mnt/<other-disk>/kerneltab && sha256sum -c $BID.tar.zst.sha256)
python3 -c "
import sys; sys.path.insert(0,'.')
from core.bundle import load_bundle
b = load_bundle('/mnt/<other-disk>/kerneltab/$BID')   # verify=True 가 기본
print('사본 무결성 OK:', b.bundle_id)"
```

> **5~6 번을 마치기 전에는 `results/` 안의 무엇도 수정하지 마라.**
> 7 절(마이그레이션) 이후의 코드 수정은 파생 지표 계산식을 바꿀 수 있고,
> `results.jsonl` 은 append-only 라 **되돌릴 방법이 원본 사본뿐**이다.

## 4-c. 백업이 끝났으면 — 밀린 수정

백업 1~6 단계(위 4-b 까지)를 마쳤으면 `docs/pending_fixes.md` 의 R-1~R-5 를
그 순서대로 진행한다. **그 전에는 시작하지 않는다.**

가장 먼저 R-1(테스트가 조용히 스킵되는 문제)을 한다 — 이후 모든 수정의
회귀 테스트가 실제로 도는지 먼저 보장해야 하기 때문이다.

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

## 10. 다른 GPU 에서 재측정할 때 — **드리프트 절차를 먼저 돌려라**

> 이 절은 "측정이 끝난 뒤" 가 아니라 **시작 전** 체크리스트다. README 의
> "새 GPU 에서는 드리프트 3 값을 먼저 재라" 와 같은 내용이며, 여기서는
> 절차를 자세히 적는다.

A6000 에서 다중 시간 측정 드리프트를 발견했다 (7.5 시간에 +5.06%).
`docs/measurement_drift.md` 에 원인과 대책이 있다. **이 절차는 GPU 마다
다시 해야 한다** — 드리프트의 크기도, 대책의 주기도 하드웨어에 의존한다.

새 GPU(4090, H100 ...)에서 Phase 3 를 켜기 **전에**:

1. **클럭 고정 검증** — `-lgc` 는 SM 만 잠근다. 메모리 클럭도 따로 잠그고
   부하 중 실측이 요청값과 같은지 확인한다 (A6000 은 8001 요청 → P2 에서
   7601 로 캡). `verify_clock_lock.py --minutes 5`
2. **실효 피크 재측정** — 고정된 클럭에서의 실효 TFLOP/s 와 대역폭.
   ridge point 가 여기서 나오고, 틀리면 `is_memory_bound` 가 전부 틀린다.
3. **드리프트 3 값 측정** — 커널을 하나씩 "터치" 하면서 기준 커널 여러 개를
   여러 형상에서 재고 `t = a + b·work` 를 맞춘다. 얻을 값:
   * **문턱** — 몇 개까지는 `a` 가 전혀 안 늘어나는가 (A6000: ~1,200)
   * **상수 성분** — 문턱 위에서 모듈 1,000 개당 `a` 증가량 (A6000: **75 us**).
     전 구간 평균으로 재면 42 us 가 나오는데, 문턱 아래 평탄 구간이 섞인
     값이다. 세그먼트 크기를 정하는 데 쓰는 것은 문턱 위 기울기다.
   * **왜곡 배율** — 가장 짧은 커널 대 가장 긴 커널의 드리프트 비
     (A6000: 100 배)

   ⚠️ **큰 형상 하나로 재면 안 된다.** 드리프트는 런치당 상수라 긴 커널에서는
   안 보인다. A6000 에서 4096³ 만 보고 +5.06 % 로 판정했는데 512³ 은
   +1380 % 오염돼 있었다.
4. **세그먼트 크기 결정** — `segments.kernels` 를 문턱의 절반 이하로.
   재시작 오버헤드가 5 % 를 넘으면 문턱에 더 가깝게 올린다 (E-2).
   절차 전체는 `docs/new_gpu_checklist.md` 에 게이트 형태로 정리돼 있다.

5. **2~3 시간 짧은 검증** — `sweep.py --max-rounds N` 으로 돌리고
   `check_anchors.py` 로 판정한다. 통과 기준:
   * 짧은 앵커의 세그먼트 간 변동폭 ≤ 1 %
   * 라운드에 따른 단조 증가 없음 (있으면 세그먼트 밖에 다른 누적이 있다 —
     프로세스 재시작이 완전히 리셋하지 못하는 것이므로 드라이버 수준 상태를
     의심해야 한다)
   * 세그먼트 시작/끝 사이 이동 ≤ 1 %

   통과한 뒤에야 전체 스윕을 켠다.

건너뛰면 어떻게 되는지는 이미 안다: A6000 에서 226,100 행(23 %)을 버렸다.

---

## 요약 — 한 화면

```bash
# 0) 완료 확인 — pgrep 을 쓰지 마라 (자기 자신에 매칭된다, D-4)
python3 scripts/watch.py; echo "exit=$?"     # 0=진행중 3=끝남 5=죽음

# 1) 백업
tar -C .. -czf ~/kerneltab-results-$(date +%Y%m%d-%H%M).tar.gz kernelTab/results

# 2) 클럭 해제
sudo nvidia-smi -i 3 -rgc && sudo nvidia-smi -i 3 -rmc

# 3) 검증 (실패하면 여기서 멈춘다)
python3 scripts/validate_table.py --expect full || echo "STOP"

# 4) 산출물 + 번들
python3 scripts/export.py
python3 scripts/bundle.py --archive --archive-raw

# 5~7) 다른 물리 디스크 / 오프사이트 / 체크섬 — 여기까지 끝나야 수정 허용
cp -a datasets/* /mnt/<other-disk>/kerneltab/
rclone copy datasets/*.tar.zst remote:kerneltab/
sha256sum -c datasets/*.tar.zst.sha256

# 리포트 · 소비 확인
python3 scripts/report_phase3.py
python3 -c "import sys;sys.path.insert(0,'.');from core.table import *;print('ok')"

# 7~9) 코드 변경은 그 다음
```
