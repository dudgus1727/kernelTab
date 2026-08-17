# 테스트

**GPU 를 전혀 쓰지 않는다.** 측정이 도는 중에도 안전하게 돌릴 수 있다.
CI 는 아직 없다 (측정 중 리소스 소비를 피하기 위해).

```bash
pip install -e '.[test]'      # 또는: pip install pytest pyarrow pandas
python3 -m pytest tests/ -q
```

## 이 테스트들이 지키는 것

### `tests/test_config.py` — alignment 유도

| 테스트 | 지키는 것 |
|---|---|
| `test_layer_d_k_variants` | K=4100/4098/4097 이 각각 (4,4,8)/(2,2,8)/(1,1,8) 을 낸다 |
| `test_layer_d_n_variants` | C 는 row-major 이므로 N 이 연속 차원이다 |
| `test_layout_changes_which_dim_binds` | 레이아웃을 바꾸면 걸리는 차원이 바뀐다 — 이 함수의 핵심 |
| `test_max_is_8_not_more` | fp16 은 128비트 = 8원소가 상한 |
| `test_unknown_raises` | 모르는 dtype 에서 조용히 기본값을 쓰지 않는다 |

alignment 가 틀리면 커널이 잘못된 벡터 폭으로 읽어 **조용히 틀린 결과**가 나온다.

### `tests/test_features.py` — 파생 물리 피처

| 테스트 | 지키는 것 |
|---|---|
| `test_waves_hand_computed` | 손계산 64/84 = 0.7619 |
| `test_tail_waste_hand_computed` | 0.2381 |
| `test_mainloop_iters_hand_computed` | K=4096, tile_k=64, sk=6 → 11 |
| `test_square_converges_to_n_over_3` | 정방형 AI = n/3 |
| `test_ridge_point_uses_effective_peak` | **스펙(201.5)이 아니라 실효(159.1)** |
| `test_only_k4097_is_blocked` | cp.async 가용성이 형상에서 결정된다 |
| `test_waves_uses_hw_sm_count` | sm_count 하드코딩 검출 |
| **`test_all_feature_fns_work_with_ext_none`** | **★ 피처가 `cfg.ext` 를 참조하지 않는다** |
| **`test_features_module_does_not_import_backends`** | **★ core → backends 역의존 없음** |

마지막 두 개가 **아키텍처 전이의 전제**다. 피처가 `ext` 를 보기 시작하면
SM80 에서 배운 것을 SM90 에 적용할 수 없다.

### `tests/test_shapes.py` — 형상 그리드

| 테스트 | 지키는 것 |
|---|---|
| `test_documented_counts` | 층별 40/12/11/5/5, 합계 73, 고유 66 |
| **`test_different_sm_count_gives_different_m`** | **★ 층 C 가 sm_count 에서 M 을 역산한다** |
| `test_m_scales_with_sm_count` | SM 이 많을수록 같은 waves 에 더 큰 M 이 필요 |
| `test_a_b_d_e_are_constant` | 층 A/B/D/E 는 hw 와 무관 (시그니처로도 강제) |
| `test_covers_all_low_alignments` | 층 D 가 alignment 5종을 모두 덮는다 |

층 C 가 hw 에 반응하지 않으면 GPU 간 비교가 성립하지 않는다.

### `tests/test_backend_sm80.py` — 백엔드 제약과 공식

제약 **하나마다** 통과/탈락 사례를 고정해 두었다. 제약을 건드리면 즉시 드러난다.

| 테스트 | 지키는 것 |
|---|---|
| `test_*` (제약 9종) | tile 나눔 / warp 개수 / instruction shape / warp_k / smem / 누산 레지스터 / horizontal swizzle / cp.async / **warp_n=128 오답** |
| `test_align_c_changes_fragments` | `kFragmentsPerIteration=2` 는 `align_c==8` 에서만 (120개 과대평가 버그 회귀) |
| `test_mainloop_rule`, `test_epilogue_rule_*` | thread map 예측식 |
| `test_expected_hmma` | `(warp_m/16)·(warp_n/8)·(warp_k/16)` |
| `test_effective_split_k_matches_cutlass_formula` | K=4096 에서 sk=3→3, 6→6, **12→11** |
| `test_alignment_must_match_shape` | a888 커널이 K=4100 에 쓰이지 않는다 |
| `test_split_k_axis_keeps_3_6_12` | 84 = 2²·3·7 가설 검증용 축을 지운 적 없음 |
| **`test_enumeration_reacts_to_hw`** | **★ 84/101376 하드코딩 검출** |

### `tests/test_table.py` — ★ 정답 누출 방지

`docs/consumer_contract.md` 는 "문서는 지켜지지 않으므로 코드로 강제한다" 고
썼다. 그렇다면 **그 코드를 지키는 것도 문서여서는 안 된다.** 이 파일이 그
역할이다. 합성 parquet 으로만 돌아 실제 측정 데이터에 의존하지 않는다.

| 테스트 | 지키는 것 |
|---|---|
| `test_ranking_has_no_answer_cols` | `load_for_ranking` 에 정답이 없다 |
| `test_derived_answers_are_removed_too` | **`time_ms` 만 빼는 사고를 막는다** (`tflops`/`vs_cublas` 도 정답) |
| `test_scoring_has_answers_but_no_features` | 채점용은 피처를 주지 않는다 |
| `test_catches_manual_readparquet` | 로더 우회를 `assert_no_answers` 가 잡는다 |
| `test_mixed_conditions_raise_without_env_hash` | 조건이 섞이면 예외 |
| **`test_unknown_column_warns` / `_is_dropped` / `_can_escalate`** | **★ export.py 가 새 컬럼을 추가했는데 분류를 안 하면 경고하고, 그래도 노출하지는 않는다** |
| `test_partitions_are_disjoint` | 컬럼이 두 분류에 동시에 있지 않다 |

마지막 장치가 이 구조의 약점을 막는다. **제거 목록만 관리하면 새 컬럼이
조용히 샌다.** 허용 목록(`KNOWN_FEATURE_COLS`)을 함께 두고 어느 쪽에도 없는
컬럼은 미분류로 보아 경고 + 제외한다.

## 여기서 검증하지 않는 것

GPU 가 필요한 것은 전부 별도 스크립트다 (`docs/entrypoints.md` 참조).

| 무엇 | 어디서 |
|---|---|
| `smem_bytes` vs 실제 `sizeof(SharedStorage)` | `scripts/check_smem.py` |
| 제약 예측 vs 실제 빌드 결과 | `scripts/validate_constraints.py` |
| 커널 계산 정확성 | `scripts/check_correctness.py` |
| 직접 구현한 swizzle | `scripts/verify_swizzle.py` |
| split-K 경로 | `scripts/smoke_splitk.py` |
| 클럭 고정 유지 | `scripts/verify_clock_lock.py` |
| 표 무결성 | `scripts/validate_table.py` |

즉 **단위 테스트는 "공식과 규약"을, 스크립트는 "GPU 에서 정말 그런가"를**
검증한다. 둘 다 필요하다 — 공식이 맞아도 CUTLASS 가 다르게 동작할 수 있고,
GPU 검증이 통과해도 리팩터링이 공식을 깨뜨릴 수 있다.

## 마이그레이션 회귀 테스트로 쓰기

`docs/migration_plan.md` 의 수정 11건을 적용할 때 **각 단계마다** 돌린다.

```bash
python3 -m pytest tests/ -q && python3 scripts/validate_table.py --expect full
```

특히 P-3(`env_hash` 재정의)는 `core/features.py` 를 건드리지 않지만
`phase0_env.py` 와 `rehearse.py` 를 바꾼다. 테스트가 통과하면 최소한
피처 계산은 그대로라는 뜻이다.
