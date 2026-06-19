# SOURCE-MAP

本仓库每章提炼自个人自学笔记的哪些 § 节、对应哪些 MiniMind 源码文件、复用哪张图。

用途：自学笔记仍在更新。源笔记某 § 改动后，按本表反查受影响的教学章，按需重新提炼。**§ 号指自学笔记 `notes/minimind_study_notes.md`（MiniMind2 主线）；mm3 §x 指 `minimind-3/notes/minimind3_study_notes.md`。**

## 映射表

| 章节文件 | 来源 §节 | 源码文件 | 复用图 |
|---|---|---|---|
| 00-overview/01-what-is-minimind | §1(事实部分), §2 | model/ dataset/ trainer/ eval_llm.py | project-source-mainline-map.svg |
| 00-overview/02-quickstart | §41–48(提炼) | requirements.txt, eval_llm.py | — |
| 01-foundations/01-tokenizer | §12.4, mm3 README §Ⅰ | trainer/train_tokenizer.py, model/tokenizer*.json | — |
| 01-foundations/02-embedding | §11.6 | model/model_minimind.py (MiniMindModel L556–620, lm_head L632–639) | — |
| 01-foundations/03-data-format | §12, mm3 README §Ⅱ–Ⅳ | dataset/lm_dataset.py | — |
| 02-model/01-block-and-rmsnorm | §21, §22 | model_minimind.py (MiniMindBlock L517–553, RMSNorm L96–121) | minimind-block-flow.svg |
| 02-model/02-attention | §13, §51 | model_minimind.py (Attention L224–316) | attention-forward-flow.svg |
| 02-model/03-rope | §14, §59 | model_minimind.py (precompute_freqs_cis L124–178, apply_rotary_pos_emb L181–211) | ⚠️ 无（图待补） |
| 02-model/04-gqa | §15 | model_minimind.py (repeat_kv L214–221) | gqa-diagram.svg |
| 02-model/05-swiglu | §23 | model_minimind.py (FeedForward L319–346) | ffn-swiglu-flow.svg |
| 02-model/06-moe | §24, §62→71.x | model_minimind.py (MoEGate L349–424, MOEFeedForward L427–514) | moe-router-flow.svg, moe-router-aux-loss-source-chain.svg |
| 03-pretrain/01-data-and-labels | §12, §11.7–11.9 | dataset/lm_dataset.py (PretrainDataset L31–49) | — |
| 03-pretrain/02-forward-to-loss | §11, §50 | model_minimind.py (MiniMindForCausalLM.forward L641–672) | causal-lm-forward-flow.svg |
| 03-pretrain/03-training-loop | §10 | trainer/train_pretrain.py | — |
| 04-inference/01-kv-cache-and-generate | §25 | model_minimind.py (Attention/Block use_cache) | kv-cache-generate-flow.svg |
| 04-inference/02-eval-and-service | §36 | eval_llm.py, scripts/web_demo.py, scripts/serve_openai_api.py | inference-service-flow.svg |
| 04-inference/03-weight-formats | §37 | scripts/convert_model.py | weight-format-conversion-flow.svg |
| 05-sft/01-assistant-only-supervision | §26, §52 | dataset/lm_dataset.py (SFTDataset.generate_labels L74–90), trainer/train_full_sft.py | sft-label-mask-flow.svg, sft-generate-labels-flow.svg |
| 06-dpo/01-preference-optimization | §27 | trainer/train_dpo.py, lm_dataset.py (DPODataset L108–178) | dpo-preference-flow.svg |
| 06-dpo/02-dpo-loss-and-math | §53, §58 | trainer/train_dpo.py (dpo_loss) | dpo-loss-source-chain.svg |
| 07-ppo-grpo/01-rl-overview | §28 | trainer/ | rlhf-ppo-grpo-spo-overview.svg |
| 07-ppo-grpo/02-ppo | §29, §57 | trainer/train_ppo.py | ppo-ratio-clip-flow.svg |
| 07-ppo-grpo/03-grpo | §30 | trainer/train_grpo.py | grpo-group-relative-flow.svg |
| 07-ppo-grpo/04-spo | §31 | trainer/train_spo.py | spo-adaptive-baseline-flow.svg |
| 07-ppo-grpo/05-training-signal-and-unified-source | §32, §54 | trainer/ | training-signal-map.svg, rl-train-step-unified-flow.svg |
| 08-training-mechanics/01-update-skeleton | §62→62.1–62.20 | trainer/*.py, trainer_utils.py | loss-backward-step-flow.svg |
| 08-training-mechanics/02-logits-to-logprob | §62→63.1–63.19 | model_minimind.py, trainer/*.py | logits-logprob-gather-flow.svg |
| 08-training-mechanics/03-token-to-sequence-objective | §62→64.1–64.19 | trainer/train_dpo/ppo/grpo.py | mask-sum-mean-aggregation-flow.svg |
| 08-training-mechanics/04-full-training-math-chain | §62→65.x | — | logits-to-update-full-chain.svg, training-math-script-review-map.svg |
| 08-training-mechanics/05-optimizer-adamw-scheduler | §62→66.x | trainer/*.py | adamw-lr-weight-decay-flow.svg |
| 08-training-mechanics/06-clipping | §62→67.x | trainer/*.py | gradient-clip-vs-ppo-clip.svg |
| 08-training-mechanics/07-stability-and-hyperparams | §62→68.x, 69.x | trainer/*.py | training-stability-control-map.svg, training-hyperparameter-reading-map.svg |
| 09-minimind2-vs-3/01-overview-five-changes | §63.1, mm3 §2/§4 | — | — |
| 09-minimind2-vs-3/02-architecture-diffs | §63.2 | model_minimind.py (v2 vs v3) | — |
| 09-minimind2-vs-3/03-ppo-rewrite | §63.3 | trainer/train_ppo.py (v2 vs v3) | — |
| 09-minimind2-vs-3/04-grpo-cispo | §63.4 | trainer/train_grpo.py (v2 vs v3) | — |
| 09-minimind2-vs-3/05-thinking-scale-removals | §63.5–63.7 | trainer/ (v2 vs v3), tokenizer_config.json | — |
| 10-experiments/01-fixed-prompt-design | §34, §35, §38 | eval_llm.py | — |
| 10-experiments/02-server-training-records | mm3 §5.1–5.4, §5.6 | — | swanlab/*.png |
| 10-experiments/03-eval-conclusions-sft-vs-rl | mm3 §5.5 | — | swanlab/*.png |
| appendix/02-advanced-pointers | §63.7(Agent RL) | model/model_lora.py, trainer/train_lora.py, train_distillation.py | — |

## 删除 / 不收（个人痕迹或 v2+ backlog）

- 删：§1(大部分)、§3.3–3.4、§4、§6(个人 checklist)、§7、§8、§9、§33、§39、§40–49(留环境结论入 00)、§55、§56、§60、§61；各节内 `自测题/参考答案`(转章末练习)、`暂不展开`、`当前进度/下一步`、`#### YYYY-MM-DD` 日志。
- 不收（v2+ backlog）：GRPO 变体家族、Flash Attention 升级为正式章、归一化 survey、面试八股、Part 分层。
- 不收（工程非主线）：§16–20（trainer_utils lr 调度 / checkpoint / 分布式、SkipBatchSampler、setup_seed、from_weight、AMP/GradScaler/compile）——其中影响主线理解的最小点（from_weight 起训、梯度累积、为何用 GradScaler）已在 03-pretrain/03-training-loop 正文随主流程点到，不单开工程篇。

## 折叠去向

- §62→70.x（阶段复盘）→ 各章小结素材，不单列。
- §62→71.x（MoE）→ 并入 02-model/06-moe。
- §63.8（受影响 v2 小节速查）→ 09-minimind2-vs-3/01 的交叉引用表。
