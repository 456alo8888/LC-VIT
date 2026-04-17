# End-to-End LC-VIT Finetuning Implementation Plan

## Overview

Implement an end-to-end training pipeline in main_finetune.py that directly trains image encoder pathways and regression head together from raw 3-view PNG inputs plus tabular features, replacing the current staged dependency on precomputed CSV image embeddings.

## Current State Analysis

The current experiment flow is staged and file-based:
- Manifest construction from SOOP view folders and tabular CSV in code/baseline_encoder/LC-VIT/experiment/build_regression_manifest.py:68.
- Feature extraction with TCFormer in inference mode and no_grad into per-view CSV files in code/baseline_encoder/LC-VIT/experiment/extract_features.py:182 and code/baseline_encoder/LC-VIT/experiment/extract_features.py:254.
- Feature merge into merged_features.csv in code/baseline_encoder/LC-VIT/experiment/merge_features.py:45 and code/baseline_encoder/LC-VIT/experiment/merge_features.py:61.
- Regression head training from merged numeric features in code/baseline_encoder/LC-VIT/experiment/train_regression.py:202.

The existing finetuning entry file is currently extraction-oriented and mirrors extract_features behavior:
- CLI and extraction options in code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py:25.
- TCFormer encoder loaded and set to eval in code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py:64 and code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py:108.
- no_grad inference loop in code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py:187.
- CSV feature writing in code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py:253.

## Desired End State

A single training entrypoint at code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py does all of the following:
- Reads manifest and tabular metadata directly.
- Loads and preprocesses raw image files on the fly for Axial, Coronal, Sagittal.
- Builds three-view encoder branches and a regression model path in one trainable graph.
- Supports freeze and unfreeze policies for backbone finetuning.
- Trains, validates, checkpoints, and evaluates without generating intermediate feature CSVs.
- Saves metrics, predictions, and run manifest artifacts in a dedicated finetuning run directory.

Verification of desired end state:
- One command can train and evaluate from manifest inputs and checkpoint path settings.
- Checkpoint contains encoder weights and regression head weights.
- Predictions and metrics are generated per split similarly to current train_regression outputs.

### Key Discoveries:
- Current training code consumes merged feature columns, not image tensors, in code/baseline_encoder/LC-VIT/experiment/dataset.py:94 and code/baseline_encoder/LC-VIT/experiment/train_regression.py:225.
- Image preprocessing (crop, resize, normalization) already exists and is reusable in code/baseline_encoder/LC-VIT/experiment/extract_features.py:112 and code/baseline_encoder/LC-VIT/experiment/extract_features.py:123.
- Fusion head architecture is already defined and can be reused after image embedding extraction in code/baseline_encoder/LC-VIT/experiment/model.py:72.
- Current eval_regression expects merged-feature dataset semantics and must be adapted or wrapped for end-to-end checkpoints in code/baseline_encoder/LC-VIT/experiment/eval_regression.py:33.

## What We Are NOT Doing

- Replacing or redesigning the current staged pipeline scripts in experiment root.
- Changing manifest generation schema in build_regression_manifest.py.
- Changing target definitions or split protocol names.
- Introducing distributed training frameworks or Lightning migration.
- Modifying external TCFormer implementation internals.

## Implementation Approach

Create a finetuning-local training stack in code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py that:
- Reuses manifest conventions and split semantics from the existing experiment pipeline.
- Reuses image preprocessing logic currently used for feature extraction.
- Reuses regression fusion head design from model.py while replacing precomputed image feature tensors with live encoder forward outputs.
- Preserves output artifact patterns (metrics JSON, predictions CSV, checkpoint, run manifest) for parity with existing scripts.

## Phase 1: Build End-to-End Data Pipeline

### Overview

Add a finetuning dataset path that reads raw image files and tabular columns directly from manifest/all_subjects inputs and returns train-ready tensors.

### Changes Required:

#### 1. Introduce finetuning dataset utilities inside main_finetune.py
File: code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py
Changes:
- Add a dataset bundle loader that consumes manifest JSON and all_subjects.csv directly.
- Add split_dataframe function for train/valid/test filtering.
- Add tabular normalization based only on train split.
- Add EndToEndLCVITDataset class returning participant_id, clinical, axial_img, coronal_img, sagittal_img, target.
- Reuse foreground crop and transform policy from extract_features.

```python
class EndToEndLCVITDataset(Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        axial = self.load_view(row["axial_path"])
        coronal = self.load_view(row["coronal_path"])
        sagittal = self.load_view(row["sagittal_path"])
        clinical = self.get_tabular(row)
        target = torch.tensor(row[self.target_col], dtype=torch.float32)
        return {
            "participant_id": str(row["participant_id"]),
            "clinical": clinical,
            "axial_img": axial,
            "coronal_img": coronal,
            "sagittal_img": sagittal,
            "target": target,
        }
```

### Success Criteria:

#### Automated Verification:
- [x] Data loader construction works for all splits using manifest input: python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py --help
- [x] Dataset can iterate one batch without runtime errors using limit argument: python code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py --manifest-dir code/baseline_encoder/LC-VIT/experiment/artifacts/manifest_fixed_split --target-col gs_rankin_6isdeath --max-epochs 1 --limit 8 --dry-run
- [x] Static syntax check passes: python -m compileall code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py

#### Manual Verification:
- [ ] Confirm one sampled subject shows all three image views aligned to expected participant_id.
- [ ] Confirm tabular normalization uses train split stats only.
- [ ] Confirm missing/invalid image paths fail with clear error trace.

Implementation Note: After completing this phase and automated checks, pause for manual confirmation before Phase 2.

---

## Phase 2: Build Trainable End-to-End Model Assembly

### Overview

Create a model graph that produces image embeddings from trainable encoders and feeds them with clinical features into existing regression modes.

### Changes Required:

#### 1. Add finetuning model builder in main_finetune.py
File: code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py
Changes:
- Add TCFormer encoder builder using timm and checkpoint loading logic currently in extract_features.
- Create ThreeViewEncoder wrapper to run axial/coronal/sagittal through shared or separate backbones.
- Reuse build_regression_model from code/baseline_encoder/LC-VIT/experiment/model.py:135.
- Add freeze policy flags: freeze_backbone, unfreeze_after_epoch, train_backbone_lr.

```python
class EndToEndRegressor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, clinical, axial_img, coronal_img, sagittal_img):
        axial_feat = self.backbone(axial_img)
        coronal_feat = self.backbone(coronal_img)
        sagittal_feat = self.backbone(sagittal_img)
        return self.head(clinical, axial_feat, coronal_feat, sagittal_feat)
```

#### 2. Add optimizer parameter-group policy
File: code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py
Changes:
- Separate encoder and head parameter groups with independent learning rates.
- Respect requires_grad freeze toggles for backbone.

### Success Criteria:

#### Automated Verification:
- [x] End-to-end forward pass returns shape batch_size x 1 for all model modes.
- [x] Checkpoint loading succeeds with provided TCFormer checkpoint path.
- [x] Frozen mode prevents backbone gradient updates during first epoch.

#### Manual Verification:
- [ ] Confirm GPU memory behavior is acceptable with selected batch size.
- [ ] Confirm encoder unfreeze schedule starts at configured epoch.
- [ ] Confirm model mode switching (fusion/image_only/clinical_only) behaves as intended.

Implementation Note: After completing this phase and automated checks, pause for manual confirmation before Phase 3.

---

## Phase 3: Integrate Training, Validation, Checkpointing, and Evaluation

### Overview

Integrate the end-to-end dataloaders and model into a complete training loop with artifact outputs equivalent to current train_regression behavior.

### Changes Required:

#### 1. Replace extraction-oriented main with train/eval runner
File: code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py
Changes:
- Replace current feature CSV export loop with epoch training loop.
- Reuse metric computation from code/baseline_encoder/LC-VIT/experiment/metrics.py:8.
- Save best checkpoint, val/test metrics, and predictions CSV in finetuning run directory.
- Save run manifest fields including backbone config, freeze policy, preprocessing settings, split counts.

```python
for epoch in range(1, max_epochs + 1):
    train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate(model, valid_loader, criterion, device)
    if val_metrics[selection_metric] improves:
        save_checkpoint(...)
```

#### 2. Add compatible CLI for end-to-end execution
File: code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py
Changes:
- Add args for manifest_dir, tabular_csv override, output_dir, target_col, model_mode, checkpoint, freeze policy, lr policy, selection metric, limit, dry-run.
- Keep defaults aligned with common.py directory constants.

### Success Criteria:

#### Automated Verification:
- [x] One-epoch smoke run completes and writes artifacts under output_dir.
- [x] best checkpoint file is written and reloadable.
- [x] val/test metrics JSON and predictions CSV files exist and are non-empty.
- [x] run manifest JSON records end-to-end configuration and split counts.

#### Manual Verification:
- [ ] Confirm training log shows decreasing train loss for smoke run.
- [ ] Confirm predicted values are numeric and mapped to participant_id correctly.
- [ ] Confirm rerunning with same seed produces stable metrics trend.

Implementation Note: After completing this phase and automated checks, pause for manual confirmation before Phase 4.

---

## Phase 4: Operationalization and Backward-Compatible Workflow Integration

### Overview

Make the new finetuning flow discoverable and runnable alongside the current staged pipeline.

### Changes Required:

#### 1. Add run examples and execution notes
File: code/baseline_encoder/LC-VIT/experiment/bash.sh
Changes:
- Add end-to-end finetuning command block using finetuning/main_finetune.py.
- Keep existing staged commands unchanged.

#### 2. Add finetuning config section or file
File: code/baseline_encoder/LC-VIT/experiment/config_regression.yaml
Changes:
- Add or document finetuning defaults for backbone/head learning rates, freeze schedule, and image preprocessing options.

#### 3. Add concise research traceability note
File: code/baseline_encoder/LC-VIT/experiment/research/report.md
Changes:
- Add short follow-up section linking to finetuning/plan.md and the new end-to-end script behavior once implemented.

### Success Criteria:

#### Automated Verification:
- [x] Help command for finetuning script prints all expected arguments.
- [x] Documented example command runs without argument parsing errors.
- [x] Existing staged train_regression command continues to run unchanged.

#### Manual Verification:
- [ ] Team can reproduce an end-to-end run from the command examples.
- [ ] Existing users of staged pipeline are not blocked by finetuning additions.
- [ ] Run directory layout is understandable and consistent with current experiment outputs.

Implementation Note: After completing this phase and automated checks, pause for manual confirmation before merging final workflow documentation.

---

## Testing Strategy

### Unit Tests:
- Dataset item correctness for image loading, transform application, and target extraction.
- Tabular normalization behavior with train-only statistics.
- Model forward path for fusion/image_only/clinical_only modes.
- Checkpoint save/load roundtrip for backbone and head parameters.

### Integration Tests:
- End-to-end one-epoch run from manifest_dir to output artifacts.
- Resume-from-checkpoint training continuation scenario.
- Freeze-then-unfreeze schedule scenario.

### Manual Testing Steps:
1. Run one-epoch dry smoke with limit=16 and inspect logged split counts.
2. Run short training with backbone frozen and verify encoder grads stay disabled.
3. Run short training with unfreeze_after_epoch=1 and verify encoder grads activate.
4. Compare output metrics/predictions format against train_regression outputs for compatibility.

## Performance Considerations

- On-the-fly image preprocessing introduces higher per-step CPU cost than CSV feature loading; num_workers and pinned memory should be exposed as CLI options.
- End-to-end finetuning increases GPU memory usage due to encoder activation storage; gradient accumulation and batch-size controls should be supported.
- Shared-backbone inference for three views triples image forward cost per sample, so checkpointing frequency and validation cadence should be configurable.

## Migration Notes

- The staged pipeline remains available and unchanged.
- End-to-end finetuning is introduced as an additional workflow under finetuning/main_finetune.py.
- Existing artifacts in experiment/artifacts remain valid and are not rewritten by this plan unless explicitly configured.

## References

- Existing staged entrypoints:
  - code/baseline_encoder/LC-VIT/experiment/build_regression_manifest.py:68
  - code/baseline_encoder/LC-VIT/experiment/extract_features.py:200
  - code/baseline_encoder/LC-VIT/experiment/merge_features.py:45
  - code/baseline_encoder/LC-VIT/experiment/train_regression.py:202
  - code/baseline_encoder/LC-VIT/experiment/eval_regression.py:33
- Existing dataset and model components:
  - code/baseline_encoder/LC-VIT/experiment/dataset.py:62
  - code/baseline_encoder/LC-VIT/experiment/model.py:72
- Finetuning entrypoint baseline to redesign:
  - code/baseline_encoder/LC-VIT/experiment/finetuning/main_finetune.py:25
