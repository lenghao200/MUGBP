# MUGBP Project Architecture

This document describes the overall architecture of the MUGBP codebase and explains how the implementation corresponds to the paper **Momentum-Updated Granular-Ball Prototypes: Stabilizing Cross-Modal Alignment for Multimodal Intent Recognition**. The project targets multimodal intent recognition, where textual, visual, and acoustic signals are jointly modeled to infer fine-grained user intents.

## 1. Project Objective

MUGBP addresses semantic heterogeneity and prototype staleness in multimodal intent recognition. Text, video, and audio features have different distributions, noise patterns, and temporal structures. Direct fusion or point-wise prototype alignment may lead to unstable cross-modal matching and blurred class boundaries.

The project implements the following main mechanisms:

- Text-anchor-based semantic representation for stable intent guidance.
- Adaptive granular-ball splitting to construct class-aware semantic regions.
- Online momentum updating of granular-ball centers and radii at the mini-batch level.
- Granular-ball guidance loss for visual, acoustic, and fused representations.
- Multi-view contrastive learning and classification optimization.

The overall training objective follows:

```text
L_total = L_cls + L_cons + lambda_gb * L_gb
```

where `L_cls` is the classification loss, `L_cons` is the multi-view contrastive loss, and `L_gb` is the granular-ball guidance loss.

## 2. Top-Level Directory Structure

```text
MUGBP/
+-- run.py
+-- configs/
|   +-- base.py
|   +-- MUGBP_MIntRec.py
|   +-- MUGBP_MIntRec2.py
+-- data/
|   +-- __init__.py
|   +-- base.py
|   +-- text_pre.py
|   +-- mm_pre.py
|   +-- MMDataset.py
|   +-- utils.py
+-- methods/
|   +-- __init__.py
|   +-- MUGBP/
|       +-- model.py
|       +-- manager.py
|       +-- loss.py
|       +-- AlignNets.py
|       +-- PeepholeLSTM.py
|       +-- GranularBall/
|       +-- SubNets/
+-- utils/
    +-- functions.py
    +-- metrics.py
```

## 3. Entry Point and Execution Flow

The main entry point is `run.py`. It parses command-line arguments, loads configuration files, prepares multimodal data, initializes the selected method, and starts training or testing.

The execution flow is:

```text
run.py
  -> parse_arguments()
  -> ParamManager(args)
  -> add_config_param(args, config_file_name)
  -> DataManager(args)
  -> method_map[args.method]
  -> MUGBP_manager(args, data, labels_weight)
  -> _train(args) / _test(args)
```

The default method is:

```text
--method mugbp
```

The default configuration file is:

```text
--config_file_name MUGBP_MIntRec.py
```

Example command for MIntRec:

```bash
py run.py --train --save_model --save_results --dataset MIntRec --method mugbp --config_file_name MUGBP_MIntRec.py
```

Example command for MIntRec2:

```bash
py run.py --train --save_model --save_results --dataset MIntRec2 --method mugbp --config_file_name MUGBP_MIntRec2.py
```

## 4. Configuration Layer

The configuration files are stored in `configs/`.

| File | Description |
| --- | --- |
| `configs/base.py` | Converts command-line arguments into an `EasyDict` and dynamically imports the selected experiment configuration. |
| `configs/MUGBP_MIntRec.py` | Hyperparameter settings for the MIntRec dataset. |
| `configs/MUGBP_MIntRec2.py` | Hyperparameter settings for the MIntRec2 dataset. |

Important hyperparameters include:

| Parameter | Description |
| --- | --- |
| `purity_train` | Purity threshold used for granular-ball splitting during training. |
| `min_ball_train` | Minimum number of samples required for further granular-ball splitting. |
| `lambda_gb` | Weight of the granular-ball guidance loss. |
| `momentum` | Momentum coefficient for EMA-based online center and radius updating. |
| `loss` | Type of contrastive loss, currently supporting `InfoNCE` and `SupCon`. |
| `aligned_method` | Sequence alignment method for multimodal features, such as `ctc`. |
| `max_depth` | Maximum depth of the dynamic fully connected layer. |

## 5. Data Layer

The data processing modules are located in `data/`. They convert raw annotations and pre-extracted multimodal features into PyTorch datasets and dataloaders.

### 5.1 Dataset Metadata

`data/__init__.py` defines dataset-level metadata, including intent labels, label mappings, feature dimensions, and maximum sequence lengths. The current codebase contains metadata for:

- `MIntRec`
- `MIntRec2`
- `MELD`

MIntRec and MIntRec2 are the main benchmark datasets used in the paper.

### 5.2 Data Loading and Packaging

`data/base.py` provides the `DataManager`, which serves as the entry point of the data layer. It performs the following steps:

1. Reads `train.tsv`, `dev.tsv`, and `test.tsv`.
2. Converts intent labels into numerical label IDs.
3. Processes text inputs through `text_pre.py`.
4. Loads and pads visual and acoustic features through `mm_pre.py`.
5. Wraps text, video, audio, and label information into `MMDataset`.

Each sample returned by `MMDataset.__getitem__()` contains:

| Field | Description |
| --- | --- |
| `label_ids` | Intent label ID. |
| `text_feats` | Standard text input, including `input_ids`, `input_mask`, and `segment_ids`. |
| `cons_text_feats` | Text-anchor input containing the ground-truth intent token. |
| `condition_idx` | Start position of the intent-condition token span. |
| `video_feats` | Visual feature sequence. |
| `audio_feats` | Acoustic feature sequence. |

### 5.3 Text-Anchor Construction

`data/text_pre.py` constructs two text inputs:

- `text_feats`: the original utterance followed by `[MASK]` intent slots.
- `cons_text_feats`: the original utterance followed by the ground-truth intent label tokens.

The second input is used by the `Anchor` branch to extract text-anchor features. These anchors serve as stable semantic references for granular-ball construction and online momentum updating.

## 6. Method Registration Layer

`methods/__init__.py` maps method names to their corresponding manager classes:

```python
from .MUGBP.manager import MUGBP_manager

method_map = {
    'mugbp': MUGBP_manager,
}
```

`run.py` retrieves the manager class from `method_map` according to `args.method`. To add a new method, create a new method directory under `methods/` and register it in this mapping.

## 7. Core MUGBP Model Layer

The core implementation is located in `methods/MUGBP/`.

### 7.1 `model.py`

`model.py` defines the multimodal encoding, fusion, and forward propagation modules.

| Class / Module | Description |
| --- | --- |
| `Anchor` | BERT-based text-anchor encoder for extracting anchor features from `cons_text_feats`. |
| `Positive` | Main multimodal encoder, including text, visual, acoustic, and fusion branches. |
| `Positive_Model` | Adds the classification head and extracts condition-level representations. |
| `DAF` | Dynamic Attention Fusion module for multimodal feature fusion. |
| `MUGBP` | Top-level model that combines the main branch and the anchor branch. |

The forward pass returns:

```text
logits,
pooled_output,
fused_condition,
cons_condition,
text_condition,
visual_condition,
acoustic_condition
```

These outputs are used for classification, multi-view contrastive learning, and granular-ball boundary guidance.

### 7.2 Dynamic Attention Fusion

`DAF` corresponds to the Dynamic Attention Fusion component in the paper. It performs the following operations:

1. Projects textual, visual, and acoustic representations into compatible semantic dimensions.
2. Uses `DynamicLayer` to generate dynamic weights for visual and acoustic modalities.
3. Applies BiLSTM-based attention to estimate modality-specific contributions.
4. Combines weighted visual features, weighted acoustic features, and text features.
5. Applies LayerNorm and dropout to produce the final fused representation.

Related files:

- `methods/MUGBP/model.py`
- `methods/MUGBP/SubNets/dynamicfc.py`

### 7.3 Sequence Alignment

`AlignNets.py` aligns text, video, and audio sequences to a unified temporal length before fusion. The supported alignment modes are:

- `avg_pool`
- `ctc`
- `conv1d`
- `sim`

The current default configuration uses `ctc`, which maps visual and acoustic sequences to the text sequence length through a CTC-style alignment module.

## 8. Granular-Ball Prototype Layer

Granular-ball-related modules are located in:

```text
methods/MUGBP/GranularBall/
```

The main entry point is `gbcluster` in `cluster.py`. During training, `MUGBP_manager` first extracts text-anchor features from the full training set and then constructs a global granular-ball set:

```python
gb_centroids, gb_radii, gb_labels = self.gb_cluster(
    args, global_feats, global_labels, select=False
)
```

Each granular ball is represented by:

| Component | Description |
| --- | --- |
| `gb_centroids` | Granular-ball centers. |
| `gb_radii` | Granular-ball radii. |
| `gb_labels` | Class labels assigned to granular balls. |

This layer implements the adaptive granular-ball splitting strategy described in the paper. Impure coarse regions are recursively split into finer class-aware sub-balls according to the purity threshold and minimum sample constraint.

## 9. Training Management Layer

`methods/MUGBP/manager.py` controls training, validation, testing, optimization, and checkpointing.

### 9.1 Initialization

`MUGBP_manager.__init__()` performs:

- Device selection.
- Granular-ball clustering module initialization.
- `MUGBP` model initialization.
- Optimizer and learning-rate scheduler initialization.
- Train, validation, and test dataloader construction.
- Classification and contrastive loss initialization.

### 9.2 Epoch-Level Training Procedure

The training procedure can be summarized as:

```text
for each epoch:
  1. Extract full-dataset text-anchor features with the Anchor branch.
  2. Build global granular-ball prototypes from text-anchor features.
  3. Iterate over mini-batches for model training.
  4. Update matched granular-ball centers and radii online using EMA.
  5. Compute granular-ball guidance loss L_gb.
  6. Compute multi-view contrastive loss L_cons.
  7. Compute classification loss L_cls.
  8. Backpropagate and update model parameters.
  9. Evaluate on the validation set and save best_model.pth.
```

### 9.3 Online Momentum Update

The online momentum update mechanism is implemented inside the mini-batch training loop in `manager.py`. For each sample:

1. Candidate granular balls with the same class label are selected.
2. The nearest candidate granular ball is matched to the current text anchor.
3. Its center is updated by EMA:

```text
new_centroid = momentum * old_centroid + (1 - momentum) * current_anchor
```

4. Its radius is updated according to the distance between the current anchor and the updated center.
5. A minimum-radius constraint is applied to prevent prototype collapse.

This design allows granular-ball prototypes to follow the evolving feature manifold during training, reducing the lag caused by static or epoch-level prototype refresh.

### 9.4 Granular-Ball Guidance Loss

The granular-ball guidance loss constrains three types of representations:

- `visual_condition`
- `acoustic_condition`
- `condition`, i.e., the fused representation

For each feature, the model searches for the nearest same-class granular ball. If the feature lies outside the ball boundary, a ReLU-truncated boundary violation penalty is applied:

```text
loss = relu(distance(feature, nearest_center) - nearest_radius)
```

The visual, acoustic, and fused penalties are accumulated as `gb_loss`, which is then weighted by `lambda_gb` and added to the total loss.

## 10. Loss Function Layer

`methods/MUGBP/loss.py` implements the multi-view contrastive learning objectives.

| Class | Description |
| --- | --- |
| `Multi_infoNCE` | Uses the text anchor as the query and aligns text, visual, acoustic, and fused views. |
| `InfoNCE` | Base InfoNCE implementation. |
| `Multi_SupCon` | Multi-view wrapper for supervised contrastive learning. |
| `SupConLoss` | Base supervised contrastive loss. |

The current default setting is:

```text
loss = InfoNCE
```

Therefore, the contrastive loss is computed as:

```python
cons_loss = Multi_infoNCE.compute_loss(
    text_anchor=cons_condition,
    text_view=text_condition,
    visual_view=visual_condition,
    acoustic_view=acoustic_condition,
    global_view=condition
)
```

## 11. Evaluation and Output

Testing is implemented in `MUGBP_manager._test()`. The manager loads:

```text
outputs/.../models/best_model.pth
```

and evaluates the model on the test set. The reported metrics include:

- `ACC`
- `WF1`
- `WP`
- `R`

Metric computation and result saving are implemented in:

- `utils/metrics.py`
- `utils/functions.py`

If `--save_results` is enabled, results are written to:

```text
results/results.csv
```

## 12. Mapping Between Paper Components and Code

| Paper Component | Code Location | Implementation Role |
| --- | --- | --- |
| Multimodal Feature Encoding | `methods/MUGBP/model.py` | BERT text encoder, Transformer visual encoder, and PeepholeLSTM acoustic encoder. |
| Dynamic Attention Fusion | `DAF`, `DynamicLayer` | Dynamically fuses textual, visual, and acoustic representations. |
| Text Anchor | `Anchor`, `cons_text_feats` | Extracts stable text-anchor features from label-conditioned text inputs. |
| Adaptive Granular-Ball Splitting | `methods/MUGBP/GranularBall/` | Builds class-aware granular-ball prototypes from text anchors. |
| Online Momentum Update | `MUGBP_manager._train()` | Updates granular-ball centers and radii during mini-batch optimization. |
| Granular-Ball Guidance Loss | `MUGBP_manager._train()` | Applies boundary-aware constraints to visual, acoustic, and fused representations. |
| Multi-View Contrastive Learning | `methods/MUGBP/loss.py` | Aligns multiple modality views with InfoNCE or SupCon. |
| Intent Classification | `Positive_Model.classifier` | Predicts intent labels from fused pooled representations. |

## 13. Extension Points

The following locations are useful for further experiments:

| Objective | Modification Location |
| --- | --- |
| Adjust granular-ball regularization strength | `lambda_gb` in `configs/MUGBP_MIntRec.py` or `configs/MUGBP_MIntRec2.py`. |
| Adjust granular-ball splitting granularity | `purity_train` and `min_ball_train`. |
| Adjust online update smoothness | `momentum`. |
| Switch contrastive loss | `loss`, choosing between `InfoNCE` and `SupCon`. |
| Change sequence alignment strategy | `aligned_method`. |
| Add a new dataset | Add benchmark metadata in `data/__init__.py` and ensure compatibility with `DataManager`. |
| Add a new method | Create a new method directory under `methods/` and update `methods/__init__.py`. |

## 14. Summary

The codebase follows a layered design. `run.py` controls the experiment pipeline, `configs/` defines hyperparameters, `data/` prepares multimodal inputs, `methods/MUGBP/model.py` builds multimodal representations, `methods/MUGBP/GranularBall/` constructs text-anchor-based granular-ball prototypes, and `manager.py` coordinates online prototype updating, granular-ball guidance, contrastive learning, classification, validation, and testing.

In short, the implementation realizes the central idea of MUGBP: dynamic, boundary-aware, text-anchored granular-ball prototypes for stable cross-modal alignment in multimodal intent recognition.
