# Research Ideas

## idea-001: Simple CNN Baseline
- **Priority**: high
- **Category**: architecture
- **Hypothesis**: Establish baseline accuracy with a simple 3-layer CNN on CIFAR-10.

```yaml
model:
  type: simple_cnn
  channels: [32, 64, 128]
training:
  lr: 0.001
  epochs: 5
```

## idea-002: ResNet-Style with Skip Connections
- **Priority**: high
- **Category**: architecture
- **Hypothesis**: Skip connections should improve gradient flow and beat the simple CNN baseline.

```yaml
model:
  type: resnet_small
  channels: [64, 128]
  blocks_per_stage: 2
training:
  lr: 0.001
  epochs: 10
```

## idea-003: Wide CNN
- **Priority**: medium
- **Category**: architecture
- **Hypothesis**: Wider layers with fewer stages may capture more features per layer.

```yaml
model:
  type: wide_cnn
  width: 256
training:
  lr: 0.0005
  epochs: 10
```

## idea-004: High Learning Rate Simple CNN
- **Priority**: medium
- **Category**: hyperparameter
- **Hypothesis**: Higher learning rate with the baseline CNN might converge faster.

```yaml
model:
  type: simple_cnn
  channels: [32, 64, 128]
training:
  lr: 0.01
  epochs: 5
```

## idea-005: Deep ResNet with More Blocks
- **Priority**: low
- **Category**: architecture
- **Hypothesis**: More residual blocks per stage should increase model capacity.

```yaml
model:
  type: resnet_small
  channels: [64, 128]
  blocks_per_stage: 4
training:
  lr: 0.001
  epochs: 10
```
