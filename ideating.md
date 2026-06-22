# JEPA Architecture Design

## Goals
- Minimal parameter count
- Decently generalizable
- Inputs: image observations + joint positions
- Minimal loss terms (no loss zoo)

## Components

### 1. Observation Encoder
- Input: RGB image + joint positions
- Output: latent embedding
- **First candidate: MobileNetV3-Small** (truncated) — ~1-2M params, designed for efficiency
- Future experiment: **Slot Attention** on top of encoder for object-centric representations
  - Motivation: handle compositional long-horizon tasks (e.g. "put 3 cubes in a bowl")
  - Slots decompose scene into per-object latents, enabling combinatorial generalization
  - Could sit between image backbone and predictor as a structuring layer
- Open questions:
  - Fuse image + joints early (concat before encoding) or late (encode separately, merge)?
  - How many slots? Fixed or dynamic?
  - Latent dimensionality?

### 2. Action Encoder
- Input: action vector (or action sequence with frameskip)
- Output: action embedding in same latent space
- Open questions:
  - Simple MLP sufficient?
  - Frameskip strategy?

### 3. Predictor
- Input: current latent + action embedding
- Output: predicted next latent
- **Candidate: Residual MLP** (`z' = z + MLP(z, a)`) — ~0.3-0.5M params
  - Learns dynamics (deltas) rather than absolute states
  - Single-step, rolled out at inference
- Alternative: GRU cell for longer horizon rollouts
- Note: if using slot attention, predictor operates per-slot or over slot set
  - Per-slot + interaction: allows reasoning about multi-object dynamics
- Open questions:
  - Action conditioning method (concatenation vs AdaLN)?
  - How to handle slot interactions in predictor (if using slots)?

### 4. Collapse Prevention
- Strategy: SIGReg (no EMA target encoder needed)
- Keeps latent space spread out without adding parameters or a target encoder copy
- Alternative: VICReg (variance + covariance terms) — slightly more loss terms but simpler to implement

### 5. Loss Function
- Prediction MSE: ||predicted_latent - actual_latent||^2
- Regularizer: SIGReg or VICReg
- Total: 2 terms (maybe 3 if VICReg splits into variance + covariance)

### 6. Planner (Inference)
- CEM or gradient-based optimization over action sequences
- Cost: distance between predicted final latent and goal latent
- Open questions:
  - Planning horizon?
  - Number of action samples?
  - Learned policy to seed proposals (later optimization)?

## Architecture Decisions Still Needed
- [x] Image backbone choice → MobileNetV3-Small (first experiment)
- [ ] How to fuse image + joint modalities
- [x] Predictor architecture → Residual MLP (first experiment)
- [ ] Latent dimensionality
- [x] Single-step vs multi-step prediction → single-step, rollout at inference
- [ ] Action conditioning method
- [ ] SIGReg vs VICReg for collapse prevention
- [ ] Slot attention design (num slots, where it sits, interaction model)

## Experiment Roadmap
1. **V1 (baseline)**: MobileNetV3-Small + residual MLP predictor + SIGReg. Get something training.
2. **V2 (slot attention)**: Add slot attention between encoder and predictor. Test on multi-object tasks.
3. **V3 (long horizon)**: Evaluate compositional generalization on tasks like "put N cubes in bowl."

## Reference Implementations
- **le-wm**: Pixel-based, ViT-Tiny encoder, Transformer predictor with AdaLN, SIGReg, 2 loss terms, ~15M params, no EMA
- **Mini-JEPA**: State-based, MLP encoder, GRU predictor, VICReg + EMA, 10+ loss terms, ~2-6M params
