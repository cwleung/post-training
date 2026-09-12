# 18 Interactive Visual Lab Simulation Components Reference

This catalog details each of the **18 parameter simulation laboratories** built into the interactive learning platform (`frontend/src/entities/simulation/labCatalog.ts`), including mathematical formulations, interactive controls, and rendering components.

---

## 1. Classical Control & Reinforcement Learning Labs (`category: 'control'`)

### 1. CartPole 物理動力學與基礎控制 (`cartpole`)
- **Associated Chapter**: `ch01`
- **Key Equation**:
  $$\ddot{\theta} = \frac{g \sin\theta + \cos\theta \left( \frac{-F - m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p} \right)}{l \left( \frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p} \right)}$$
- **Interactive Controls**: Manual push force buttons (Left/Right), reset trigger, gravity $g$ slider, cart mass $m_c$, pole length $l$.
- **Rendering**: Real-time 60fps 2D physics canvas with angle $\theta$, cart position $x$, and velocity vectors.

### 2. 多臂老虎機·ε-Greedy 與 UCB1 (`bandit`)
- **Associated Chapter**: `ch02`
- **Key Equation**:
  $$A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$
- **Interactive Controls**: Exploration constant $c$, $\epsilon$-greedy exploration rate $\epsilon$, bandit arm count, step batch buttons (+1, +10, +100).
- **Rendering**: Payout distribution bar chart and cumulative regret bound curves.

### 3. DQN 經驗回放與 TD-Error 震盪 (`dqn`)
- **Associated Chapter**: `ch03`
- **Key Equation**:
  $$\mathcal{L}(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
- **Interactive Controls**: Replay buffer capacity $N$, target network sync interval $C$, batch size $B$, learning rate $\alpha$.
- **Rendering**: Buffer queue sampling heatmaps and TD loss variance curves showing moving averages.

### 4. 策略梯度 REINFORCE 與 Baseline 方差削減 (`policy_gradient`)
- **Associated Chapter**: `ch04`
- **Key Equation**:
  $$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \left( G_t - b(s_t) \right) \right]$$
- **Interactive Controls**: Trajectory count $N$, baseline selector ($b=0$ vs $b=V(s)$ moving average), reward noise scale.
- **Rendering**: Gradient estimate variance scatter plots displaying dramatic spread reduction with baseline.

### 5. Actor-Critic 與 GAE 優勢估計 (`actor_critic`)
- **Associated Chapter**: `ch05`
- **Key Equation**:
  $$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$
- **Interactive Controls**: Temporal discount factor $\gamma \in [0.8, 1.0]$, GAE parameter $\lambda \in [0.0, 1.0]$.
- **Rendering**: Bias-variance trade-off spectrum and multi-step advantage decay bars.

### 6. 連續動作空間高斯策略與熵正則化 (`continuous`)
- **Associated Chapter**: `ch06`
- **Key Equation**:
  $$a_t \sim \mathcal{N}(\mu_\theta(s_t), \sigma_\theta^2(s_t)), \quad \mathcal{H}(\pi) = \frac{1}{2} \ln(2\pi e \sigma^2)$$
- **Interactive Controls**: Entropy temperature $\alpha$, mean $\mu$, standard deviation $\sigma$, $\tanh$ squashing toggle.
- **Rendering**: Dynamic Gaussian probability density function before and after $\tanh$ boundary squashing.

### 7. 離線保守策略 (CQL) 與分佈偏移 (`offline`)
- **Associated Chapter**: `ch08`
- **Key Equation**:
  $$\min_Q \alpha \left( \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu} [Q(s, a)] - \mathbb{E}_{(s,a) \sim \mathcal{D}} [Q(s, a)] \right) + \frac{1}{2} \mathcal{L}_{\text{TD}}$$
- **Interactive Controls**: Conservative weight $\alpha$, OOD action deviation, dataset sample coverage.
- **Rendering**: Q-value estimation landscape contrasting naive Q-value explosion with CQL lower-bound conservatism.

---

## 2. Preference Optimization & Alignment Labs (`category: 'alignment'`)

### 8. PPO 截斷代理目標 (Clipped Objective) (`ppo`)
- **Associated Chapter**: `ch10`
- **Key Equation**:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
- **Interactive Controls**: Importance ratio $r_t(\theta) \in [0.0, 2.5]$, advantage $\hat{A}_t \in [-2.0, 2.0]$, clipping threshold $\epsilon \in [0.1, 0.3]$.
- **Rendering**: Interactive 2D coordinate plot displaying unclipped surrogate, clipped surrogate, and final objective curve with active region highlighting.

### 9. DPO 隱式獎勵邊界 (`dpo`)
- **Associated Chapter**: `ch11`
- **Key Equation**:
  $$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
- **Interactive Controls**: Temperature $\beta \in [0.01, 1.0]$, winner log-ratio $\log \frac{\pi}{\pi_{\text{ref}}}(y_w)$, loser log-ratio $\log \frac{\pi}{\pi_{\text{ref}}}(y_l)$.
- **Rendering**: Implicit reward differential bar chart and gradient push-pull magnitude curves.

### 10. GRPO 群組相對優勢歸一化 (DeepSeek-R1) (`grpo`)
- **Associated Chapter**: `ch12`
- **Key Equation**:
  $$\hat{A}_i = \frac{R_i - \text{mean}(\{R_1 \dots R_G\})}{\text{std}(\{R_1 \dots R_G\} + \epsilon)}$$
- **Interactive Controls**: Group size $G \in [2, 16]$, candidate scores array $[R_1, \dots, R_G]$, epsilon $\epsilon$.
- **Rendering**: Group score distribution chart and computed normalized advantage $\hat{A}_i$ bars.

### 11. SimPO (無參考模型) vs 經典 DPO 邊界對比 (`rlvr_simpo`)
- **Associated Chapter**: `rlvr11`
- **Key Equation**:
  $$\mathcal{L}_{\text{SimPO}} = -\log \sigma \left( \frac{\beta}{|y_w|} \log \pi(y_w|x) - \frac{\beta}{|y_l|} \log \pi(y_l|x) - \gamma \right)$$
- **Interactive Controls**: Target margin $\gamma$, length penalty multiplier, temperature $\beta$.
- **Rendering**: Comparison chart showing zero Reference Model memory savings and length bias suppression.

---

## 3. Post-Training & Systems Engineering Labs (`category: 'rlvr_systems'`)

### 12. XML 標籤解析與左側填充 (Left-Padding) (`rlvr_data`)
- **Associated Chapter**: `rlvr01`
- **Key Equation**:
  $$\text{Prompt} \oplus \text{Reasoning} \oplus \text{Answer}, \quad \text{Mask}_{\text{pad}} = 0$$
- **Interactive Controls**: Sample response selector, XML tag mutation, padding alignment toggle (Left vs Right).
- **Rendering**: Dynamic tensor mask matrix visualizer showing causal attention keys across padded batches.

### 13. 複合獎勵函數加權平衡器 (`rlvr_rewards`)
- **Associated Chapter**: `rlvr02`
- **Key Equation**:
  $$R = w_{\text{acc}} R_{\text{acc}} + w_{\text{fmt}} R_{\text{fmt}} - \lambda_{\text{len}} \Delta L$$
- **Interactive Controls**: Accuracy reward weight $w_{\text{acc}}$, format reward weight $w_{\text{fmt}}$, length penalty $\lambda_{\text{len}}$.
- **Rendering**: Real-time radar chart illustrating Goodhart gaming resistance and policy score trajectories.

### 14. 顯卡 VRAM 預算與 LoRA 顯存分配器 (`rlvr_lora_vram`)
- **Associated Chapter**: `rlvr04`
- **Key Equation**:
  $$W = W_0 + \frac{\alpha}{r} B \times A, \quad \text{VRAM} = W_0 + W_{\text{LoRA}} + \text{Opt} + \text{KV} + \text{Act}$$
- **Interactive Controls**: Base model selector (0.5B, 1.5B, 3B, 7B, 14B), LoRA rank $r \in [8, 64]$, batch size, context length.
- **Rendering**: GPU VRAM allocation breakdown (16GB T4 / 24GB 3090 / 80GB A100) with real-time OOM warning alerts.

### 15. SFT 冷啟動蒸餾與拒絕採樣 (`rlvr_coldstart`)
- **Associated Chapter**: `rlvr09`
- **Key Equation**:
  $$\mathcal{D}_{\text{SFT}} = \{ (x, y_i) \mid r(x, y_i) = 1, \text{Quality}(y_i) \ge \tau \}$$
- **Interactive Controls**: Candidate rollouts count, chain-of-thought density threshold $\tau$, verification pass filter.
- **Rendering**: Distillation pipeline sieve diagram showing high-quality trajectory retention rates.

### 16. 3D-HybridEngine (veRL + vLLM) 分散式架構計算器 (`rlvr_distributed`)
- **Associated Chapter**: `rlvr10`
- **Key Equation**:
  $$\text{Throughput} \propto N_{\text{GPU}} \times \text{TPS}_{\text{vLLM}} \times (1 - \text{Comm}_{\text{TP}})$$
- **Interactive Controls**: GPU cluster node count, Tensor Parallelism TP degree, PagedAttention block size.
- **Rendering**: Distributed engine topology visualizer showing All-Reduce latency overhead vs zero-copy KV cache throughput.

---

## 4. Reasoning & Test-Time Scaling Labs (`category: 'reasoning'`)

### 17. Pass@k 與多數投票 (Majority@k) 組合曲線 (`rlvr_passk`)
- **Associated Chapter**: `rlvr05`
- **Key Equation**:
  $$\text{Pass@}k = 1 - \frac{\binom{N-c}{k}}{\binom{N}{k}}, \quad \text{Maj@}k = \arg\max_a \text{count}(a)$$
- **Interactive Controls**: Single-sample pass probability $p$, total rollouts $N$, evaluation budget $k$.
- **Rendering**: Interactive hypergeometric combinatorial estimation curve vs majority voting consensus rate.

### 18. 推理期思考 Token 擴展定律 (`rlvr_scaling`)
- **Associated Chapter**: `rlvr08`
- **Key Equation**:
  $$\text{Acc} \propto \alpha \log(\text{Tokens}_{\text{think}}) + \beta \log(G), \quad \text{FLOPs} \propto N \times L$$
- **Interactive Controls**: Thinking token limit slider (128 to 4096), group exploration size $G$, difficulty level.
- **Rendering**: Log-linear accuracy scaling curves demonstrating test-time compute returns on complex reasoning benchmarks.
