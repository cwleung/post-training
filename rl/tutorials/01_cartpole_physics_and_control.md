# Chapter 1: CartPole 物理動力學與經典控制 (CartPole Dynamics & Control)

> *「在深入現代大語言模型對齊與策略梯度之前，倒立擺（Inverted Pendulum）是每位強化學習研究者的第一座燈塔——它以極簡的 4 維狀態空間，揭示了相空間幾何、非線性不穩定性與延遲回報信用分配的本質。」*

---

## 核心心智模型：倒立擺平衡與非線性控制

CartPole（小車倒立擺）是強化學習控制領域的「Hello World」，也是 OpenAI Gym / Farama Gymnasium 的黃金基準測試。系統由一個在光滑水平導軌上移動的小車（質量為 $m_c$），以及通過無摩擦鉸鏈鉸接在小車重心的均勻擺桿（質量為 $m_p$、半長度為 $l$）構成：

```mermaid
flowchart LR
    subgraph Env["CartPole 閉環物理動力學系統"]
        F["水平控制外力 F ∈ {-10N, +10N}"] --> Dyn["二階非線性拉格朗日運動方程<br/>θ̈(角加速度) & ẍ(小車加速度)"]
        Dyn --> S["4-D 連續狀態空間 s = [x, ẋ, θ, θ̇]ᵀ"]
    end
    
    subgraph Controller["控制策略 Agent"]
        S --> PI["控制策略 π(a|s)"]
        PI -->|離散動作 a ∈ {0, 1}| F
    end

    S -->|終止條件: |θ| > 12° 或 |x| > 2.4m| DONE["回合截斷 Terminated"]

    classDef env fill:#1a365d,stroke:#3182ce,stroke-width:1.5px,color:#fff;
    classDef agent fill:#234e52,stroke:#319795,stroke-width:1.5px,color:#e6fffa;
    class Env,F,Dyn,S,DONE env;
    class Controller,PI agent;
```

### 為什麼純粹依靠位置 $(x, \theta)$ 無法維持平衡？
直覺上，初學者往往認為只要看見擺桿向右傾斜（$\theta > 0$），向右推車即可。但如果僅觀測位置，系統屬於**部分可觀測馬爾可夫決策過程（POMDP）**：
- 若擺桿向右傾斜 $\theta = +5^\circ$，但角速度 $\dot{\theta} = -20^\circ/\text{s}$（擺桿正在急速向左回正），此時若仍強行向右猛推小車，擺桿將直接向左劇烈翻轉失控。
- 因此，必須引入一階時間導數 $(\dot{x}, \dot{\theta})$ 構成**相空間（Phase Space）**，使系統在數學上恢復嚴格的**馬爾可夫無後效性（Markov Property）**。

---

## 1.1 拉格朗日力學建模與運動微分方程

對於倒立擺系統，選擇廣義坐標 $q = [x, \theta]^T$。小車與擺桿質心的坐標分別為：
$$x_c = x, \quad y_c = 0$$
$$x_p = x + l \sin\theta, \quad y_p = l \cos\theta$$

系統的動能 $T$ 與勢能 $V$ 分別為：
$$T = \frac{1}{2} m_c \dot{x}^2 + \frac{1}{2} m_p \left( \dot{x}_p^2 + \dot{y}_p^2 \right) + \frac{1}{2} I \dot{\theta}^2$$
$$V = m_p g l \cos\theta$$
其中均勻剛體擺桿繞質心的轉動慣量 $I = \frac{1}{3} m_p l^2$（當 $l$ 定義為擺桿全長時，若 $l$ 為半長則 $I = \frac{1}{12} m_p (2l)^2 = \frac{1}{3} m_p l^2$）。

代入歐拉-拉格朗日方程 $\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = Q_i$，可推導出系統的解析加速度方程：

### 擺桿角加速度微分方程
$$\ddot{\theta} = \frac{g \sin\theta + \cos\theta \left( \frac{-F - m_p l \dot{\theta}^2 \sin\theta}{m_c + m_p} \right)}{l \left( \frac{4}{3} - \frac{m_p \cos^2\theta}{m_c + m_p} \right)}$$

### 小車水平加速度微分方程
$$\ddot{x} = \frac{F + m_p l \left( \dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta \right)}{m_c + m_p}$$

> [!NOTE]
> **慣性恢復力矩直覺**：當小車向右加速時（$\ddot{x} > 0$），在擺桿的非慣性參考系中，質心會受到一個向左的等效慣性力 $-m_p \ddot{x}$，對支點產生逆時針方向的慣性恢復力矩 $\tau_{\text{inertial}} = m_p l \ddot{x} \cos\theta$。只要此力矩大於重力傾覆力矩 $m_p g l \sin\theta$，擺桿即可被拉回垂直平衡位置。

---

## 1.2 狀態空間與馬爾可夫決策過程 (MDP) 規格

在 Gymnasium 的標準實現中，環境被形式化定義為四元組 $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R})$：

| 變量 | 物理含義 | 單位 | 失敗邊界 (Termination) |
|---|---|---|---|
| $x$ | 小車在水平導軌上的位移 | $\text{m}$ | $x < -2.4$ 或 $x > +2.4$ |
| $\dot{x}$ | 小車水平速度 | $\text{m/s}$ | $[-\infty, +\infty]$ |
| $\theta$ | 擺桿偏離垂直向上的夾角 | $\text{rad}$ | $\theta < -12^\circ (-0.2095\text{ rad})$ 或 $\theta > +12^\circ$ |
| $\dot{\theta}$ | 擺桿擺動角速度 | $\text{rad/s}$ | $[-\infty, +\infty]$ |

- **動作空間 $\mathcal{A} = \{0, 1\}$**：$0$ 代表向左施加 $F = -10\text{N}$，$1$ 代表向右施加 $F = +10\text{N}$。
- **獎勵函數 $\mathcal{R}(s, a)$**：每維持一個時間步（$\Delta t = 0.02\text{s}$）未觸發終止條件，獲得 $r_t = +1.0$。
- **最大步數**：標準為 500 步（滿分 500 分代表倒立擺成功維持 10 秒不倒）。

> [!TIP]
> <button class="deep-link-btn" data-target="viz">🔬 啟動 CartPole 物理模擬器</button>，可以手動調節小車質量 $m_c$、擺桿質量 $m_p$ 與外力大小，觀察李雅普諾夫（Lyapunov）穩定性邊界的動態演化。

---

## 1.3 可執行的 PyTorch 仿真閉環代碼

以下代碼展示了標準 Gymnasium CartPole-v1 環境的封裝與基準前饋控制器：

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

class CartPoleHeuristicPolicy:
    """基於相空間幾何的啟發式線性閾值控制器"""
    def __init__(self, angle_weight: float = 1.0, angular_vel_weight: float = 0.5):
        self.w_theta = angle_weight
        self.w_omega = angular_vel_weight

    def act(self, obs: np.ndarray) -> int:
        x, x_dot, theta, theta_dot = obs
        # 當預期傾角 (位置 + 速度超前量) > 0 時，向右加速推車 (action=1)
        phase_signal = self.w_theta * theta + self.w_omega * theta_dot
        return 1 if phase_signal > 0 else 0

def run_simulation(episodes: int = 5):
    env = gym.make("CartPole-v1")
    policy = CartPoleHeuristicPolicy()
    
    total_rewards = []
    for ep in range(episodes):
        obs, info = env.reset(seed=42 + ep)
        ep_reward = 0.0
        done = False
        
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
        total_rewards.append(ep_reward)
        print(f"Episode {ep + 1}: 累計回報 = {ep_reward:.1f} / 500.0")
        
    env.close()
    return np.mean(total_rewards)

if __name__ == "__main__":
    avg_score = run_simulation()
    print(f"平均控制得分: {avg_score:.2f}")
```

---

## 1.4 參數敏感性與物理極限對比

| 物理參數 | 默認數值 | 數值加倍時的物理影響 | 強化學習策略的收斂難度 |
|---|---|---|---|
| 小車質量 $m_c$ | $1.0\text{ kg}$ | 小車慣性變大，需要更長時間加速，慣性力響應滯後 | 難度顯著上升，容易因導軌超界而失敗 |
| 擺桿長度 $l$ | $0.5\text{ m}$ | 固有頻率 $\omega_n = \sqrt{g/l}$ 降低，擺動變慢 | 難度降低，Agent 有更充裕的時間反應 |
| 擺桿質量 $m_p$ | $0.1\text{ kg}$ | 質心上移，對小車的反作用力矩劇增，系統非線性加劇 | 難度上升，數值積分容易出現吉布斯震盪 |
| 控制週期 $\Delta t$ | $0.02\text{ s}$ | 若增大到 $0.1\text{ s}$，離散化截斷誤差破壞歐拉積分 | 難度暴增，經典離散策略極易發散崩潰 |

> [!WARNING]
> 在連續動力學離散化求解中，Gymnasium 採用半隱式歐拉法（Semi-implicit Euler）。若隨意加大步長 $\Delta t$，數值能量將不再守恆，會人為引入「數值泵能」現象，導致擺桿無故劇烈加速發散。

---

## 🤔 架構深度思辨與工業界陷阱 (Architectural Insight & Production Pitfalls)

### 問題 1：在只有單步照片觀測的 Visual RL（如 Atari 或機器人攝影機）中，系統退化為 POMDP，工業界最標準的數學修復手段是什麼？
- **解答**：當無法直接獲取一階時間導數 $\dot{x}$ 和 $\dot{\theta}$ 時，單幀圖像丟失了物體的動量和運動方向。工業界有三種經典解法：
  1. **歷史幀堆疊（Frame Stacking）**：將當前幀與過去 $K-1$ 幀沿通道維度拼接（如 Nature DQN 採用 4 幀堆疊，通道數為 $4 \times 84 \times 84$）。由於相鄰幀的時間差 $\Delta t$ 已知，卷積神經網絡的跨通道差分能自發擬合出離散速度導數 $\dot{s}_t \approx \frac{s_t - s_{t-1}}{\Delta t}$。
  2. **遞歸神經網絡（Recurrent RL / R2D2 / DRQN）**：利用 GRU / LSTM 隱藏狀態 $h_t$ 維持歷史序列的信念狀態（Belief State）$b(s_t) = P(s_t | o_{\le t}, a_{< t})$。
  3. **Transformer 記憶架構（Decision Transformer / In-Context Policy）**：將長窗口的上下文 Tokens 直接送入自注意力機制，利用因果掩碼建立跨時間步的動量建模。

### 問題 2：CartPole 的獎勵函數設定為每步 $r_t = +1.0$，這種「稀疏生存獎勵」在複雜連續控制環境（如四足機器人或大模型長思維鏈生成）中會引發什麼致命後果？
- **解答**：
  1. **信用分配災難（Credit Assignment Problem）**：若一隻機器狗奔跑了 1000 步在最後一刻摔倒，全過程每步均為 $+1.0$，策略梯度無法精準辨識到底是「第 200 步的微小姿態失衡」還是「第 999 步的腿部滑移」導致了崩潰。
  2. **獎勵欺騙與不變怠速（Reward Hacking / Procrastination）**：Agent 可能發現只要「站在原地極其微小地顫抖」，就能無風險地持續薅取生存獎勵，從而拒絕探索向前奔跑的動作。
  3. **工業界破局方案**：必須引入**勢函數獎勵塑形（Potential-based Reward Shaping）** $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$，或利用 GAE（廣義優勢估計）與步驟級過程獎勵模型（PRM）進行精細化優勢拆解。

---

## 參考文獻與經典論文

1. **Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983).** *Neuronlike adaptive elements that can solve difficult learning control problems.* IEEE Transactions on Systems, Man, and Cybernetics, SMC-13(5), 834-846.
2. **Brockman, G., et al. (2016).** *OpenAI Gym.* arXiv preprint arXiv:1606.01540.
3. **Mnih, V., et al. (2015).** *Human-level control through deep reinforcement learning.* Nature, 518(7540), 529-533.
