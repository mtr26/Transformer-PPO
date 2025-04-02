# Transformer-PPO

**Transformer-PPO** integrates the Decision Transformer architecture with Proximal Policy Optimization (PPO) to enhance reinforcement learning (RL) performance. By leveraging the Transformer's attention mechanisms, this project aims to improve policy learning in complex environments.

---

## 🚀 Features

- **Transformer-Based Policy Network**: Utilizes the Transformer's capacity to model sequential data, enhancing decision-making processes.
- **Stable Training with PPO**: Employs PPO, a reliable RL algorithm, ensuring stable and efficient policy updates.
- **Customizable Environments**: Supports integration with various RL environments for diverse experimentation.

---

## 📂 Installation

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/mtr26/Transformer-PPO.git
cd Transformer-PPO
```

## 2️⃣ Create a virtual environment (optional but recommended):
```bash
python3 -m venv env
source env/bin/activate  # On Windows, use 'env\Scripts\activate'
```

## 3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

```python
from ppo import DT_PPO
import gym

# Initialize environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Create and train agent
agent = DT_PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_size=32,
    clip=0.2,
    epoch=10,
    gamma=0.99,
    lr=3e-6,
    nhead=1,
    nlayer=1
)

# Train for 20000 timesteps
rewards = agent.Learn(int(2e4), 'CartPole-v1')
```

## 📄 License
This project is licensed under the MIT License.

## ⭐ Acknowledgments
CleanRL's PPO with Transformer-XL: Inspiration for integrating Transformer architectures with PPO.
Hugging Face's TRL: Tools and libraries for training transformer models with reinforcement learning.
OpenAI Gym: Environment interface for reinforcement learning
