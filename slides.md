---
marp: true
size: 16:9
style: |
  img {
    display: block;
    margin: 0 auto;
  }

  /* Darker footer text (and subtle background for readability) */
  footer {
    color: #111;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.85);
    padding: 4px 16px;
  }
---

# Introduce of ReinforceReinforcement Learning
Reinforcement learning is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal.
<img src="rl.png" alt="RL overview" width="80%">

**A complete, interactive goal-seeking agent, explores and learn from enviorment.**

---

## Main subelements of RL system

- Policy: A **mapping** from perceived states of the environment to actions to be taken when in those states.
- Reward: The **signal** that defines the goal of a reinforcement learning problem, indicates what is good in an immediate sense
- **Value function**: Specifies what is **good in the long run**. Action choices are made based on value judgments.
- Model of the environment: allows **inferences** to be made about how the environment will behave.

---
## Jetcar Example
- Agent: the car
- Environment: runway and the rewards it gives: $r \in \mathcal{R}$, 0 for running, -1 for touching obstacles.
- Action: $a \in \mathcal{A}$, throttle and steering control.
- State: $s \in \mathcal{S}$, the position on the runway and the speed of the car $s_t \dot{=} (x, y, v_x, v_y)$.
- Policy: $\pi(a|s)$, what the control set should be under a specific state $s$.
- Value Function: $v_\pi(s)$, the total reward the car will gain after taking a specific action in state $s$.

---

## Features of reinforcement learning
- trial-and-error search
- delayed reward

## Markov decision Processes
$$
p(s', r|s, a) \dot{=} Pr\{ S_t=s', R_t = r | S_{t-1}=s, A_{t-1} = a \}
$$

**MDPs** model sequential decision making in which actions influence **immediate rewards** and **future states**.

---

## Bellman Equation
$$
v_\pi(s) \dot{=} \sum_a{\pi(a|s)} \sum_{s', r}p(s', r|s, a)[r+\gamma v_\pi(s')]
$$

---

<img src="Gemini_Generated_Image_axbqhpaxbqhpaxbq.png">

---

# Model-Based algorithms
## Dynamic Prohramming


---

# Model-Free algorithms

## Value-Based

---

## Policy-Based

---

### Q-Learning

---



---