
# GACRL: Graph-Augmented Cooperative Reinforcement Learning

This repository contains the official implementation of **Graph-Augmented Cooperative Reinforcement Learning (GACRL)**, a novel approach to multi-agent reinforcement learning. GACRL combines **Variational Autoencoders (VAE)** for observation encoding, **Graph Neural Networks (GNNs)** for inter-agent communication, and a shared **policy network** for coordinated action selection.

The implementation is benchmarked against standard multi-agent RL algorithms: **QMIX**, **MAPPO**, and **IPPO**, using the `simple_spread_v3` environment from the `pettingzoo[mpe]` suite.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `GACRL.py` | Trains and evaluates the GACRL algorithm with performance visualization. |
| `comparison.py` | Trains GACRL, QMIX, MAPPO, and IPPO and compares their performance metrics. |
| `requirements.txt` | Lists all Python dependencies required to run the code. |

---

## ✅ Prerequisites

- Python 3.8 or higher  
- Compatible environment with packages listed in `requirements.txt`  
- (Optional) CUDA-enabled GPU for faster training

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Team5-code/GACRL.git
cd GACRL
````

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🔹 Running GACRL Only

Trains GACRL and visualizes key metrics.

```bash
python GACRL.py
```

**Output:**

* Progress printed every 50 episodes.
* Metrics: shaped reward, raw reward, agent collisions, distances, and landmark coverage.
* Final metrics over last 50 episodes printed.
* Saved plot: `custom_algorithm_metrics.png`

---

### 🔹 Running Algorithm Comparison

Trains and compares **GACRL**, **QMIX**, **MAPPO**, and **IPPO**.

```bash
python comparison.py
```

**Output:**

* Training metrics printed every 50 episodes for each algorithm.
* Final performance metrics printed at the end.
* Saved plot: `algorithm_comparison.png`

---

## 📊 Metrics Tracked

* **Shaped Reward**
* **Raw Reward**
* **Landmark Coverage**
* **Minimum Landmark Distance**
* **Average Agent Distance**
* **Number of Collisions**

Plots are smoothed using a 10-episode moving average.

---

## ⚙️ Environment Details

* Environment: `simple_spread_v3` from `pettingzoo[mpe]`
* Configuration: 3 agents, 75 max steps per episode, discrete action space
* Episodes: 1500 per run
* Learning Rate & Exploration: Linearly decaying `epsilon` and `tau`
* Hardware: Automatically uses GPU if available (`torch.cuda.is_available()`)

---

## 📦 Key Dependencies

* `torch`: Deep learning backend
* `numpy`: Numerical computations
* `matplotlib`: For plotting
* `pettingzoo[mpe]`: Multi-agent environments

Full list in [`requirements.txt`](./requirements.txt)

---

## 🛠 Troubleshooting

| Issue                 | Solution                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again.                                                                       |
| CUDA errors           | Ensure CUDA-compatible PyTorch is installed, or change `device` to CPU.                                            |
| Env install issues    | Verify `pettingzoo[mpe]` supports your Python version. See [PettingZoo docs](https://www.pettingzoo.ml/) for help. |

---

## 🤝 Contributing

We welcome contributions!

* Fork the repository
* Create a new branch
* Submit a Pull Request
* Or open an Issue for discussion

GitHub: [https://github.com/Team5-code/GACRL](https://github.com/Team5-code/GACRL)

---

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.
