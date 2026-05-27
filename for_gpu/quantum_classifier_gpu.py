import os
import json
import copy
import multiprocessing
import datetime
import concurrent.futures
import traceback
import sys

import pennylane as qml
from pennylane import numpy as np

import torch
import torch.nn as nn

# ======================================================
# 設定ファイルロード関数
# ======================================================
def load_config(config_path="config.json"):
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

# ======================================================
# GPU古典層 (PyTorch nn.Module)
# ======================================================
class ClassicalNetwork(nn.Module):
    def __init__(self, q_output_dim, hidden_nodes, output_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(q_output_dim, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, output_classes)
        )

    def forward(self, x):
        return self.net(x)

# ======================================================
# 汎用ハイブリッド量子AIクラス (完全PyTorch統合版)
# ======================================================
class HybridQuantumClassifier(nn.Module):
    def __init__(self, config, initial_q_weights=None):
        super().__init__()
        self.input_dim      = config["data"]["input_dimensions"]
        self.output_classes = config["data"].get("output_classes", 2)
        self.n_qubits       = config["quantum_model"]["n_qubits"]
        self.num_layers     = config["quantum_model"]["num_layers"]
        self.hidden_nodes   = config["classical_model"]["hidden_nodes"]
        self.batch_size     = config["classical_model"].get("batch_size", 32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # C++バックエンド
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
        
        self.q_param_count = self.n_qubits * self.num_layers
        self.q_output_dim  = self.n_qubits * 2

        if initial_q_weights is not None:
            self.q_weights = nn.Parameter(torch.tensor(initial_q_weights, dtype=torch.float32))
        else:
            self.q_weights = nn.Parameter(torch.rand(self.q_param_count, dtype=torch.float32) - 0.5)

        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface="torch", diff_method="parameter-shift")
        
        self.classical_net = ClassicalNetwork(
            self.q_output_dim, self.hidden_nodes, self.output_classes
        ).to(self.device)

    def _quantum_circuit(self, inputs, q_weights):
        for layer in range(self.num_layers):
            if layer == 0:
                for j in range(self.input_dim):
                    qubit = j % self.n_qubits
                    cycle = j // self.n_qubits
                    angle = inputs[j] + q_weights[qubit]
                    if cycle % 3 == 0:
                        qml.RY(angle, wires=qubit)
                    elif cycle % 3 == 1:
                        qml.RZ(angle, wires=qubit)
                    else:
                        qml.RX(angle, wires=qubit)
            else:
                for i in range(self.n_qubits):
                    weight_idx = layer * self.n_qubits + i
                    qml.RY(q_weights[weight_idx], wires=i)

            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        z_meas = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        x_meas = [qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)]
        return z_meas + x_meas

    def _translation_layer(self, raw_values, step_size):
        raw_tensor = torch.stack(raw_values).to(torch.float32)
        scaled     = raw_tensor / step_size
        rounded    = torch.round(scaled)
        return (rounded.detach() - scaled.detach()) * step_size + raw_tensor

    def _quantum_features_batch(self, inputs_batch, step_size):
        features = []
        for inputs in inputs_batch:
            raw   = self.qnode(inputs, self.q_weights)
            trans = self._translation_layer(raw, step_size)
            features.append(trans)
        # 【修正】確実に float32 に変換してからGPU/CPUに渡す
        return torch.stack(features).to(dtype=torch.float32, device=self.device)

    def evaluate(self, test_data, step_size):
        self.classical_net.eval()
        correct = 0
        with torch.no_grad():
            inputs_list  = [torch.tensor(x, dtype=torch.float32) for x, _ in test_data]
            targets_list = [y for _, y in test_data]
            feat   = self._quantum_features_batch(inputs_list, step_size)
            logits = self.classical_net(feat)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            for p, t in zip(preds, targets_list):
                if int(p) == int(t):
                    correct += 1
        return correct / len(test_data)

# ======================================================
# 内部ワーカープロセス (完全エラー捕捉版)
# ======================================================
def _run_parallel_worker(step_size, config, train_data, test_data,
                         initial_q_weights, shared_dict, timestamp, log_dir):
    print(f"\n🚀 [STEP {step_size}] WORKER PROCESS START!")
    try:
        print(f"  -> [{step_size}] 1. Environment Setup...")
        os.environ["OMP_NUM_THREADS"] = "1"

        h_set          = config["hunter_settings"]
        target_epochs  = h_set["max_epochs"]
        log_int        = h_set.get("log_interval", 5)
        lr_q           = h_set["learning_rate"]
        lr_c           = h_set.get("learning_rate_classical", lr_q)
        reversal_limit = h_set.get("cost_reversal_limit", 3)
        batch_size     = config["classical_model"].get("batch_size", 32)

        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"hunter_{timestamp}_step_{step_size}.log")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  -> [{step_size}] 2. Device set to: {device}")

        with open(log_filename, "w", encoding="utf-8") as f:
            f.write(f"======================================================\n")
            f.write(f" 量子ハンター (完全PyTorch統合版) Step: {step_size}\n")
            f.write(f" Device: {device}\n")
            f.write(f"======================================================\n\n")

            print(f"  -> [{step_size}] 3. Initializing Models...")
            classifier = HybridQuantumClassifier(config, initial_q_weights)

            print(f"  -> [{step_size}] 4. Setting up Optimizers...")
            c_opt   = torch.optim.Adam(classifier.classical_net.parameters(), lr=lr_c)
            q_opt   = torch.optim.Adam([classifier.q_weights], lr=lr_q)
            ce_loss = nn.CrossEntropyLoss()

            print(f"  -> [{step_size}] 5. Pre-allocating states (Safe Memory Copy)...")
            prev_cost           = float('inf')
            cost_reversal_count = 0
            best_q_weights      = classifier.q_weights.detach().cpu().numpy().copy()
            # 【修正】CUDAからの安全なディープコピー
            best_c_state        = {k: v.cpu().clone() for k, v in classifier.classical_net.state_dict().items()}
            best_accuracy       = 0.0
            best_epoch          = 0

            state = {
                "epoch": 0, "cost": 0.0, "status": "RUNNING",
                "best_acc": 0.0, "best_epoch": 0,
                "current_batch": 0, "total_batches": 0,
                "best_q_weights": best_q_weights.tolist()
            }
            
            print(f"  -> [{step_size}] 6. Pushing RUNNING state to GUI...")
            shared_dict[step_size] = state

            print(f"  -> [{step_size}] 7. Converting data to Tensors...")
            all_inputs  = [torch.tensor(x, dtype=torch.float32) for x, _ in train_data]
            all_targets = [y for _, y in train_data]
            n_train     = len(all_inputs)

            print(f"  -> [{step_size}] === ALL CLEAR: STARTING EPOCH LOOP ===")

            for epoch in range(target_epochs):
                classifier.classical_net.train()
                epoch_loss = 0.0
                n_batches  = 0
                
                total_batches = (n_train + batch_size - 1) // batch_size
                state["total_batches"] = total_batches

                for batch_start in range(0, n_train, batch_size):
                    current_batch = (batch_start // batch_size) + 1
                    state["current_batch"] = current_batch
                    shared_dict[step_size] = state

                    batch_inputs  = all_inputs[batch_start:batch_start + batch_size]
                    batch_targets = all_targets[batch_start:batch_start + batch_size]
                    targets_t     = torch.tensor(batch_targets, dtype=torch.long, device=device)

                    with torch.no_grad():
                        feat_detached = classifier._quantum_features_batch(batch_inputs, step_size)
                    
                    c_opt.zero_grad()
                    logits_c = classifier.classical_net(feat_detached)
                    loss_c   = ce_loss(logits_c, targets_t)
                    loss_c.backward()
                    c_opt.step()

                    q_sample_size = min(16, len(batch_inputs))
                    q_inputs      = batch_inputs[:q_sample_size]
                    q_targets_t   = targets_t[:q_sample_size]

                    q_opt.zero_grad()
                    feat_q   = classifier._quantum_features_batch(q_inputs, step_size)
                    logits_q = classifier.classical_net(feat_q)
                    loss_q   = ce_loss(logits_q, q_targets_t)
                    loss_q.backward()
                    q_opt.step()

                    epoch_loss += loss_q.item()
                    n_batches  += 1

                cost = epoch_loss / max(n_batches, 1)
                state["epoch"] = epoch + 1
                state["cost"]  = cost

                if (epoch + 1) % log_int == 0:
                    current_acc = classifier.evaluate(test_data, step_size)

                    if current_acc > best_accuracy:
                        best_accuracy  = current_acc
                        best_epoch     = epoch + 1
                        best_q_weights = classifier.q_weights.detach().cpu().numpy().copy()
                        best_c_state   = {k: v.cpu().clone() for k, v in classifier.classical_net.state_dict().items()}
                        state["best_acc"]       = best_accuracy
                        state["best_epoch"]     = best_epoch
                        state["best_q_weights"] = best_q_weights.tolist()

                    if cost > prev_cost:
                        cost_reversal_count += 1
                    else:
                        cost_reversal_count = 0

                    log_line = (
                        f"Epoch {epoch+1:3d} | Cost: {cost:.6f} | "
                        f"Acc: {current_acc*100:.2f}% "
                        f"(Best: {best_accuracy*100:.2f}% @ Ep{best_epoch}) | "
                        f"Rev: {cost_reversal_count}/{reversal_limit}\n"
                    )
                    f.write(log_line)
                    f.flush()

                    if cost_reversal_count >= reversal_limit:
                        state["status"] = "KILL (LOOP)"
                        shared_dict[step_size] = state
                        return

                    prev_cost = cost

                shared_dict[step_size] = state

            state["status"] = "GOAL"
            shared_dict[step_size] = state

    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"\n💥💥💥 [STEP {step_size}] FATAL CRASH 💥💥💥\n{err_msg}")
        try:
            shared_dict[step_size] = {"status": f"CRASH: {str(e)[:15]}"}
        except:
            pass
        sys.exit(1)

# ======================================================
# 外部呼び出し用 メイン並列学習マネージャー
# ======================================================
def launch_quantum_hunt(config, X_train, y_train, X_test, y_test, log_dir="logs"):
    train_data = [(x, int(y)) for x, y in zip(X_train, y_train)]
    test_data  = [(x, int(y)) for x, y in zip(X_test, y_test)]

    dummy = HybridQuantumClassifier(config)
    np.random.seed(config.get("system", {}).get("random_seed", 42))
    initial_q_weights = list(np.random.uniform(-0.5, 0.5, size=dummy.q_param_count))

    timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    step_candidates = config["hunter_settings"]["step_candidates"]

    manager     = multiprocessing.Manager()
    shared_dict = manager.dict()
    processes   = []

    print(f"[INFO] 量子ハンター起動: {len(step_candidates)} 粒度プロセスをデプロイ")
    print(f"[INFO] Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    for step in step_candidates:
        p = multiprocessing.Process(
            target=_run_parallel_worker,
            args=(step, config, train_data, test_data,
                  initial_q_weights, shared_dict, timestamp, log_dir)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    best_overall_acc = 0.0
    best_step        = None
    best_q_weights   = None

    for step in step_candidates:
        result = shared_dict.get(step)
        if result and result["best_acc"] > best_overall_acc:
            best_overall_acc = result["best_acc"]
            best_step        = step
            best_q_weights   = np.array(result["best_q_weights"])

    print(f"=> 🏆 [BEST MATCH] Step: {best_step} | Acc: {best_overall_acc*100:.2f}%")
    return best_q_weights, best_step
