import os
import json
import copy
import multiprocessing

import pennylane as qml
from pennylane import numpy as np

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
# 汎用ハイブリッド量子AI クラス
# ======================================================
class HybridQuantumClassifier:
    def __init__(self, config):
        self.input_dim   = config["data"]["input_dimensions"]
        self.output_classes = config["data"].get("output_classes", 2)
        self.n_qubits    = config["quantum_model"]["n_qubits"]
        self.num_layers  = config["quantum_model"]["num_layers"]
        self.hidden_nodes = config["classical_model"]["hidden_nodes"]

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights = None

        # --- パラメータ数の自動計算 ---
        self.q_param_count = self.n_qubits * self.num_layers
        self.q_output_dim  = self.n_qubits * 2  # PauliZ + PauliX

        # 隠れ層: (q_output_dim + 1バイアス) × hidden_nodes
        hidden_params = (self.q_output_dim + 1) * self.hidden_nodes
        # 出力層: (hidden_nodes + 1バイアス) × output_classes
        output_params = (self.hidden_nodes + 1) * self.output_classes

        self.c_param_count = hidden_params + output_params
        self.total_params  = self.q_param_count + self.c_param_count

        self.qnode = qml.QNode(self._quantum_circuit, self.dev)

    def _quantum_circuit(self, inputs, q_weights):
        for layer in range(self.num_layers):
            if layer == 0:
                for j in range(self.input_dim):
                    qubit = j % self.n_qubits
                    cycle = j // self.n_qubits
                    # 全cycleへ共有重みを適用（ブラインドスポットの解消）
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
        raw_tensor      = qml.math.stack(raw_values)
        scaled          = raw_tensor / step_size
        scaled_unwrapped = qml.math.unwrap(scaled)
        scaled_detached = np.array(scaled_unwrapped, requires_grad=False)
        rounded_detached = np.round(scaled_detached)
        return (rounded_detached - scaled_detached) * step_size + raw_tensor

    def _softmax(self, logits):
        stacked = qml.math.stack(logits)
        e = qml.math.exp(stacked - qml.math.max(stacked))
        return e / (qml.math.sum(e) + 1e-12)

    def _neural_network(self, inputs, weights, step_size):
        q_weights   = weights[0:self.q_param_count]
        raw_outputs = self.qnode(inputs, q_weights)
        translated  = self._translation_layer(raw_outputs, step_size)

        offset = self.q_param_count

        # 隠れ層
        hidden_out = []
        for i in range(self.hidden_nodes):
            idx = offset + i * (self.q_output_dim + 1)
            h = sum(weights[idx + j] * translated[j] for j in range(self.q_output_dim)) \
                + weights[idx + self.q_output_dim]
            hidden_out.append(np.tanh(h))

        # 出力層
        out_offset = offset + self.hidden_nodes * (self.q_output_dim + 1)
        logits = []
        for c in range(self.output_classes):
            idx = out_offset + c * (self.hidden_nodes + 1)
            logit = sum(weights[idx + k] * hidden_out[k] for k in range(self.hidden_nodes)) \
                    + weights[idx + self.hidden_nodes]
            logits.append(logit)

        return logits

    def _cross_entropy_cost(self, weights, dataset, step_size):
        total_loss = 0.0
        for inputs, target in dataset:
            logits = self._neural_network(inputs, weights, step_size)
            probs = self._softmax(logits)
            t = int(target)
            total_loss = total_loss + (-qml.math.log(probs[t] + 1e-12))
        return total_loss / len(dataset)

    def evaluate(self, test_data, step_size):
        correct = 0
        for inputs, target in test_data:
            logits    = self._neural_network(inputs, self.weights, step_size)
            logits_arr = np.array([float(l) for l in logits])
            pred_class = int(np.argmax(logits_arr))
            if pred_class == int(target):
                correct += 1
        return correct / len(test_data)

    def predict(self, inputs, step_size):
        if self.weights is None:
            raise ValueError("モデルの重みが設定されていません。")
        logits    = self._neural_network(inputs, self.weights, step_size)
        logits_arr = np.array([float(l) for l in logits])
        e = np.exp(logits_arr - np.max(logits_arr))
        probs = e / (np.sum(e) + 1e-12)
        return int(np.argmax(probs)), probs

# ======================================================
# 内部ワーカープロセス
# ======================================================
def _run_parallel_worker(step_size, config, train_data, test_data,
                         initial_weights, shared_dict, timestamp, log_dir):
    os.environ["OMP_NUM_THREADS"] = "1"

    h_set         = config["hunter_settings"]
    target_epochs = h_set["max_epochs"]
    log_int       = h_set.get("log_interval", 5)
    lr            = h_set["learning_rate"]
    reversal_limit = h_set.get("cost_reversal_limit", 3)

    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"hunter_{timestamp}_step_{step_size}.log")

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"======================================================\n")
        f.write(f" 量子ハンター・ワーカープロセス (Step Size: {step_size})\n")
        f.write(f"======================================================\n\n")

        classifier = HybridQuantumClassifier(config)
        classifier.weights = np.array(initial_weights, requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=lr)

        prev_cost           = float('inf')
        cost_reversal_count = 0
        
        best_weights  = copy.deepcopy(classifier.weights)
        best_accuracy = 0.0
        best_epoch    = 0

        state = {
            "epoch": 0, "cost": 0.0, "status": "RUNNING",
            "best_acc": 0.0, "best_epoch": 0, "best_weights": best_weights.tolist()
        }
        shared_dict[step_size] = state

        for epoch in range(target_epochs):
            classifier.weights, cost = opt.step_and_cost(
                lambda w: classifier._cross_entropy_cost(w, train_data, step_size),
                classifier.weights
            )
            cost = float(cost)

            state["epoch"] = epoch + 1
            state["cost"]  = cost

            # ログ出力間隔のタイミングでのみ判定（細かいブレを平滑化）
            if (epoch + 1) % log_int == 0:
                current_acc = classifier.evaluate(test_data, step_size)

                if current_acc > best_accuracy:
                    best_accuracy = current_acc
                    best_epoch    = epoch + 1
                    best_weights  = copy.deepcopy(classifier.weights)
                    state["best_acc"]     = best_accuracy
                    state["best_epoch"]   = best_epoch
                    state["best_weights"] = best_weights.tolist()

                # Cost逆行判定（キルロジック）
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

# ======================================================
# 外部呼び出し用 メイン並列学習マネージャー
# ======================================================
def launch_quantum_hunt(config, X_train, y_train, X_test, y_test, log_dir="logs"):
    import datetime

    train_data = [(np.array(x, requires_grad=False), int(y))
                  for x, y in zip(X_train, y_train)]
    test_data  = [(np.array(x, requires_grad=False), int(y))
                  for x, y in zip(X_test, y_test)]

    dummy_model = HybridQuantumClassifier(config)
    np.random.seed(config.get("system", {}).get("random_seed", 42))
    initial_weights = list(np.random.uniform(-0.5, 0.5, size=dummy_model.total_params))

    timestamp       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    step_candidates = config["hunter_settings"]["step_candidates"]

    manager     = multiprocessing.Manager()
    shared_dict = manager.dict()
    processes   = []

    for step in step_candidates:
        p = multiprocessing.Process(
            target=_run_parallel_worker,
            args=(step, config, train_data, test_data,
                  initial_weights, shared_dict, timestamp, log_dir)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    best_overall_acc = 0.0
    best_step        = None
    best_weights_out = None

    for step in step_candidates:
        result = shared_dict.get(step)
        if result and result["best_acc"] > best_overall_acc:
            best_overall_acc  = result["best_acc"]
            best_step         = step
            best_weights_out  = np.array(result["best_weights"], requires_grad=True)

    print(f"=> 🏆 [BEST MATCH] Step: {best_step} | Acc: {best_overall_acc*100:.2f}%")
    return best_weights_out, best_step
