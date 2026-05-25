import os
import json
import multiprocessing
import copy

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pennylane as qml
from pennylane import numpy as np

# ======================================================
# 設定ファイルロード関数
# ======================================================
def load_config(config_path="config.json"):
    """
    外部から設定ファイルを読み込む。
    存在しない場合はデフォルト構成を返却（ファイル生成はメイン処理側に委ねる）。
    """
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

# ======================================================
# 汎用ハイブリッド量子AI クラス (純粋なモジュール版)
# ======================================================
class HybridQuantumClassifier:
    def __init__(self, config):
        self.input_dim = config["data"]["input_dimensions"]
        self.n_qubits = config["quantum_model"]["n_qubits"]
        self.num_layers = config["quantum_model"]["num_layers"]
        self.hidden_nodes = config["classical_model"]["hidden_nodes"]
        
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights = None
        
        # --- パラメータの自動計算 ---
        # 量子パラメータ: 各層で1Qubitあたり1つの重み
        self.q_param_count = self.n_qubits * self.num_layers
        
        # 古典層の入力次元数 (ZとXの測定結果)
        self.q_output_dim = self.n_qubits * 2
        
        # 古典隠れ層パラメータ: (入力次元数 * ノード数) + ノード数(バイアス)
        hidden_params = (self.q_output_dim * self.hidden_nodes) + self.hidden_nodes
        # 古典出力層パラメータ: (ノード数 * 1) + 1(バイアス)
        output_params = self.hidden_nodes + 1
        
        self.c_param_count = hidden_params + output_params
        self.total_params = self.q_param_count + self.c_param_count
        
        self.qnode = qml.QNode(self._quantum_circuit, self.dev)

    def _quantum_circuit(self, inputs, q_weights):
        for layer in range(self.num_layers):
            if layer == 0:
                # 動的エンコードロジック（入力次元・Qubit数に依存しない）
                for j in range(self.input_dim):
                    qubit = j % self.n_qubits
                    cycle = j // self.n_qubits
                    
                    angle = inputs[j] + q_weights[qubit] if cycle == 0 else inputs[j]
                    
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
        raw_tensor = qml.math.stack(raw_values)
        scaled = raw_tensor / step_size
        scaled_unwrapped = qml.math.unwrap(scaled)
        scaled_detached = np.array(scaled_unwrapped, requires_grad=False)
        rounded_detached = np.round(scaled_detached)
        return (rounded_detached - scaled_detached) * step_size + raw_tensor

    def _neural_network(self, inputs, weights, step_size):
        q_weights = weights[0:self.q_param_count]
        raw_outputs = self.qnode(inputs, q_weights)
        translated = self._translation_layer(raw_outputs, step_size)
        
        offset = self.q_param_count
        hidden_nodes_out = []
        
        # 動的古典NN計算
        for i in range(self.hidden_nodes):
            idx = offset + (i * (self.q_output_dim + 1))
            h_in = sum(weights[idx + j] * translated[j] for j in range(self.q_output_dim)) + weights[idx + self.q_output_dim]
            hidden_nodes_out.append(np.tanh(h_in))
            
        out_idx = offset + (self.hidden_nodes * (self.q_output_dim + 1))
        prediction = sum(weights[out_idx + k] * hidden_nodes_out[k] for k in range(self.hidden_nodes)) + weights[out_idx + self.hidden_nodes]
        
        return prediction

    def _mse_cost(self, weights, dataset, step_size):
        total_error = 0.0
        for inputs, target in dataset:
            pred = self._neural_network(inputs, weights, step_size)
            total_error += (pred - target) ** 2
        return total_error / len(dataset)

    def evaluate(self, test_data, step_size):
        correct = 0
        for inputs, target in test_data:
            pred = self._neural_network(inputs, self.weights, step_size)
            pred_class = float(np.round(pred))  # ← ★四捨五入に変更（これで何クラスでもOK）
            if pred_class == target:
                correct += 1
        return correct / len(test_data)
    
    def predict(self, inputs, step_size):
        if self.weights is None:
            raise ValueError("モデルの重みが設定されていません。")
        return self._neural_network(inputs, self.weights, step_size)

# ======================================================
# 内部ワーカープロセス (外部から直接呼ばない想定)
# ======================================================
def _run_parallel_worker(step_size, config, train_data, test_data, initial_weights, shared_dict, timestamp, log_dir):
    os.environ["OMP_NUM_THREADS"] = "1"
    
    h_set = config["hunter_settings"]
    target_epochs = h_set["max_epochs"]
    patience = h_set["patience_limit"]
    cost_tol = h_set["cost_tolerance"]
    log_int = h_set.get("log_interval", 15)
    lr = h_set["learning_rate"]
    
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"hunter_{timestamp}_step_{step_size}.log")
    
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"======================================================\n")
        f.write(f" 量子ハンター・ワーカープロセス (Step Size: {step_size})\n")
        f.write(f"======================================================\n\n")
        
        classifier = HybridQuantumClassifier(config)
        classifier.weights = np.array(initial_weights, requires_grad=True)
    
        opt_main = qml.AdamOptimizer(stepsize=lr)
        
        prev_cost = float('inf')
        best_weights = copy.deepcopy(classifier.weights)
        best_accuracy = 0.0
        best_epoch = 0
        stagnation_count = 0
    
        state = {
            "epoch": 0, "cost": 0.0, "status": "RUNNING",
            "best_acc": 0.0, "best_epoch": 0, "best_weights": best_weights.tolist()
        }
        shared_dict[step_size] = state
    
        for epoch in range(target_epochs):
            classifier.weights, cost = opt_main.step_and_cost(
                lambda w: classifier._mse_cost(w, train_data, step_size), classifier.weights
            )
            
            state["epoch"] = epoch + 1
            state["cost"] = float(cost)
            
            if (epoch + 1) % log_int == 0:
                current_acc = classifier.evaluate(test_data, step_size)
                
                if current_acc > best_accuracy:
                    best_accuracy = current_acc
                    best_epoch = epoch + 1
                    best_weights = copy.deepcopy(classifier.weights)
                    stagnation_count = 0
                    
                    state["best_acc"] = best_accuracy
                    state["best_epoch"] = best_epoch
                    state["best_weights"] = best_weights.tolist()
                else:
                    stagnation_count += 1
                
                log_line = (
                    f"Epoch {epoch + 1:3d} | Cost: {cost:.6f} | "
                    f"Acc: {current_acc*100:.2f}% (Best: {best_accuracy*100:.2f}% @ Ep{best_epoch}) | "
                    f"Stag: {stagnation_count}/{patience}\n"
                )
                f.write(log_line)
                f.flush()
                
                if stagnation_count >= patience:
                    state["status"] = "KILL (STAG)"
                    shared_dict[step_size] = state
                    return
            
                if (prev_cost - cost) <= cost_tol:
                    state["status"] = "KILL (LOOP)"
                    shared_dict[step_size] = state
                    return
                    
                prev_cost = float(cost)
                
            shared_dict[step_size] = state
    
        state["status"] = "GOAL"
        shared_dict[step_size] = state

# ======================================================
# 外部呼び出し用 メイン並列学習マネージャー
# ======================================================
def launch_quantum_hunt(config, X_train, y_train, X_test, y_test, log_dir="logs"):
    """
    外部スクリプトから呼び出して並列ハンティングを実行する関数。
    学習完了後、最も精度の高かったモデルの重みとStep Sizeを返す。
    """
    import datetime
    
    # テンソル化
    train_data = [(np.array(x, requires_grad=False), y) for x, y in zip(X_train, y_train)]
    test_data = [(np.array(x, requires_grad=False), y) for x, y in zip(X_test, y_test)]
    
    dummy_model = HybridQuantumClassifier(config)
    np.random.seed(config.get("system", {}).get("random_seed", 42))
    initial_weights = list(np.random.uniform(-0.5, 0.5, size=dummy_model.total_params))
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    step_candidates = config["hunter_settings"]["step_candidates"]
    
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    processes = []

    print(f"[INFO] 量子ハンター起動: {len(step_candidates)} 粒度プロセスをデプロイ")
    
    for step in step_candidates:
        p = multiprocessing.Process(
            target=_run_parallel_worker,
            args=(step, config, train_data, test_data, initial_weights, shared_dict, timestamp, log_dir)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[INFO] 全プロセスの学習が完了しました。最適モデルを抽出します。")
    
    best_overall_acc = 0.0
    best_step = None
    best_weights = None
    
    for step in step_candidates:
        result = shared_dict.get(step)
        if result and result["best_acc"] > best_overall_acc:
            best_overall_acc = result["best_acc"]
            best_step = step
            best_weights = np.array(result["best_weights"], requires_grad=True)

    print(f"=> 🏆 [BEST MATCH] Step: {best_step} | Acc: {best_overall_acc*100:.2f}%")
    return best_weights, best_step
