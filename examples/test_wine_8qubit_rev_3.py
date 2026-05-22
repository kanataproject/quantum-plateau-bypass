import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine

print("======================================================")
print("  完全ハイブリッド量子AI - [ワイン13次元：第2版 多軸ハイブリッドエンコード]")
print("======================================================\n")

class WineQuantumClassifier8Qubit_V2:
    def __init__(self, n_qubits=8, num_layers=4, step_candidates=None):
        if step_candidates is None:
            self.step_candidates = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
        else:
            self.step_candidates = step_candidates
            
        self.n_qubits = n_qubits
        self.num_layers = num_layers 
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.best_step = None
        self.weights = None
        
        # 8 qubits * 4 layers = 32量子パラメータ
        self.q_param_count = self.n_qubits * self.num_layers
        # 古典NN: 16入力(8量子ビットのZ/X測定) -> 4隠れノード -> 1出力
        self.c_param_count = 73
        self.total_params = self.q_param_count + self.c_param_count
        
        self.qnode = qml.QNode(self._quantum_circuit, self.dev)

    def _quantum_circuit(self, inputs, q_weights):
        for layer in range(self.num_layers):
            for i in range(self.n_qubits):
                weight_idx = layer * self.n_qubits + i
                if layer == 0:
                    # 【第2版：多軸エンコード】
                    # inputsは13次元の生データ。足し算(相殺)を一切せず、RYとRZに独立配置
                    
                    # 0〜7次元目を全QubitのRYへ配置
                    qml.RY(inputs[i] + q_weights[weight_idx], wires=i)
                    
                    # 残りの5次元(8〜12次元目)をQubit 0〜4のRZへ配置
                    if i < 5:
                        qml.RZ(inputs[i + 8], wires=i)
                else:
                    qml.RY(q_weights[weight_idx], wires=i)
            
            # 8 Qubit全体を強力に繋ぐリングトポロジーもつれ
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
        # STE (Straight-Through Estimator)
        return (rounded_detached - scaled_detached) * step_size + raw_tensor

    def _neural_network(self, inputs, weights, step_size):
        q_weights = weights[0:self.q_param_count]
        raw_outputs = self.qnode(inputs, q_weights)
        translated = self._translation_layer(raw_outputs, step_size)

        offset = self.q_param_count
        hidden_nodes = []
        for i in range(4):
            idx = offset + (i * 17) 
            h_in = sum(weights[idx + j] * translated[j] for j in range(16)) + weights[idx + 16]
            hidden_nodes.append(np.tanh(h_in))

        out_idx = offset + 68
        # エラーになっていた構文を修正済み
        prediction = (weights[out_idx]*hidden_nodes[0]) + (weights[out_idx+1]*hidden_nodes[1]) + \
                     (weights[out_idx+2]*hidden_nodes[2]) + (weights[out_idx+3]*hidden_nodes[3]) + weights[out_idx+4]
        return prediction

    def _mse_cost(self, weights, dataset, step_size):
        total_error = 0.0
        for inputs, target in dataset:
            pred = self._neural_network(inputs, weights, step_size)
            total_error += (pred - target) ** 2
        return total_error / len(dataset)

    def fit(self, train_data, epochs=150, log_interval=15):
        np.random.seed(42)
        initial_weights = np.random.uniform(-0.5, 0.5, size=self.total_params, requires_grad=True)
        
        print(f"=== [Phase 1] アイドリングによる最適粒度の自動探索 (Total Params: {self.total_params}) ===")
        best_cost_drop = -float('inf')
        self.best_step = self.step_candidates[0]
        
        for s in self.step_candidates:
            test_weights = np.copy(initial_weights)
            opt_idle = qml.AdamOptimizer(stepsize=0.05) 
            initial_cost = self._mse_cost(test_weights, train_data, s)
            
            for _ in range(3):
                test_weights, current_cost = opt_idle.step_and_cost(
                    lambda w: self._mse_cost(w, train_data, s), test_weights
                )
            cost_drop = initial_cost - current_cost
            print(f"粒度 {s:<5} | 探索での誤差減少量: {cost_drop:.6f}")
            if cost_drop > best_cost_drop:
                best_cost_drop = cost_drop
                self.best_step = s
                
        print(f"=> 最適粒度を【 {self.best_step} 】にロックオン。\n")

        print("=== [Phase 2] ワイン実データ メイン学習フェーズ ===")
        self.weights = np.copy(initial_weights)
        opt_main = qml.AdamOptimizer(stepsize=0.05)
        grad_fn = qml.grad(self._mse_cost, argnum=0)

        for epoch in range(epochs):
            if (epoch + 1) % log_interval == 0:
                grads = grad_fn(self.weights, train_data, self.best_step)
                q_grad_norm = np.linalg.norm(grads[0:self.q_param_count])
            
            self.weights, cost = opt_main.step_and_cost(
                lambda w: self._mse_cost(w, train_data, self.best_step), self.weights
            )
            
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch + 1:3d} | 平均誤差: {cost:.6f} | 量子勾配ノルム: {q_grad_norm:.8f}")

    def evaluate(self, test_data):
        correct = 0
        for inputs, target in test_data:
            pred = self._neural_network(inputs, self.weights, self.best_step)
            pred_binary = 1.0 if pred >= 0.5 else 0.0
            if pred_binary == target:
                correct += 1
        accuracy = correct / len(test_data)
        return accuracy

if __name__ == "__main__":
    wine = load_wine()
    X = wine.data  
    y = wine.target
    
    y_binary = np.array([1.0 if label == 1 else 0.0 for label in y])
    
    # 標準化（13次元のまま）
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    np.random.seed(42)
    indices = np.random.permutation(len(X_scaled))
    # 13次元の生データをそのまま使用する
    X_scaled, y_binary = X_scaled[indices], y_binary[indices]
    
    formatted_data = [(np.array(X_scaled[i], requires_grad=False), y_binary[i]) for i in range(len(X_scaled))]
    train_data = formatted_data[:140]
    test_data = formatted_data[140:]

    classifier = WineQuantumClassifier8Qubit_V2(n_qubits=8, num_layers=4)
    classifier.fit(train_data, epochs=150, log_interval=15)

    print("\n=== テストデータ（未知の38件）に対する最終評価 ===")
    acc = classifier.evaluate(test_data)
    print(f"【最終テスト正解率 (Accuracy)】: {acc * 100:.2f} %")
