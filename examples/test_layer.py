import pennylane as qml
from pennylane import numpy as np

print("======================================================")
print("  完全ハイブリッド量子AI - [純粋検証：Data Re-uploading排除版]")
print("======================================================\n")

class PureDeepHybridClassifier:
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
        
        self.q_param_count = self.n_qubits * self.num_layers
        self.c_param_count = 73
        self.total_params = self.q_param_count + self.c_param_count
        
        self.qnode = qml.QNode(self._quantum_circuit, self.dev)

    def _quantum_circuit(self, inputs, q_weights):
        for layer in range(self.num_layers):
            for i in range(self.n_qubits):
                weight_idx = layer * self.n_qubits + i
                
                if layer == 0:
                    # 【修正】第1層のみ、データ(inputs)と重みを合成してエンコード
                    qml.RY(inputs[i] + q_weights[weight_idx], wires=i)
                else:
                    # 【修正】第2層以降は純粋な重みのみ（Data Re-uploadingを完全排除）
                    qml.RY(q_weights[weight_idx], wires=i)
            
            # CNOTによる強烈なもつれは各層で維持
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
        
        translated = (rounded_detached - scaled_detached) * step_size + raw_tensor
        return translated

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
        prediction = (weights[out_idx]*hidden_nodes[0]) + (weights[out_idx+1]*hidden_nodes[1]) + \
                     (weights[out_idx+2]*hidden_nodes[2]) + (weights[out_idx+3]*hidden_nodes[3]) + weights[out_idx+4]
        return prediction

    def _mse_cost(self, weights, dataset, step_size):
        total_error = 0.0
        for inputs, target in dataset:
            pred = self._neural_network(inputs, weights, step_size)
            total_error += (pred - target) ** 2
        return total_error / len(dataset)

    def fit(self, dataset, epochs=150, log_interval=15):
        np.random.seed(42)
        initial_weights = np.random.uniform(-0.5, 0.5, size=self.total_params, requires_grad=True)
        
        print("=== [Phase 1] アイドリングによる最適粒度の自動探索 ===")
        best_cost_drop = -float('inf')
        self.best_step = self.step_candidates[0]
        
        for s in self.step_candidates:
            test_weights = np.copy(initial_weights)
            opt_idle = qml.AdamOptimizer(stepsize=0.05) 
            initial_cost = self._mse_cost(test_weights, dataset, s)
            
            for _ in range(5):
                test_weights, current_cost = opt_idle.step_and_cost(
                    lambda w: self._mse_cost(w, dataset, s), test_weights
                )
                
            cost_drop = initial_cost - current_cost
            print(f"粒度 {s:<5} | 5Epoch後の誤差減少量: {cost_drop:.6f}")

            if cost_drop > best_cost_drop:
                best_cost_drop = cost_drop
                self.best_step = s
                
        print(f"=> 最適粒度を【 {self.best_step} 】にロックオン。\n")

        print("=== [Phase 2] 純粋深層 メイン学習フェーズ（勾配ノルム監視） ===")
        self.weights = np.copy(initial_weights)
        opt_main = qml.AdamOptimizer(stepsize=0.05)
        
        grad_fn = qml.grad(self._mse_cost, argnum=0)

        for epoch in range(epochs):
            if (epoch + 1) % log_interval == 0:
                grads = grad_fn(self.weights, dataset, self.best_step)
                q_grad_norm = np.linalg.norm(grads[0:self.q_param_count])
            
            self.weights, cost = opt_main.step_and_cost(
                lambda w: self._mse_cost(w, dataset, self.best_step), self.weights
            )
            
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch + 1:3d} | 平均誤差: {cost:.6f} | 量子勾配ノルム: {q_grad_norm:.8f}")
                
        print("\n=== 学習完了 ===")

    def predict(self, inputs):
        if self.weights is None or self.best_step is None:
            raise ValueError("先にfit()メソッドで学習させてください。")
        return self._neural_network(inputs, self.weights, self.best_step)


if __name__ == "__main__":
    dataset = [
        (np.array([0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8], requires_grad=False), 1.0),
        (np.array([-0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8], requires_grad=False), 1.0),
        (np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], requires_grad=False), 0.0),
        (np.array([-0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8], requires_grad=False), 0.0),
        (np.array([0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8], requires_grad=False), 0.0),
        (np.array([0.8, 0.4, -0.2, -0.8, -0.8, -0.2, 0.4, 0.8], requires_grad=False), 0.0),
    ]

    classifier = PureDeepHybridClassifier(n_qubits=8, num_layers=4, step_candidates=[0.2, 0.1, 0.05, 0.01])
    classifier.fit(dataset, epochs=150, log_interval=15)

    print("=== 未知の 8次元データ での推論テスト ===")
    test_data_A = np.array([0.9, -0.7, 0.85, -0.9, 0.8, -0.8, 0.9, -0.7], requires_grad=False)
    test_data_B = np.array([-0.9, -0.9, -0.8, -0.8, 0.8, 0.8, 0.9, 0.9], requires_grad=False)
    print(f"テストA: {classifier.predict(test_data_A):.6f}")
    print(f"テストB: {classifier.predict(test_data_B):.6f}")
