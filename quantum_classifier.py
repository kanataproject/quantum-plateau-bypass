import pennylane as qml
from pennylane import numpy as np

class HybridQuantumClassifier:
    def __init__(self, n_qubits=4, step_candidates=None):
        if step_candidates is None:
            self.step_candidates = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
        else:
            self.step_candidates = step_candidates
            
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.best_step = None
        self.weights = None
        
        self.qnode = qml.QNode(self._quantum_circuit, self.dev)

    def _quantum_circuit(self, inputs, q_weights):
        for i in range(self.n_qubits):
            qml.RY(inputs[i] + q_weights[i], wires=i)
            
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[1, 2])
        
        z_meas = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        x_meas = [qml.expval(qml.PauliX(i)) for i in range(self.n_qubits)]
        return z_meas + x_meas

    def _translation_layer(self, raw_values, step_size):
        # 1. 勾配追跡を維持したままテンソルにまとめる（np.arrayより安全な公式メソッド）
        raw_tensor = qml.math.stack(raw_values)
        scaled = raw_tensor / step_size
        
        # 2. qml.math.unwrap() を使って Autograd の 'ArrayBox' (追跡箱) を安全に外す
        scaled_unwrapped = qml.math.unwrap(scaled)
        
        # 3. 勾配追跡を持たない純粋な定数配列として丸め処理を行う
        scaled_detached = np.array(scaled_unwrapped, requires_grad=False)
        rounded_detached = np.round(scaled_detached)
        
        # 4. 定数(丸めの差分) + 変数(元のテンソル)
        # 前半部分は完全に定数なので微分で消滅し、raw_tensor の勾配だけが綺麗にスルーされる！
        translated = (rounded_detached - scaled_detached) * step_size + raw_tensor
        return translated

    def _neural_network(self, inputs, weights, step_size):
        q_weights = weights[0:4]
        raw_outputs = self.qnode(inputs, q_weights)
        translated = self._translation_layer(raw_outputs, step_size)

        hidden_nodes = []
        for i in range(4):
            idx = 4 + (i * 9)
            h_in = sum(weights[idx + j] * translated[j] for j in range(8)) + weights[idx + 8]
            hidden_nodes.append(np.tanh(h_in))

        prediction = (weights[40]*hidden_nodes[0]) + (weights[41]*hidden_nodes[1]) + \
                     (weights[42]*hidden_nodes[2]) + (weights[43]*hidden_nodes[3]) + weights[44]
        return prediction

    def _mse_cost(self, weights, dataset, step_size):
        total_error = 0.0
        for inputs, target in dataset:
            pred = self._neural_network(inputs, weights, step_size)
            total_error += (pred - target) ** 2
        return total_error / len(dataset)

    def fit(self, dataset, epochs=150, log_interval=15):
        np.random.seed(42)
        initial_weights = np.random.uniform(-0.5, 0.5, size=45, requires_grad=True)
        
        print("=== [Phase 1] アイドリングによる最適粒度の自動探索 ===")
        best_cost_drop = -float('inf')
        self.best_step = self.step_candidates[0]

        for s in self.step_candidates:
            test_weights = np.copy(initial_weights)
            opt_idle = qml.AdamOptimizer(stepsize=0.1)
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

        print("=== [Phase 2] メイン学習フェーズ（量子回路・古典NNの同時最適化） ===")
        self.weights = np.copy(initial_weights)
        opt_main = qml.AdamOptimizer(stepsize=0.1)

        for epoch in range(epochs):
            self.weights, cost = opt_main.step_and_cost(
                lambda w: self._mse_cost(w, dataset, self.best_step), self.weights
            )
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch + 1:3d} | 平均誤差(Cost): {cost:.6f}")
    
        print("=== 学習完了 ===\n")

    def predict(self, inputs):
        if self.weights is None or self.best_step is None:
            raise ValueError("先にfit()メソッドでモデルを学習させてください。")
        return self._neural_network(inputs, self.weights, self.best_step)
