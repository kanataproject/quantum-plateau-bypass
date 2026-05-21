import pennylane as qml
from pennylane import numpy as np
from quantum_classifier import HybridQuantumClassifier

print("======================================================")
print("  完全ハイブリッド量子AI（STE実装版） - 外部呼び出しテスト")
print("======================================================\n")

if __name__ == "__main__":
    dataset = [
        (np.array([0.8, -0.8, 0.8, -0.8], requires_grad=False), 1.0),
        (np.array([-0.8, 0.8, -0.8, 0.8], requires_grad=False), 1.0),
        (np.array([0.8, 0.8, 0.8, 0.8], requires_grad=False), 0.0),
        (np.array([-0.8, -0.8, -0.8, -0.8], requires_grad=False), 0.0),
        (np.array([0.8, 0.8, -0.8, -0.8], requires_grad=False), 0.0),
        (np.array([0.8, -0.8, -0.8, 0.8], requires_grad=False), 0.0),
    ]

    print("[INFO] HybridQuantumClassifier を初期化中...")
    classifier = HybridQuantumClassifier(step_candidates=[0.2, 0.1, 0.05, 0.01])

    # 学習実行（この内部で、丸め処理をスルーして量子回路まで勾配が逆伝播する）
    classifier.fit(dataset, epochs=150, log_interval=15)
    print("q_weights（量子側）:", classifier.weights[0:4])

    print("\n=== 未知の複雑な波形データでの推論テスト ===")
    test_data_A = np.array([0.9, -0.7, 0.85, -0.9], requires_grad=False)
    test_data_B = np.array([-0.9, -0.8, 0.85, 0.9], requires_grad=False)
    test_data_C = np.array([0.8, -0.8, -0.8, 0.8], requires_grad=False)

    print(f"テストA (未知のジグザグ) の推論結果: {classifier.predict(test_data_A):.6f}")
    print(f"テストB (未知の階段波)   の推論結果: {classifier.predict(test_data_B):.6f}")
    print(f"テストC (未知のＵ字波)   の推論結果: {classifier.predict(test_data_C):.6f}")
