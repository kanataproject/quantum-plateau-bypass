import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import multiprocessing
import pygame
import datetime

# ======================================================
# 量子AI クラス本体
# ======================================================
class WineQuantumClassifier8Qubit_V3:
    def __init__(self, n_qubits=8, num_layers=4):
        self.n_qubits = n_qubits
        self.num_layers = num_layers 
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
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
                    qml.RY(inputs[i] + q_weights[weight_idx], wires=i)
                    if i < 5:
                        qml.RZ(inputs[i + 8], wires=i)
                else:
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

    def evaluate(self, test_data, step_size):
        correct = 0
        for inputs, target in test_data:
            pred = self._neural_network(inputs, self.weights, step_size)
            pred_binary = 1.0 if pred >= 0.5 else 0.0
            if pred_binary == target:
                correct += 1
        return correct / len(test_data)

# ======================================================
# 並列ワーカー (GUIへの送信 ＆ 個別ファイルへのログ書き出し)
# ======================================================
def run_parallel_training(step_size, X_train, y_train, X_test, y_test, initial_weights, epochs, log_interval, shared_dict, timestamp):
    os.environ["OMP_NUM_THREADS"] = "1"
    
    log_filename = f"logs/run_{timestamp}_step_{step_size}.log"
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"======================================================\n")
        f.write(f"  粒度 (Step Size): {step_size}\n")
        f.write(f"  目標 Epoch     : {epochs}\n")
        f.write(f"  開始時刻       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"======================================================\n\n")
        
        train_data = [(np.array(x, requires_grad=False), y) for x, y in zip(X_train, y_train)]
        test_data = [(np.array(x, requires_grad=False), y) for x, y in zip(X_test, y_test)]
        
        classifier = WineQuantumClassifier8Qubit_V3(n_qubits=8, num_layers=4)
        classifier.weights = np.array(initial_weights, requires_grad=True)
    
        opt_main = qml.AdamOptimizer(stepsize=0.05)
        grad_fn = qml.grad(classifier._mse_cost, argnum=0)
        
        prev_cost = float('inf')
        current_grad = 0.0
    
        shared_dict[step_size] = {"epoch": 0, "cost": 0.0, "grad": 0.0, "status": "INITIALIZING..."}
    
        for epoch in range(epochs):
            classifier.weights, cost = opt_main.step_and_cost(
                lambda w: classifier._mse_cost(w, train_data, step_size), classifier.weights
            )
            
            state = {"epoch": epoch + 1, "cost": float(cost), "grad": current_grad, "status": "RUNNING"}
            
            if (epoch + 1) % log_interval == 0:
                grads = grad_fn(classifier.weights, train_data, step_size)
                current_grad = float(np.linalg.norm(grads[0:classifier.q_param_count]))
                state["grad"] = current_grad
                
                log_line = f"Epoch {epoch + 1:3d} | Cost: {cost:.6f} | Q-Grad Norm: {current_grad:.8f}\n"
                f.write(log_line)
                f.flush() 
                
                # 【Kill判定1】絶対閾値(砂漠化)
                if current_grad < 1e-7:
                    # 【修正】死に際にテストデータで評価
                    acc = classifier.evaluate(test_data, step_size)
                    state["status"] = "KILL (DESERT)"
                    shared_dict[step_size] = state
                    f.write(f"\n=> 💀 [KILL] 勾配消失(バレンプラトー)を検知。終了。\n")
                    f.write(f"   [死に際テスト正解率 (Accuracy)]: {acc * 100:.2f} %\n")
                    return
                
                # 【Kill判定2】停滞(ウロウロ)
                if (prev_cost - cost) <= 0.0001:
                    # 【修正】死に際にテストデータで評価
                    acc = classifier.evaluate(test_data, step_size)
                    state["status"] = "KILL (LOOP)"  # ← LOOP表記に完全修正
                    shared_dict[step_size] = state
                    f.write(f"\n=> ⚠️ [KILL] 成長の停滞(LOOP)を検知。終了。\n")
                    f.write(f"   [死に際テスト正解率 (Accuracy)]: {acc * 100:.2f} %\n")
                    return
                
                prev_cost = cost
    
            shared_dict[step_size] = state
    
        # 完走（ゴールイン）
        acc = classifier.evaluate(test_data, step_size)
        state["status"] = "GOAL (SUCCESS)"
        shared_dict[step_size] = state
        f.write(f"\n=> 🏁 [GOAL] {epochs} Epoch 完走。\n")
        f.write(f"   [最終テスト正解率 (Accuracy)]: {acc * 100:.2f} %\n")
    return

# ======================================================
# Pygame GUI メインループ (LOOP表記対応)
# ======================================================
def run_control_panel(X_train, y_train, X_test, y_test, step_candidates, initial_weights, log_interval, shared_dict, timestamp):
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quantum AI Control Panel")
    clock = pygame.time.Clock()

    font_xl = pygame.font.SysFont("consolas", 32, bold=True)
    font_l = pygame.font.SysFont("consolas", 24, bold=True)
    font_m = pygame.font.SysFont("consolas", 18)
    font_s = pygame.font.SysFont("consolas", 14)

    BG_COLOR = (20, 20, 30)
    TEXT_COLOR = (220, 220, 220)
    COLOR_BTN_IDLE = (60, 60, 80)
    COLOR_BTN_HOVER = (80, 80, 100)
    COLOR_START_IDLE = (50, 150, 80)
    COLOR_START_HOVER = (70, 180, 100)
    
    COLOR_RUNNING = (50, 200, 100)
    COLOR_INIT = (150, 150, 150)
    COLOR_KILL_DESERT = (220, 50, 50)
    COLOR_KILL_LOOP = (200, 150, 50) # ウロウロ用のオレンジ
    COLOR_GOAL = (100, 200, 255)

    ui_state = "SETUP"
    target_epochs = 150
    processes = []

    btn_minus = pygame.Rect(300, 250, 50, 40)
    btn_plus = pygame.Rect(450, 250, 50, 40)
    btn_start = pygame.Rect(250, 350, 300, 60)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and ui_state == "SETUP":
                if btn_minus.collidepoint(mouse_pos):
                    target_epochs = max(15, target_epochs - 10)
                elif btn_plus.collidepoint(mouse_pos):
                    target_epochs += 10
                elif btn_start.collidepoint(mouse_pos):
                    ui_state = "RUNNING"
                    for step in step_candidates:
                        p = multiprocessing.Process(
                            target=run_parallel_training,
                            args=(step, X_train, y_train, X_test, y_test, initial_weights, target_epochs, log_interval, shared_dict, timestamp)
                        )
                        p.start()
                        processes.append(p)
                        
        screen.fill(BG_COLOR)
        
        if ui_state == "SETUP":
            title = font_xl.render("QUANTUM AI SURVIVAL TEST", True, TEXT_COLOR)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
            epoch_label = font_l.render("Target Epochs", True, (150, 150, 150))
            screen.blit(epoch_label, (WIDTH//2 - epoch_label.get_width()//2, 200))
            
            pygame.draw.rect(screen, COLOR_BTN_HOVER if btn_minus.collidepoint(mouse_pos) else COLOR_BTN_IDLE, btn_minus, border_radius=5)
            minus_txt = font_l.render("-", True, TEXT_COLOR)
            screen.blit(minus_txt, (btn_minus.centerx - minus_txt.get_width()//2, btn_minus.centery - minus_txt.get_height()//2))
            
            epoch_val = font_xl.render(f"{target_epochs}", True, COLOR_GOAL)
            screen.blit(epoch_val, (WIDTH//2 - epoch_val.get_width()//2, 255))

            pygame.draw.rect(screen, COLOR_BTN_HOVER if btn_plus.collidepoint(mouse_pos) else COLOR_BTN_IDLE, btn_plus, border_radius=5)
            plus_txt = font_l.render("+", True, TEXT_COLOR)
            screen.blit(plus_txt, (btn_plus.centerx - plus_txt.get_width()//2, btn_plus.centery - plus_txt.get_height()//2))

            pygame.draw.rect(screen, COLOR_START_HOVER if btn_start.collidepoint(mouse_pos) else COLOR_START_IDLE, btn_start, border_radius=10)
            start_txt = font_xl.render("START LAUNCH", True, (255, 255, 255))
            screen.blit(start_txt, (btn_start.centerx - start_txt.get_width()//2, btn_start.centery - start_txt.get_height()//2))

        elif ui_state == "RUNNING":
            title = font_l.render("Quantum AI Survival Dashboard [Phase 1]", True, TEXT_COLOR)
            screen.blit(title, (20, 20))
            
            y_offset = 80
            active_processes = 0

            for step in step_candidates:
                state = shared_dict.get(step, {"epoch": 0, "cost": 0.0, "grad": 0.0, "status": "WAITING"})
                
                panel_rect = pygame.Rect(20, y_offset, WIDTH - 40, 60)
                pygame.draw.rect(screen, (40, 40, 50), panel_rect, border_radius=5)
                
                step_text = font_m.render(f"Step: {step:<8}", True, TEXT_COLOR)
                epoch_text = font_m.render(f"Epoch: {state['epoch']:>3} / {target_epochs}", True, TEXT_COLOR)
                cost_text = font_s.render(f"Cost: {state['cost']:.5f} | Grad: {state['grad']:.6f}", True, (180, 180, 180))
                
                status_str = state["status"]
                if "RUNNING" in status_str:
                    status_color = COLOR_RUNNING
                    active_processes += 1
                elif "INITIALIZING" in status_str:
                    status_color = COLOR_INIT
                    active_processes += 1
                elif "DESERT" in status_str:
                    status_color = COLOR_KILL_DESERT
                elif "LOOP" in status_str:  # ← 表示をLOOPに変更
                    status_color = COLOR_KILL_LOOP
                elif "GOAL" in status_str:
                    status_color = COLOR_GOAL
                else:
                    status_color = (100, 100, 100)
                    
                status_text = font_m.render(status_str, True, status_color)
                
                screen.blit(step_text, (30, y_offset + 10))
                screen.blit(epoch_text, (160, y_offset + 10))
                screen.blit(cost_text, (160, y_offset + 35))
                
                bar_width = 180
                bar_bg = pygame.Rect(340, y_offset + 15, bar_width, 10)
                progress = min(1.0, state["epoch"] / target_epochs)
                bar_fill = pygame.Rect(340, y_offset + 15, bar_width * progress, 10)
                pygame.draw.rect(screen, (80, 80, 90), bar_bg)
                pygame.draw.rect(screen, status_color, bar_fill)
                
                screen.blit(status_text, (540, y_offset + 10))
                
                y_offset += 70

            if active_processes == 0 and len(shared_dict) == len(step_candidates):
                footer_txt = "All processes finished. Close window to exit."
                footer_color = COLOR_GOAL
            else:
                footer_txt = f"Active Processes: {active_processes}"
                footer_color = TEXT_COLOR
                
            footer = font_m.render(footer_txt, True, footer_color)
            screen.blit(footer, (20, HEIGHT - 40))

        pygame.display.flip()
        clock.tick(15)

    for p in processes:
        if p.is_alive():
            p.terminate()
        p.join()
    pygame.quit()

# ======================================================
# 実行エントリーポイント
# ======================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    wine = load_wine()
    X = wine.data  
    y = wine.target
    y_binary = np.array([1.0 if label == 1 else 0.0 for label in y])
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    pca = PCA(n_components=13, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    np.random.seed(42)
    indices = np.random.permutation(len(X_pca))
    X_pca, y_binary = X_pca[indices], y_binary[indices]
    
    # テストデータをしっかり分離
    X_train, y_train = X_pca[:140], y_binary[:140]
    X_test, y_test = X_pca[140:], y_binary[140:]

    # 【極限実験用】次なる奈落（10^-8以下）へアクセルを踏み込めるリスト
    step_candidates = [1.0, 0.05, 0.01, 0.00001, 0.00000001, 0.0000000001]
    
    np.random.seed(42)
    initial_weights = list(np.random.uniform(-0.5, 0.5, size=105))
    log_interval = 15

    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    print("\n[INFO] コントロールパネルを起動しました。")
    run_control_panel(X_train, y_train, X_test, y_test, step_candidates, initial_weights, log_interval, shared_dict, timestamp)
    print("\n[INFO] プログラムを終了しました。")
