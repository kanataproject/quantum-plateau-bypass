import os
import sys
import datetime

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_breast_cancer
import multiprocessing
import pygame
import copy

# ======================================================
# 量子AI クラス本体 (10Qubit・4層・古典16ノード 最終検証版)
# ======================================================
class BreastCancerQuantumClassifier10Qubit:
    def __init__(self, n_qubits=10, num_layers=4): 
        self.n_qubits = n_qubits
        self.num_layers = num_layers 
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.weights = None
        
        # 10Qubit * 4層 = 40
        self.q_param_count = self.n_qubits * self.num_layers
        
        # 量子出力は 20次元 (Zが10個、Xが10個)
        # 隠れ層16ノード: 16 * 20 + 16(bias) = 336
        # 出力層: 16 * 1 + 1(bias) = 17
        # 古典層合計: 353
        self.c_param_count = 353 
        self.total_params = self.q_param_count + self.c_param_count
        self.qnode = qml.QNode(self._quantum_circuit, self.dev)

    def _quantum_circuit(self, inputs, q_weights):
        for layer in range(self.num_layers):
            for i in range(self.n_qubits):
                weight_idx = layer * self.n_qubits + i
                if layer == 0:
                    # 30次元データを10Qubitに分散エンコード
                    if i < 10: qml.RY(inputs[i] + q_weights[weight_idx], wires=i) # 0〜9
                    if i < 10: qml.RZ(inputs[i + 10], wires=i)                    # 10〜19
                    if i < 10: qml.RX(inputs[i + 20], wires=i)                    # 20〜29
                else:
                    qml.RY(q_weights[weight_idx], wires=i)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                
        # 出力は Z(10) + X(10) = 20次元
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
        
        for i in range(16): # 16ノード
            idx = offset + (i * 21) # 20入力 + 1バイアス
            h_in = sum(weights[idx + j] * translated[j] for j in range(20)) + weights[idx + 20]
            hidden_nodes.append(np.tanh(h_in))
            
        out_idx = offset + 336
        prediction = sum(weights[out_idx + k] * hidden_nodes[k] for k in range(16)) + weights[out_idx + 16]
        
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
# 並列ワーカー (完全自律型ハンター仕様)
# ======================================================
def run_parallel_hunter(step_size, X_train, y_train, X_test, y_test, initial_weights, epochs, log_interval, shared_dict, timestamp):
    os.environ["OMP_NUM_THREADS"] = "1"
    
    log_filename = f"logs/cancer_hunter_10Q_16N_{timestamp}_step_{step_size}.log"
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"======================================================\n")
        f.write(f" 乳がん30次元対応(10Qubit拡張 / 16ノード) - 最終検証版\n")
        f.write(f" 担当粒度 (Step Size) : {step_size}\n")
        f.write(f"======================================================\n\n")
        
        train_data = [(np.array(x, requires_grad=False), y) for x, y in zip(X_train, y_train)]
        test_data = [(np.array(x, requires_grad=False), y) for x, y in zip(X_test, y_test)]
        
        classifier = BreastCancerQuantumClassifier10Qubit(n_qubits=10, num_layers=4)
        classifier.weights = np.array(initial_weights, requires_grad=True)
    
        opt_main = qml.AdamOptimizer(stepsize=0.05)
        grad_fn = qml.grad(classifier._mse_cost, argnum=0)
        
        prev_cost = float('inf')
        
        best_weights = copy.deepcopy(classifier.weights)
        best_accuracy = 0.0
        best_epoch = 0
        stagnation_count = 0
        PATIENCE = 3 # ★ 猶予を3に延長
    
        state = {
            "epoch": 0, "cost": 0.0, "grad": 0.0, 
            "status": "INITIALIZING...", "current_acc": 0.0,
            "best_acc": 0.0, "best_epoch": 0, "stagnation": 0
        }
        shared_dict[step_size] = state
    
        for epoch in range(epochs):
            classifier.weights, cost = opt_main.step_and_cost(
                lambda w: classifier._mse_cost(w, train_data, step_size), classifier.weights
            )
            
            state["epoch"] = epoch + 1
            state["cost"] = float(cost)
            state["status"] = "RUNNING"
            
            if (epoch + 1) % log_interval == 0:
                grads = grad_fn(classifier.weights, train_data, step_size)
                current_grad = float(np.linalg.norm(grads[0:classifier.q_param_count]))
                state["grad"] = current_grad
                
                current_acc = classifier.evaluate(test_data, step_size)
                state["current_acc"] = current_acc
                
                if current_acc > best_accuracy:
                    best_accuracy = current_acc
                    best_epoch = epoch + 1
                    best_weights = copy.deepcopy(classifier.weights)
                    stagnation_count = 0
                    state["best_acc"] = best_accuracy
                    state["best_epoch"] = best_epoch
                else:
                    stagnation_count += 1
                    
                state["stagnation"] = stagnation_count
                
                log_line = (
                    f"Epoch {epoch + 1:3d} | Cost: {cost:.6f} | "
                    f"Acc: {current_acc*100:.2f}% (Best: {best_accuracy*100:.2f}% @ Ep{best_epoch}) | "
                    f"Stag: {stagnation_count}/{PATIENCE}\n"
                )
                f.write(log_line)
                f.flush()
                
                if stagnation_count >= PATIENCE:
                    state["status"] = "KILL (STAG)"
                    shared_dict[step_size] = state
                    f.write(f"\n=> ⚠️ [KILL] Test Acc が {PATIENCE}回連続で最高値を更新せず。過学習と判断し自動停止。\n")
                    return
            
                if (prev_cost - cost) <= 0.0001:
                    state["status"] = "KILL (LOOP)"
                    shared_dict[step_size] = state
                    f.write(f"\n=> ⚠️ [KILL] Cost減少幅の停滞(LOOP限界)を検知し自動停止。\n")
                    return
                    
                prev_cost = float(cost)
                
            shared_dict[step_size] = state
    
        state["status"] = "GOAL (SUCCESS)"
        shared_dict[step_size] = state
        f.write(f"\n=> 🏁 [GOAL] {epochs} Epoch 完走。\n")
    return

# ======================================================
# Pygame GUI メインループ
# ======================================================
def run_control_panel(X_train, y_train, X_test, y_test, step_candidates, initial_weights, log_interval, shared_dict, timestamp):
    pygame.init()
    WIDTH, HEIGHT = 900, 750
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("30D Cancer Data (10Qubit/16Nodes) High-Res Hunt")
    clock = pygame.time.Clock()

    font_xl = pygame.font.SysFont("consolas", 32, bold=True)
    font_l = pygame.font.SysFont("consolas", 24, bold=True)
    font_m = pygame.font.SysFont("consolas", 18, bold=True)
    font_s = pygame.font.SysFont("consolas", 14)

    BG_COLOR = (25, 15, 20)
    TEXT_COLOR = (230, 220, 220)
    COLOR_BTN_IDLE = (80, 60, 60)
    COLOR_BTN_HOVER = (100, 80, 80)
    COLOR_START_IDLE = (180, 80, 80)
    COLOR_START_HOVER = (210, 100, 100)
    
    COLOR_RUNNING = (110, 210, 110)
    COLOR_INIT = (130, 120, 120)
    COLOR_KILL_LOOP = (220, 140, 40)
    COLOR_KILL_STAG = (210, 80, 80)
    COLOR_GOAL = (255, 150, 150)

    ui_state = "SETUP"
    target_epochs = 100 # ★ 初期値を100に変更
    processes = []

    btn_minus = pygame.Rect(350, 250, 50, 40)
    btn_plus = pygame.Rect(500, 250, 50, 40)
    btn_start = pygame.Rect(300, 350, 300, 60)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and ui_state == "SETUP":
                if btn_minus.collidepoint(mouse_pos):
                    target_epochs = max(50, target_epochs - 50)
                elif btn_plus.collidepoint(mouse_pos):
                    target_epochs += 50
                elif btn_start.collidepoint(mouse_pos):
                    ui_state = "RUNNING"
                    for step in step_candidates:
                        p = multiprocessing.Process(
                            target=run_parallel_hunter,
                            args=(step, X_train, y_train, X_test, y_test, initial_weights, target_epochs, log_interval, shared_dict, timestamp)
                        )
                        p.start()
                        processes.append(p)
                        
        screen.fill(BG_COLOR)
        
        if ui_state == "SETUP":
            title = font_xl.render("10-QUBIT / 16-NODE HIGH-RES HUNT", True, TEXT_COLOR)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
            epoch_label = font_l.render("Max Target Epochs", True, (160, 150, 150))
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
            start_txt = font_xl.render("LAUNCH HIGH-RES HUNT", True, (255, 255, 255))
            screen.blit(start_txt, (btn_start.centerx - start_txt.get_width()//2, btn_start.centery - start_txt.get_height()//2))

        elif ui_state == "RUNNING":
            title = font_l.render("10Qubit High-Res Hunter Dashboard", True, (190, 180, 180))
            screen.blit(title, (20, 20))
            
            y_offset = 70
            active_processes = 0

            for step in step_candidates:
                state = shared_dict.get(step, {
                    "epoch": 0, "cost": 0.0, "grad": 0.0, "status": "WAITING",
                    "best_acc": 0.0, "best_epoch": 0, "stagnation": 0
                })
                
                panel_rect = pygame.Rect(20, y_offset, WIDTH - 40, 80)
                pygame.draw.rect(screen, (40, 30, 30), panel_rect, border_radius=6)
                
                step_text = font_m.render(f"Step: {step:<8}", True, COLOR_GOAL)
                epoch_text = font_m.render(f"Epoch: {state['epoch']:>3} / {target_epochs}", True, TEXT_COLOR)
                
                best_acc = state.get("best_acc", 0.0)
                best_ep = state.get("best_epoch", 0)
                stag = state.get("stagnation", 0)
                
                info_text1 = font_s.render(f"Cost: {state['cost']:.5f} | Best Acc: {best_acc*100:.2f}% (Ep {best_ep})", True, (200, 200, 200))
                info_text2 = font_s.render(f"Stagnation: {stag}/3", True, COLOR_KILL_STAG if stag > 0 else (160, 150, 150)) # ★ 表示を /3 に変更
                
                status_str = state["status"]
                if "RUNNING" in status_str:
                    status_color = COLOR_RUNNING
                    active_processes += 1
                elif "INITIALIZING" in status_str:
                    status_color = COLOR_INIT
                    active_processes += 1
                elif "STAG" in status_str:
                    status_color = COLOR_KILL_STAG
                elif "LOOP" in status_str:
                    status_color = COLOR_KILL_LOOP
                elif "GOAL" in status_str:
                    status_color = COLOR_GOAL
                else:
                    status_color = (100, 100, 100)
                    
                status_text = font_m.render(status_str, True, status_color)
                
                screen.blit(step_text, (35, y_offset + 15))
                screen.blit(epoch_text, (180, y_offset + 15))
                screen.blit(info_text1, (180, y_offset + 40))
                screen.blit(info_text2, (180, y_offset + 58))
                
                bar_width = 250
                bar_bg = pygame.Rect(450, y_offset + 20, bar_width, 12)
                progress = min(1.0, state["epoch"] / target_epochs)
                bar_fill = pygame.Rect(450, y_offset + 20, bar_width * progress, 12)
                pygame.draw.rect(screen, (60, 50, 50), bar_bg, border_radius=4)
                pygame.draw.rect(screen, status_color, bar_fill, border_radius=4)
                
                screen.blit(status_text, (720, y_offset + 15))
                
                y_offset += 95

            if active_processes == 0 and len(shared_dict) == len(step_candidates):
                footer_txt = "All lanes finished hunting! Check the sweet spots."
                footer_color = COLOR_GOAL
            else:
                footer_txt = f"Active Hunters: {active_processes} / {len(step_candidates)} (Heavy Load: Kill if too slow)"
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

if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cancer = load_breast_cancer()
    X = cancer.data  
    y = cancer.target
    
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    np.random.seed(42)
    indices = np.random.permutation(len(X_scaled))
    X_scaled, y = X_scaled[indices], y[indices]
    
    X_train, y_train = X_scaled[:400], y[:400]
    X_test, y_test = X_scaled[400:], y[400:]

    step_candidates = [1, 0.5, 0.1, 0.01, 0.001, 0.00000000000001]
    
    np.random.seed(42)
    # 10Qubit量子層(40) + 古典16ノード(353) = 393
    initial_weights = list(np.random.uniform(-0.5, 0.5, size=393))
    
    log_interval = 5 # ★ 5エポックごとに変更

    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    print("\n[INFO] 30次元 Breast Cancer (10Qubit / 16ノード / 高解像度5エポック版) を起動します...")
    run_control_panel(X_train, y_train, X_test, y_test, step_candidates, initial_weights, log_interval, shared_dict, timestamp)
    print("\n[INFO] プログラムを終了しました。")
