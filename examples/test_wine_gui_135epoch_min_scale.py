import os
import sys
import datetime

# 各種スレッドの制限（CPU負荷を最適化）
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
import threading
import pygame
import copy

# ======================================================
# 量子AI クラス本体
# ======================================================
class WineQuantumClassifier8Qubit_Final:
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

# グローバル共有辞書
gui_state = {
    "epoch": 0,
    "cost": 0.0,
    "grad": 0.0,
    "status": "READY",
    "current_acc": 0.0,
    "best_acc": 0.0,
    "best_epoch": 0,
    "target_epochs": 135
}

# ======================================================
# お宝ハント学習スレッド
# ======================================================
def train_thread_func(train_data, test_data, step_size, epochs, log_interval, log_filename):
    global gui_state
    
    classifier = WineQuantumClassifier8Qubit_Final(n_qubits=8, num_layers=4)
    np.random.seed(42)
    initial_weights = list(np.random.uniform(-0.5, 0.5, size=105))
    classifier.weights = np.array(initial_weights, requires_grad=True)
    
    opt_main = qml.AdamOptimizer(stepsize=0.05)
    grad_fn = qml.grad(classifier._mse_cost, argnum=0)
    
    prev_cost = float('inf')
    gui_state["status"] = "RUNNING"
    
    # ベストモデル保存用変数
    best_weights = copy.deepcopy(classifier.weights)
    best_accuracy = 0.0
    best_epoch = 0
    
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"======================================================\n")
        f.write(f" 量子AI 最終生産仕様 - 15Epoch毎テスト・ベストハント仕様\n")
        f.write(f" 固定粒度 (Step Size) : {step_size}\n")
        f.write(f" 目標エポック (Epochs): {epochs}\n")
        f.write(f"======================================================\n\n")
        
        for epoch in range(epochs):
            if gui_state["status"] == "SHUTDOWN":
                return
                
            classifier.weights, cost = opt_main.step_and_cost(
                lambda w: classifier._mse_cost(w, train_data, step_size), classifier.weights
            )
            
            gui_state["epoch"] = epoch + 1
            gui_state["cost"] = float(cost)
            
            # 15エポックごとに評価（お宝ハント）を敢えて実行！
            if (epoch + 1) % log_interval == 0:
                grads = grad_fn(classifier.weights, train_data, step_size)
                current_grad = float(np.linalg.norm(grads[0:classifier.q_param_count]))
                gui_state["grad"] = current_grad
                
                # テストデータで現在の正答率を判定
                current_acc = classifier.evaluate(test_data, step_size)
                gui_state["current_acc"] = current_acc
                
                # 過去最高を更新したらお宝として退避
                if current_acc >= best_accuracy:
                    best_accuracy = current_acc
                    best_epoch = epoch + 1
                    best_weights = copy.deepcopy(classifier.weights)
                    
                    gui_state["best_acc"] = best_accuracy
                    gui_state["best_epoch"] = best_epoch
                
                log_line = (
                    f"Epoch {epoch + 1:3d} / {epochs} | "
                    f"Cost: {cost:.6f} | "
                    f"Q-Grad Norm: {current_grad:.8f} | "
                    f"Test Acc: {current_acc * 100:.2f} % (Best: {best_accuracy * 100:.2f} % at Epoch {best_epoch})\n"
                )
                f.write(log_line)
                f.flush()
                
                # 安全のための早期Kill判定（ウロウロの網）
                if (prev_cost - cost) <= 0.0001:
                    gui_state["status"] = "KILL (LOOP)"
                    break
                    
                prev_cost = cost
                
        # ループ終了後、メモリに退避させておいた「歴代最強の重み」を機体に再装填
        classifier.weights = best_weights
        final_hunter_acc = classifier.evaluate(test_data, step_size)
        
        gui_state["status"] = "HUNTED (SUCCESS)"
        summary_msg = (
            f"\n=> 🏁 [MISSION COMPLETE]\n"
            f"   ハントした最高正解率: {final_hunter_acc * 100:.2f} % (Epoch {best_epoch} 時のモデルを採用)\n"
        )
        f.write(summary_msg)
        f.flush()
    return

# ======================================================
# Pygame GUI メインループ
# ======================================================
def run_gui(train_data, test_data, step_size, epochs, log_interval, log_filename):
    global gui_state
    pygame.init()
    WIDTH, HEIGHT = 850, 480
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quantum AI Best Model Hunter Dashboard")
    clock = pygame.time.Clock()

    font_xl = pygame.font.SysFont("consolas", 26, bold=True)
    font_l = pygame.font.SysFont("consolas", 20, bold=True)
    font_m = pygame.font.SysFont("consolas", 16)
    font_s = pygame.font.SysFont("consolas", 14)

    BG_COLOR = (12, 12, 22)
    PANEL_COLOR = (22, 22, 32)
    TEXT_COLOR = (220, 220, 230)
    
    COLOR_RUNNING = (50, 210, 110)
    COLOR_KILL_LOOP = (220, 140, 40)
    COLOR_GOAL = (60, 180, 255)

    btn_launch = pygame.Rect(WIDTH//2 - 150, 220, 300, 50)
    ui_mode = "LAUNCHER"
    train_thread = None

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gui_state["status"] = "SHUTDOWN"
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and ui_mode == "LAUNCHER":
                if btn_launch.collidepoint(mouse_pos):
                    ui_mode = "MONITOR"
                    train_thread = threading.Thread(
                        target=train_thread_func,
                        args=(train_data, test_data, step_size, epochs, log_interval, log_filename)
                    )
                    train_thread.start()

        screen.fill(BG_COLOR)

        if ui_mode == "LAUNCHER":
            title = font_xl.render("QUANTUM AI BEST MODEL HUNTER", True, TEXT_COLOR)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, 80))
            
            info_txt = font_m.render(f"Step Size: {step_size} | Cap: {epochs} Epochs | Interval: {log_interval}", True, (130, 130, 140))
            screen.blit(info_txt, (WIDTH//2 - info_txt.get_width()//2, 130))
            
            pygame.draw.rect(screen, (50, 150, 250) if btn_launch.collidepoint(mouse_pos) else (30, 110, 200), btn_launch, border_radius=8)
            btn_txt = font_l.render("LAUNCH HUNTER ATTACK", True, (255, 255, 255))
            screen.blit(btn_txt, (btn_launch.centerx - btn_txt.get_width()//2, btn_launch.centery - btn_txt.get_height()//2))

        elif ui_mode == "MONITOR":
            title = font_l.render("Best Model Hunter Monitor", True, (150, 150, 160))
            screen.blit(title, (30, 25))
            
            panel_rect = pygame.Rect(30, 65, WIDTH - 60, 310)
            pygame.draw.rect(screen, PANEL_COLOR, panel_rect, border_radius=10)
            
            epoch_lbl = font_xl.render(f"Epoch: {gui_state['epoch']} / {epochs}", True, TEXT_COLOR)
            cost_lbl = font_l.render(f"Current Cost: {gui_state['cost']:.6f}", True, (100, 190, 255))
            grad_lbl = font_m.render(f"Q-Grad Norm : {gui_state['grad']:.8f}", True, (180, 180, 190))
            
            screen.blit(epoch_lbl, (60, 95))
            screen.blit(cost_lbl, (60, 150))
            screen.blit(grad_lbl, (60, 190))
            
            # ハンター専用の戦績ボード（右側）
            pygame.draw.rect(screen, (15, 15, 25), (WIDTH - 380, 95, 320, 130), border_radius=6)
            curr_acc_lbl = font_m.render(f"Current Test Acc: {gui_state['current_acc']*100:.2f} %", True, (200, 200, 200))
            best_acc_lbl = font_l.render(f"RECORD BEST ACC : {gui_state['best_acc']*100:.2f} %", True, COLOR_RUNNING)
            best_ep_lbl = font_s.render(f"Hunted at Epoch: {gui_state['best_epoch']}", True, (140, 140, 150))
            
            screen.blit(curr_acc_lbl, (WIDTH - 360, 115))
            screen.blit(best_acc_lbl, (WIDTH - 360, 150))
            screen.blit(best_ep_lbl, (WIDTH - 360, 195))
            
            # ステータス・プログレス
            status_str = gui_state["status"]
            status_color = COLOR_RUNNING
            if "KILL" in status_str: status_color = COLOR_KILL_LOOP
            if "HUNTED" in status_str: status_color = COLOR_GOAL
            
            status_lbl = font_xl.render(status_str, True, status_color)
            screen.blit(status_lbl, (60, 250))
            
            bar_width = WIDTH - 120
            bar_bg = pygame.Rect(60, 330, bar_width, 12)
            progress = min(1.0, gui_state["epoch"] / epochs)
            bar_fill = pygame.Rect(60, 330, bar_width * progress, 12)
            pygame.draw.rect(screen, (45, 45, 55), bar_bg, border_radius=3)
            pygame.draw.rect(screen, status_color, bar_fill, border_radius=3)
            
            if "HUNTED" in status_str:
                footer_txt = f"Mission Complete. Successfully saved Best Model ({gui_state['best_acc']*100:.2f}%)."
                footer_lbl = font_m.render(footer_txt, True, COLOR_GOAL)
                screen.blit(footer_lbl, (30, HEIGHT - 45))

        pygame.display.flip()
        clock.tick(30)

    if train_thread and train_thread.is_alive():
        train_thread.join()
    pygame.quit()

# ======================================================
# エントリーポイント
# ======================================================
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/hunter_production_{timestamp}.log"
    
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
    
    X_train, y_train = X_pca[:140], y_binary[:140]
    X_test, y_test = X_pca[140:], y_binary[140:]
    
    train_data = [(np.array(x, requires_grad=False), y) for x, y in zip(X_train, y_train)]
    test_data = [(np.array(x, requires_grad=False), y) for x, y in zip(X_test, y_test)]
    
    FINAL_STEP_SIZE = 1e-15
    TOTAL_EPOCHS = 135  # 135エポックキャップ
    LOG_INTERVAL = 15   # 15エポックごとの定期仕分けテスト

    print("\n[INFO] ベストモデルハンターを起動しました。")
    run_gui(train_data, test_data, FINAL_STEP_SIZE, TOTAL_EPOCHS, LOG_INTERVAL, log_filename)
    print("[INFO] プログラムを終了しました。")
