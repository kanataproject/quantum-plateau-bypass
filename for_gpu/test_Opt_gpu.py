import os
import datetime
import multiprocessing

import numpy as np  # 【修正】pennylaneのnumpyではなく、標準のnumpyを使用！(超重要)
import pygame
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from quantum_classifier import _run_parallel_worker, HybridQuantumClassifier, load_config

# ======================================================
# Pygame GUI メインループ
# ======================================================
def run_control_panel(config, X_train, y_train, X_test, y_test,
                      step_candidates, initial_q_weights, shared_dict, timestamp, log_dir):
    pygame.init()
    WIDTH, HEIGHT = 900, 750
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    output_classes = config["data"].get("output_classes", 2)
    device_str = "CUDA" if torch.cuda.is_available() else "CPU"
    pygame.display.set_caption(
        f"QML Hunt [{device_str}] - {config['data']['input_dimensions']}D / {output_classes} Classes")
    clock = pygame.time.Clock()

    font_xl = pygame.font.SysFont("consolas", 32, bold=True)
    font_l  = pygame.font.SysFont("consolas", 24, bold=True)
    font_m  = pygame.font.SysFont("consolas", 18, bold=True)
    font_s  = pygame.font.SysFont("consolas", 14)

    BG_COLOR          = (25, 15, 20)
    TEXT_COLOR        = (230, 220, 220)
    COLOR_BTN_IDLE    = (80, 60, 60)
    COLOR_BTN_HOVER   = (100, 80, 80)
    COLOR_START_IDLE  = (180, 80, 80)
    COLOR_START_HOVER = (210, 100, 100)
    COLOR_RUNNING     = (110, 210, 110)
    COLOR_INIT        = (130, 120, 120)
    COLOR_KILL_LOOP   = (220, 140, 40)
    COLOR_GOAL        = (255, 150, 150)

    ui_state      = "SETUP"
    target_epochs = config["hunter_settings"]["max_epochs"]
    processes     = []

    # === 追加: 誤って消してしまったボタン定義を復活 ===
    btn_minus = pygame.Rect(350, 250, 50, 40)
    btn_plus  = pygame.Rect(500, 250, 50, 40)
    btn_start = pygame.Rect(300, 350, 300, 60)
    # ==================================================

    # 【修正】requires_grad=False を削除（標準NumPyには不要なため、純粋な配列として安全にプロセス間通信される）
    train_data = [(np.array(x), int(y)) for x, y in zip(X_train, y_train)]
    test_data  = [(np.array(x), int(y)) for x, y in zip(X_test, y_test)]

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
                    config["hunter_settings"]["max_epochs"] = target_epochs
                    for step in step_candidates:
                        p = multiprocessing.Process(
                            target=_run_parallel_worker,
                            args=(step, config, train_data, test_data,
                                  initial_q_weights, shared_dict, timestamp, log_dir)
                        )
                        p.start()
                        processes.append(p)

        screen.fill(BG_COLOR)

        if ui_state == "SETUP":
            title = font_xl.render(
                f"{config['quantum_model']['n_qubits']}Q / "
                f"{config['data']['input_dimensions']}D / "
                f"{output_classes} CLASSES [{device_str}]", True, TEXT_COLOR)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))

            epoch_label = font_l.render("Max Target Epochs", True, (160, 150, 150))
            screen.blit(epoch_label, (WIDTH//2 - epoch_label.get_width()//2, 200))

            pygame.draw.rect(screen,
                COLOR_BTN_HOVER if btn_minus.collidepoint(mouse_pos) else COLOR_BTN_IDLE,
                btn_minus, border_radius=5)
            minus_txt = font_l.render("-", True, TEXT_COLOR)
            screen.blit(minus_txt, (btn_minus.centerx - minus_txt.get_width()//2,
                                    btn_minus.centery - minus_txt.get_height()//2))

            epoch_val = font_xl.render(f"{target_epochs}", True, COLOR_GOAL)
            screen.blit(epoch_val, (WIDTH//2 - epoch_val.get_width()//2, 255))

            pygame.draw.rect(screen,
                COLOR_BTN_HOVER if btn_plus.collidepoint(mouse_pos) else COLOR_BTN_IDLE,
                btn_plus, border_radius=5)
            plus_txt = font_l.render("+", True, TEXT_COLOR)
            screen.blit(plus_txt, (btn_plus.centerx - plus_txt.get_width()//2,
                                   btn_plus.centery - plus_txt.get_height()//2))

            pygame.draw.rect(screen,
                COLOR_START_HOVER if btn_start.collidepoint(mouse_pos) else COLOR_START_IDLE,
                btn_start, border_radius=10)
            start_txt = font_xl.render("LAUNCH HUNT", True, (255, 255, 255))
            screen.blit(start_txt, (btn_start.centerx - start_txt.get_width()//2,
                                    btn_start.centery - start_txt.get_height()//2))

        elif ui_state == "RUNNING":
            title = font_l.render(
                f"Universal QML Dashboard [{device_str}]", True, (190, 180, 180))
            screen.blit(title, (20, 20))

            y_offset         = 70
            active_processes = 0

            for step in step_candidates:
                state = shared_dict.get(step, {
                    "epoch": 0, "cost": 0.0, "status": "WAITING",
                    "best_acc": 0.0, "best_epoch": 0,
                    "current_batch": 0, "total_batches": 0
                })

                panel_rect = pygame.Rect(20, y_offset, WIDTH - 40, 80)
                pygame.draw.rect(screen, (40, 30, 30), panel_rect, border_radius=6)

                step_text  = font_m.render(f"Step: {step:<8}", True, COLOR_GOAL)
                
                cb = state.get("current_batch", 0)
                tb = state.get("total_batches", 0)
                epoch_str = f"Epoch: {state['epoch']:>3} / {target_epochs} [B: {cb:>2}/{tb:>2}]"
                epoch_text = font_m.render(epoch_str, True, TEXT_COLOR)
                
                best_acc   = state.get("best_acc", 0.0)
                best_ep    = state.get("best_epoch", 0)
                info_text  = font_s.render(
                    f"Cost: {state['cost']:.5f} | Best Acc: {best_acc*100:.2f}% (Ep {best_ep})",
                    True, (200, 200, 200))

                status_str = state["status"]
                if "RUNNING" in status_str:
                    status_color = COLOR_RUNNING
                    active_processes += 1
                elif "INITIALIZING" in status_str:
                    status_color = COLOR_INIT
                    active_processes += 1
                elif "LOOP" in status_str:
                    status_color = COLOR_KILL_LOOP
                elif "GOAL" in status_str:
                    status_color = COLOR_GOAL
                elif "CRASH" in status_str:
                    status_color = (255, 50, 50)  # 赤色でクラッシュを通知
                else:
                    status_color = (100, 100, 100)

                status_text = font_m.render(status_str, True, status_color)

                screen.blit(step_text,  (35, y_offset + 15))
                screen.blit(epoch_text, (180, y_offset + 15))
                screen.blit(info_text,  (180, y_offset + 40))

                bar_width = 250
                bar_bg   = pygame.Rect(450, y_offset + 20, bar_width, 12)
                progress = min(1.0, state["epoch"] / target_epochs) if target_epochs > 0 else 0
                bar_fill = pygame.Rect(450, y_offset + 20, int(bar_width * progress), 12)
                pygame.draw.rect(screen, (60, 50, 50), bar_bg,   border_radius=4)
                pygame.draw.rect(screen, status_color, bar_fill, border_radius=4)
                screen.blit(status_text, (720, y_offset + 15))

                y_offset += 95

            if active_processes == 0 and len(shared_dict) == len(step_candidates):
                footer_txt   = "All lanes finished or crashed! Check the console/logs."
                footer_color = COLOR_GOAL
            else:
                nodes      = config['classical_model']['hidden_nodes']
                batch_size = config['classical_model'].get('batch_size', 32)
                footer_txt = (
                    f"Active: {active_processes}/{len(step_candidates)} | "
                    f"{config['data']['input_dimensions']}D / {output_classes}-Class / "
                    f"{nodes} Nodes / Batch {batch_size} [{device_str}]"
                )
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
# メイン処理
# ======================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    try:
        config = load_config("config.json")
        print("[INFO] 外部の config.json を正常にロードしました。")
    except FileNotFoundError:
        print("[WARNING] config.json が見つからないため、デフォルト値を使用します。")
        config = {
            "system": {"random_seed": 42},
            "data": {"input_dimensions": 64, "output_classes": 10},
            "quantum_model": {"n_qubits": 10, "num_layers": 4},
            "classical_model": {
                "hidden_nodes": 256,
                "batch_size": 128
            },
            "hunter_settings": {
                "max_epochs": 1000,
                "cost_reversal_limit": 10,
                "log_interval": 1,
                "learning_rate": 0.01,
                "learning_rate_classical": 0.001,
                "step_candidates": [0.01]
            }
        }

    print(f"[INFO] Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    print("[INFO] Optdigits（64次元・10クラス）データセットをロード中...")
    digits   = load_digits()
    X        = digits.data
    y        = digits.target

    X_scaled = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=1400, test_size=300,
        random_state=42, stratify=y
    )

    # === 【スプリント実験用】データをギュッと縮小して超高速化 ===
    X_train = X_train[:1400]
    y_train = y_train[:1400]
    X_test  = X_test[:300]
    y_test  = y_test[:300]
    # ========================================================

    n_qubits = config["quantum_model"]["n_qubits"]
    num_layers = config["quantum_model"]["num_layers"]
    q_param_count = n_qubits * num_layers
    
    np.random.seed(config["system"]["random_seed"])
    initial_q_weights = list(np.random.uniform(-0.5, 0.5, size=q_param_count))

    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manager     = multiprocessing.Manager()
    shared_dict = manager.dict()

    print(f"\n[INFO] 量子パラメータ数: {q_param_count}")
    print("[INFO] GUIコントロールパネルを起動します...")
    run_control_panel(
        config, X_train, y_train, X_test, y_test,
        config["hunter_settings"]["step_candidates"],
        initial_q_weights, shared_dict, timestamp, log_dir
    )
    print("\n[INFO] ハント完了。プログラムを終了します。")
