
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision import datasets
from torch.utils.data import DataLoader

# 1. 출력 디렉토리 생성
os.makedirs("스캐쥴러", exist_ok=True)

# 2. 모델 정의
class BaselineNet(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(BaselineNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        logits = self.fc3(x)
        return logits

def main():
    # 3. 환경 및 데이터 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용하는 디바이스: {device}")

    transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    batch_size = 64
    epochs = 30
    loss_type = "CE"
    opt_type = "Momentum"
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # 학습률과 스케줄러 조합 정의
    learning_rates = [0.1, 0.01, 0.001]
    scheduler_types = ["None", "ExponentialLR", "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    
    # 4. 결과 출력 파일 초기화
    output_txt = "스캐쥴러_결과.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("="*105 + "\n")
        f.write(f" [실험: Momentum 학습률 & 스케줄러 조합 비교]\n")
        f.write("-" * 105 + "\n")
        f.write(f" {'Optimizer':<12} | {'학습률':<8} | {'Scheduler':<20} | {'최종 정확도 (%)':<15} | {'수렴(최저Loss에폭)':<18} | {'안정성 (Std)':<15}\n")
        f.write("-" * 105 + "\n")
        
    # 측정할 에폭 리스트 지정
    grad_epochs = [1, 5, 10, 15, 20, 25, 30]
    logit_epochs = [1, 15, 30]
        
    # 5. 조합별 학습 자동 진행
    for lr in learning_rates:
        for sched_name in scheduler_types:
            print(f"\n[실행 중] Optimizer: {opt_type}, LR: {lr}, Scheduler: {sched_name}")
            model = BaselineNet().to(device)
            
            # Momentum 옵티마이저 고정
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            
            # 스케줄러 선택 로직
            if sched_name == "None":
                scheduler = None
            elif sched_name == "ExponentialLR":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            elif sched_name == "StepLR":
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
            elif sched_name == "CosineAnnealingLR":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            elif sched_name == "ReduceLROnPlateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
            
            results = {
                "config": {"loss_type": loss_type, "epochs": epochs, "optimizer_type": opt_type, "learning_rate": lr, "scheduler": sched_name},
                "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                "final_accuracy": 0.0, "converged_epoch": 0, "stability_score": "", "stability_std": 0.0,
                "logits_distribution": {str(ep): None for ep in logit_epochs},
                "gradient_flow": {str(ep): {} for ep in grad_epochs}
            }
            
            best_val_loss = float('inf')
            
            for epoch in range(1, epochs + 1):
                # --- [TRAIN LOOP] ---
                model.train()
                train_loss, correct_train, total_train = 0.0, 0, 0
                epoch_first_logits = None
                
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    logits = model(inputs)
                    
                    if batch_idx == 0:
                        epoch_first_logits = logits.detach().cpu().numpy()
                        
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(logits, labels)
                    loss.backward()
                    
                    # 그래디언트 추출 (1, 5, 10, 15, 20, 25, 30 에폭)
                    if batch_idx == 0 and epoch in grad_epochs:
                        for name, param in model.named_parameters():
                            if param.grad is not None and "weight" in name:
                                results["gradient_flow"][str(epoch)][name] = param.grad.abs().mean().item()
                                
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(logits, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                    
                epoch_train_loss = train_loss / total_train
                epoch_train_acc = (correct_train / total_train) * 100
                
                # --- [TEST LOOP] ---
                model.eval()
                test_loss, correct_test, total_test = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logits = model(inputs)
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(logits, labels)
                        test_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(logits, 1)
                        total_test += labels.size(0)
                        correct_test += (predicted == labels).sum().item()
                        
                epoch_test_loss = test_loss / total_test
                epoch_test_acc = (correct_test / total_test) * 100
                
                results["train_loss"].append(epoch_train_loss)
                results["train_acc"].append(epoch_train_acc)
                results["val_loss"].append(epoch_test_loss)
                results["val_acc"].append(epoch_test_acc)
                
                # 수렴 에폭(Loss 최솟값) 업데이트
                if epoch_test_loss < best_val_loss:
                    best_val_loss = epoch_test_loss
                    results["converged_epoch"] = epoch
                    
                # Logits 분포 저장 (1, 15, 30 에폭)
                if epoch in logit_epochs:
                    results["logits_distribution"][str(epoch)] = epoch_first_logits
                    
                # 스케줄러 업데이트 로직
                if scheduler is not None:
                    if sched_name == "ReduceLROnPlateau":
                        scheduler.step(epoch_test_loss)
                    else:
                        scheduler.step()
                
            results["final_accuracy"] = results["val_acc"][-1]
            
            # --- [결과 분석 및 안정성 평가 (수치화)] ---
            val_losses = results["val_loss"]
            if len(val_losses) > 1:
                loss_diffs = np.diff(val_losses)
                stability_val = float(np.std(loss_diffs))
                results["stability_std"] = stability_val
                results["stability_score"] = f"{stability_val:.4f}"
            else:
                results["stability_std"] = 0.0
                results["stability_score"] = "N/A"
                
            # 6. 결과 시각화 및 이미지 저장
            plt.style.use('seaborn-v0_8-whitegrid')
            fig = plt.figure(figsize=(18, 12))  # 세로 크기 확대
            
            # 1행 1열: Loss & Accuracy
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(range(1, epochs + 1), results["train_loss"], 'b-', label='Train Loss')
            ax1.plot(range(1, epochs + 1), results["val_loss"], 'r--', label='Test Loss')
            ax1.set_title(f'Loss Curves (Momentum, LR={lr}, {sched_name})')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend(loc='upper left')
            
            ax1_twin = ax1.twinx()
            ax1_twin.plot(range(1, epochs + 1), results["train_acc"], 'g-', alpha=0.3, label='Train Acc')
            ax1_twin.plot(range(1, epochs + 1), results["val_acc"], 'g--', alpha=0.5, label='Test Acc') 
            ax1_twin.set_ylabel('Accuracy (%)')
            ax1_twin.legend(loc='lower right')
            
            # 1행 2열: Logits 분포 변화 (1, 15, 30)
            ax2 = fig.add_subplot(2, 2, 2)
            colors_logits = ["#c4ffa1", "#b885ff", "#ffa34d"]
            for p, c in zip([str(ep) for ep in logit_epochs], colors_logits):
                data = results["logits_distribution"][p]
                if data is not None:
                    ax2.hist(data.flatten(), bins=30, alpha=0.5, label=f'Epoch {p}', color=c)
            ax2.set_title('Logits Distribution Changes (Epoch 1, 15, 30)')
            ax2.set_xlabel('Logit Value')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            
            # 2행 1열(전체 차지): 레이어별 Gradient Flow Magnitude (1, 5, 10, 15, 20, 25, 30)
            ax3 = fig.add_subplot(2, 1, 2)
            layers = list(results["gradient_flow"]["1"].keys()) if results["gradient_flow"]["1"] else []
            x_indexes = np.arange(len(layers))
            
            if layers:
                bar_width = 0.1  # 7개 막대 배치를 위해 너비 조정
                cmap = plt.get_cmap("viridis")
                colors_grad = [cmap(i) for i in np.linspace(0.2, 0.9, len(grad_epochs))]
                offsets = np.arange(-3, 4) * bar_width
                
                for i, ep in enumerate(grad_epochs):
                    grads = [results["gradient_flow"][str(ep)].get(l, 0) for l in layers]
                    ax3.bar(x_indexes + offsets[i], grads, bar_width, label=f'Epoch {ep}', color=colors_grad[i])
                
                display_names = [l.split('.')[0] for l in layers]
                ax3.set_xticks(x_indexes)
                ax3.set_xticklabels(display_names)
                ax3.set_yscale('log')
                ax3.set_title('Gradient Flow Magnitude over Epochs (1 ~ 30)')
                ax3.set_xlabel('Layers')
                ax3.set_ylabel('Mean Absolute Gradient (Log Scale)')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No Gradient Data', ha='center', va='center')
                
            plt.tight_layout()
            
            # 스캐쥴러/[옵티마이저 명칭]+[학습률]+[스캐쥴러].png 이름으로 저장
            save_path = f"스캐쥴러/{opt_type}+{lr}+{sched_name}.png"
            plt.savefig(save_path)
            plt.close(fig)
            print(f"-> 시각화 저장 완료: {save_path}")
            
            # 7. 결과 텍스트 파일 누적 기록
            with open(output_txt, "a", encoding="utf-8") as f:
                res_str = f" {opt_type:<12} | {lr:<8} | {sched_name:<20} | {results['final_accuracy']:<15.2f}% | {results['converged_epoch']:<12} Epoch | {results['stability_score']:<15}\n"
                f.write(res_str)

if __name__ == "__main__":
    main()
