
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
os.makedirs("출력", exist_ok=True)

# 2. 모델 정의 (기존 구조 변경 없음)
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
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # 옵티마이저와 학습률 모든 조합 리스트 정의
    combinations = [
        ("SGD", 0.1), ("SGD", 0.01), ("SGD", 0.001),
        ("Momentum", 0.1), ("Momentum", 0.01), ("Momentum", 0.001),
        ("Adam", 0.1), ("Adam", 0.01), ("Adam", 0.001)
    ]
    
    # 4. 결과 출력 파일 초기화
    with open("결과.txt", "w", encoding="utf-8") as f:
        f.write("="*85 + "\n")
        f.write(f" [실험 C: 최적화 알고리즘 비교]\n")
        f.write("-" * 85 + "\n")
        f.write(f" {'Optimizer 항목':<15} | {'학습률':<12} | {'최종 정확도 (%)':<15} | {'수렴 속도(최저 Loss 에폭)':<20} | {'안정성 (Std)':<15}\n")
        f.write("-" * 85 + "\n")
        
    # 5. 조합별 학습 자동 진행
    for opt_type, lr in combinations:
        print(f"\n[실행 중] Optimizer: {opt_type}, Learning Rate: {lr}")
        model = BaselineNet().to(device)
        
        # 옵티마이저 선택 로직
        if opt_type == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif opt_type == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif opt_type == "Momentum":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        # 측정할 에폭 리스트 지정
        grad_epochs = [1, 5, 10, 15, 20, 25, 30]
        logit_epochs = [1, 15, 30]
        
        results = {
            "config": {"loss_type": loss_type, "epochs": epochs, "optimizer_type": opt_type, "learning_rate": lr},
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
                
                # 가중치 레이어별 gradient 평균값 추출 (1, 5, 10, 15, 20, 25, 30 에폭)
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
                
            scheduler.step()
            
        results["final_accuracy"] = results["val_acc"][-1]
        
        # --- [결과 분석 및 안정성 평가 (수치화)] ---
        val_losses = results["val_loss"]
        if len(val_losses) > 1:
            loss_diffs = np.diff(val_losses)
            stability_val = float(np.std(loss_diffs))
            results["stability_std"] = stability_val
            # 안정도를 문자열이 아닌 계산된 수치값으로 직접 기록
            results["stability_score"] = f"{stability_val:.4f}"
        else:
            results["stability_std"] = 0.0
            results["stability_score"] = "N/A"
            
        # 6. 결과 시각화 및 이미지 저장 (세로 2줄 레이아웃으로 변경)
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(18, 12))  # 세로 크기 확대
        
        # 첫 번째 그래픽: Loss & Accuracy (1행 1열)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(range(1, epochs + 1), results["train_loss"], 'b-', label='Train Loss')
        ax1.plot(range(1, epochs + 1), results["val_loss"], 'r--', label='Test Loss')
        ax1.set_title(f'Loss Curves ({opt_type}, LR={lr})')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(range(1, epochs + 1), results["train_acc"], 'g-', alpha=0.3, label='Train Acc')
        ax1_twin.plot(range(1, epochs + 1), results["val_acc"], 'g--', alpha=0.5, label='Test Acc') 
        ax1_twin.set_ylabel('Accuracy (%)')
        ax1_twin.legend(loc='lower right')
        
        # 두 번째 그래픽: Logits 분포 변화 (1, 15, 30 에폭) (1행 2열)
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
        
        # 세 번째 그래픽: 레이어별 Gradient Flow Magnitude (2행 한 줄 전체)
        ax3 = fig.add_subplot(2, 1, 2)
        layers = list(results["gradient_flow"]["1"].keys()) if results["gradient_flow"]["1"] else []
        x_indexes = np.arange(len(layers))
        
        if layers:
            bar_width = 0.1  # 7개의 막대가 들어가야 하므로 너비 축소
            cmap = plt.get_cmap("viridis")
            colors_grad = [cmap(i) for i in np.linspace(0.2, 0.9, len(grad_epochs))]
            offsets = np.arange(-3, 4) * bar_width  # 막대가 가운데 정렬되도록 오프셋 계산 (-3 ~ +3)
            
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
        
        # [옵티마이저 명칭]+[학습률].png 이름으로 저장
        save_path = f"출력/{opt_type}+{lr}.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"-> 시각화 저장 완료: {save_path}")
        
        # 7. 결과.txt에 해당 조합 정보 누적 기록(Append)
        with open("결과.txt", "a", encoding="utf-8") as f:
            res_str = f" {opt_type:<14} | {lr:<12} | {results['final_accuracy']:<15.2f}% | {results['converged_epoch']:<20} Epoch | {results['stability_score']:<15}\n"
            f.write(res_str)

if __name__ == "__main__":
    main()
