"""
Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
MNIST Experiment Implementation

Paper specifications:
  Teacher : 2 hidden layers x 1200 ReLU units, dropout(0.2 input, 0.5 hidden)
  Student : 2 hidden layers x 800  ReLU units, no dropout
  Data    : MNIST, images jittered up to 2px in any direction
  Loss    : alpha * L_soft(T) + (1-alpha) * L_hard, scaled by T^2

Usage:
  pip install torch torchvision
  python hinton_mnist_distillation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Config — mirrors paper where specified, reasonable defaults elsewhere
# ---------------------------------------------------------------------------

BATCH_SIZE      = 128
TEACHER_EPOCHS  = 100
STUDENT_EPOCHS  = 100
LR              = 1e-3           # Adam; paper used SGD but didn't specify lr
TEMPERATURE     = 8.0            # T — paper found T in [2,8] worked well
ALPHA           = 0.9            # weight on soft-target loss (paper: close to 1)
SEED            = 42
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data — MNIST with 2-pixel random translation (the "jitter" in the paper)
# ---------------------------------------------------------------------------

def get_dataloaders():
    # Paper: "jittering the image by up to 2 pixels in any direction"
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST("./data", train=True,  download=True, transform=train_transform)
    test_data  = datasets.MNIST("./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class TeacherNet(nn.Module):
    """
    Paper spec: 2 hidden layers of 1200 ReLU units.
    Dropout: 0.2 on input, 0.5 after first hidden layer.
    """
    def __init__(self):
        super().__init__()
        self.drop_input  = nn.Dropout(p=0.2)
        self.fc1         = nn.Linear(784, 1200)
        self.drop_hidden = nn.Dropout(p=0.5)
        self.fc2         = nn.Linear(1200, 1200)
        self.fc3         = nn.Linear(1200, 10)

    def forward(self, x):
        x = x.view(-1, 784)          # flatten 28x28
        x = self.drop_input(x)
        x = F.relu(self.fc1(x))
        x = self.drop_hidden(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)            # return logits (softmax applied in loss)


class StudentNet(nn.Module):
    """
    Paper spec: 2 hidden layers of 800 ReLU units. No dropout.
    ~53% the parameter count of the teacher.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)            # return logits


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Equation from Section 2 of the paper:

        L = alpha * L_soft + (1 - alpha) * L_hard

    L_soft: cross-entropy between student and teacher soft distributions,
            both computed at temperature T, then scaled by T^2 to keep
            gradient magnitudes stable (see paper's note after Eq. 3).

    L_hard: standard cross-entropy between student (at T=1) and hard labels.

    Args:
        student_logits : raw logits from student  [batch, 10]
        teacher_logits : raw logits from teacher  [batch, 10]
        labels         : ground-truth class indices [batch]
        T              : temperature
        alpha          : weight on soft loss (paper recommends close to 1.0)
    """

    # --- Soft target loss ---
    # Both teacher and student use temperature T
    # KLDiv loss expects log-probabilities for input, probabilities for target
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)

    # KL divergence ≈ cross-entropy here since teacher targets are fixed
    # Scaled by T^2 per Hinton's note to compensate for gradient shrinkage
    L_soft = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)

    # --- Hard label loss ---
    # Standard cross-entropy at T=1
    L_hard = F.cross_entropy(student_logits, labels)

    return alpha * L_soft + (1 - alpha) * L_hard


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_teacher(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"  [Teacher] Epoch {epoch:3d} | Loss: {avg_loss:.4f}")


def train_student(student, teacher, loader, optimizer, epoch):
    student.train()
    teacher.eval()                    # teacher frozen during student training
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)   # no gradient through teacher

        loss = distillation_loss(
            student_logits, teacher_logits, labels,
            T=TEMPERATURE, alpha=ALPHA
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"  [Student] Epoch {epoch:3d} | Loss: {avg_loss:.4f}")


def evaluate(model, loader, label="Model"):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
    errors = len(loader.dataset) - correct
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"  [{label}] Accuracy: {accuracy:.2f}% | Errors: {errors}/{len(loader.dataset)}")
    return errors


def train_student_baseline(student, loader, optimizer, epoch):
    """
    Train student directly on hard labels — no distillation.
    Used as the baseline comparison (what the paper beats).
    """
    student.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = student(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


def main():
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Temperature: {TEMPERATURE} | Alpha: {ALPHA}\n")

    train_loader, test_loader = get_dataloaders()


# Step 1: Train teacher
  
    print("=" * 60)
    print("STEP 1: Training Teacher (1200-1200 units, dropout)")
    print("=" * 60)
    teacher = TeacherNet().to(DEVICE)
    teacher_opt = optim.Adam(teacher.parameters(), lr=LR)

    for epoch in range(1, TEACHER_EPOCHS + 1):
        train_teacher(teacher, train_loader, teacher_opt, epoch)

    print("\nTeacher evaluation:")
    teacher_errors = evaluate(teacher, test_loader, "Teacher")

# Step 2: Train student WITH distillation

    print("\n" + "=" * 60)
    print("STEP 2: Training Student WITH Distillation (800-800 units)")
    print("=" * 60)
    student_distilled = StudentNet().to(DEVICE)
    student_dist_opt  = optim.Adam(student_distilled.parameters(), lr=LR)

    for epoch in range(1, STUDENT_EPOCHS + 1):
        train_student(student_distilled, teacher, train_loader, student_dist_opt, epoch)

    print("\nDistilled student evaluation (T=1 at inference):")
    distilled_errors = evaluate(student_distilled, test_loader, "Distilled Student")

# Step 3: Train student WITHOUT distillation (baseline)

    print("\n" + "=" * 60)
    print("STEP 3: Training Student WITHOUT Distillation (baseline)")
    print("=" * 60)
    student_baseline = StudentNet().to(DEVICE)
    student_base_opt = optim.Adam(student_baseline.parameters(), lr=LR)

    for epoch in range(1, STUDENT_EPOCHS + 1):
        train_student_baseline(student_baseline, train_loader, student_base_opt, epoch)
        if epoch % 10 == 0:
            print(f"  [Baseline] Epoch {epoch:3d}")

    print("\nBaseline student evaluation:")
    baseline_errors = evaluate(student_baseline, test_loader, "Baseline Student")


# Summary — compare to paper's Table 1

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Teacher (1200 units, dropout)     : {teacher_errors} errors")
    print(f"  Student + distillation (T={TEMPERATURE})   : {distilled_errors} errors")
    print(f"  Student baseline (hard labels)    : {baseline_errors} errors")
    print()
    print("Paper's reported numbers (Table 1):")
    print("  Teacher                           : 67 errors")
    print("  Student + distillation            : 74 errors")
    print("  Student baseline                  : 146 errors")
    print("=" * 60)


if __name__ == "__main__":
    main()
