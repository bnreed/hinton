"""
Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
The Held-Out 3s Experiment (Section 3 of the paper)

This script reproduces the most striking result in the paper:

  A student trained on soft targets for every digit EXCEPT 3 —
  never shown a single image of a 3 during training —
  nonetheless correctly classifies most 3s at test time.

How is this possible? The teacher's soft outputs for digits like 8, 5, and 2
encode small but non-zero probability on class 3, because those digits share
visual structure with 3s. That residual signal — "dark knowledge" — leaks
the existence and identity of 3 into the student's representations,
even though the student never sees a 3 directly.

The paper also notes that the bias on the student's output unit for class 3
must be adjusted downward after training, since the student will have learned
to suppress 3 (it never appeared as a target). This script handles that.

Prerequisites:
  Run hinton_mnist_distillation_annotated.py first to produce:
    ./saved_models/teacher.pt

Usage:
  pip install torch torchvision
  python hinton_mnist_threes_experiment.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Config — must match the values used during original training
# ---------------------------------------------------------------------------

BATCH_SIZE   = 128
STUDENT_EPOCHS = 100
LR           = 1e-3
TEMPERATURE  = 8.0       # same T used when training the original distilled student
SEED         = 42
SAVE_DIR     = "./saved_models"
HELD_OUT_DIGIT = 3       # the digit withheld from the student's training set
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Network architectures — must be identical to the training script
# ---------------------------------------------------------------------------
# PyTorch saves weights, not architecture. To load a checkpoint you must
# first reconstruct the exact same class, then call load_state_dict().
# If the architecture doesn't match, you'll get a key mismatch error.

class TeacherNet(nn.Module):
    """2 hidden layers x 1200 ReLU units, dropout 0.2/0.5 (paper spec)."""
    def __init__(self):
        super().__init__()
        self.drop_input  = nn.Dropout(p=0.2)
        self.fc1         = nn.Linear(784, 1200)
        self.drop_hidden = nn.Dropout(p=0.5)
        self.fc2         = nn.Linear(1200, 1200)
        self.fc3         = nn.Linear(1200, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.drop_input(x)
        x = F.relu(self.fc1(x))
        x = self.drop_hidden(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class StudentNet(nn.Module):
    """2 hidden layers x 800 ReLU units, no dropout (paper spec)."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Loading saved models
# ---------------------------------------------------------------------------

def load_model(model, filename, optimizer=None):
    """
    Restores weights from a checkpoint saved by the training script.

    Args:
        model     : instantiated model of the correct architecture
        filename  : path to the .pt checkpoint file
        optimizer : optional — only needed if resuming training
    Returns:
        checkpoint dict with metadata (epoch, errors, etc.)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"\nCheckpoint not found: {filename}\n"
            f"Run hinton_mnist_distillation_annotated.py first to generate it."
        )

    checkpoint = torch.load(filename, map_location=DEVICE)
    # map_location=DEVICE remaps tensors to the current device.
    # Without this, a model saved on GPU would fail to load on a CPU-only machine.

    model.load_state_dict(checkpoint["model_state_dict"])
    # Copies the saved parameter values into the model.
    # The model is now in the exact state it was when saved.

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    saved_errors = checkpoint.get("test_errors", "unknown")
    print(f"  Loaded {filename}")
    print(f"  Trained for {checkpoint['epoch']} epochs | "
          f"Test errors when saved: {saved_errors}")
    return checkpoint


# ---------------------------------------------------------------------------
# Data — withheld digit version
# ---------------------------------------------------------------------------

def get_transform():
    """Standard MNIST transforms (no jitter — used for transfer set generation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_train_loader_without_digit(digit):
    """
    Returns a DataLoader over the MNIST training set with all examples
    of `digit` removed. This is the "transfer set" the student trains on.

    Args:
        digit : integer 0-9, the class to withhold

    How filtering works:
      datasets.MNIST stores labels as a tensor (.targets).
      We find the indices where the label != digit, then use
      torch.utils.data.Subset to create a view of those indices only.
      Subset doesn't copy data — it just holds a list of indices and
      redirects __getitem__ calls through that list.
    """
    train_data = datasets.MNIST(
        "./data", train=True, download=True, transform=get_transform()
    )

    # train_data.targets is a tensor of shape [60000] containing labels 0-9
    # (label != digit) produces a boolean mask of the same shape
    # .nonzero(as_tuple=True)[0] extracts the indices where mask is True
    keep_indices = (train_data.targets != digit).nonzero(as_tuple=True)[0].tolist()

    print(f"  Training set size without digit {digit}: "
          f"{len(keep_indices)} / {len(train_data)} examples")

    subset = Subset(train_data, keep_indices)
    # Subset wraps the dataset and makes it behave as if those are the only examples.
    # The DataLoader sees it as a normal dataset — it doesn't know about the filter.

    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)


def get_test_loaders(digit):
    """
    Returns two test loaders:
      - full_loader   : all 10,000 test examples (overall accuracy)
      - digit_loader  : only examples of `digit` (the held-out class performance)
      - other_loader  : everything except `digit` (performance on seen classes)

    Separating these lets us see exactly how much dark knowledge transferred.
    """
    test_data = datasets.MNIST(
        "./data", train=False, download=True, transform=get_transform()
    )

    digit_indices = (test_data.targets == digit).nonzero(as_tuple=True)[0].tolist()
    other_indices = (test_data.targets != digit).nonzero(as_tuple=True)[0].tolist()

    print(f"  Test set: {len(digit_indices)} examples of digit {digit}, "
          f"{len(other_indices)} examples of other digits")

    full_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    digit_loader = DataLoader(Subset(test_data, digit_indices),  batch_size=BATCH_SIZE, shuffle=False)
    other_loader = DataLoader(Subset(test_data, other_indices),  batch_size=BATCH_SIZE, shuffle=False)

    return full_loader, digit_loader, other_loader


# ---------------------------------------------------------------------------
# Soft-only distillation loss
# ---------------------------------------------------------------------------

def soft_only_loss(student_logits, teacher_logits, T):
    """
    Pure soft-target loss — no hard labels at all.

    This is the extreme version of the distillation loss:
      alpha = 1.0, so the hard-label term disappears entirely.

    Since the student never sees digit 3, there ARE no correct hard labels
    to learn from for that class. We rely entirely on the teacher's soft
    outputs to transfer whatever signal it carries about class 3.

    This is exactly what Hinton describes in the held-out experiment:
    training purely on soft targets with no ground truth labels.

    L = KL(softmax(student/T) || softmax(teacher/T)) * T^2
    """
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_student_soft_only(student, teacher, loader, optimizer, epoch):
    """
    Trains the student using ONLY the teacher's soft targets.
    No hard labels. No ground truth. Pure dark knowledge transfer.

    The teacher is set to eval() to disable its dropout, so its outputs
    are deterministic — we want consistent soft targets, not noisy ones.
    """
    student.train()
    teacher.eval()

    total_loss = 0
    for images, _ in loader:
        # Note: we unpack labels as _ (throwaway) — we don't use them at all.
        # The student only sees images and the teacher's response to them.
        images = images.to(DEVICE)

        optimizer.zero_grad()
        student_logits = student(images)

        with torch.no_grad():
            teacher_logits = teacher(images)
            # Teacher forward pass with no gradient tracking.
            # Its weights are frozen — we just use it as an oracle.

        loss = soft_only_loss(student_logits, teacher_logits, T=TEMPERATURE)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"  [Student] Epoch {epoch:3d} | Loss: {total_loss / len(loader):.4f}")


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------

def correct_bias_for_held_out_digit(model, teacher, full_train_loader, digit):
    """
    Adjusts the output bias for the held-out digit after training.

    The problem: the student trained without any 3s in the transfer set.
    The soft targets from the teacher occasionally put small probability on
    class 3, but much less than on digits that were actually present.
    As a result, the student's bias for class 3 will be miscalibrated —
    it will be biased downward, suppressing predictions of 3.

    Hinton mentions this in the paper: "the bias for the 3 needs to be
    adjusted" after training on the transfer set.

    Fix: run the full training set (including 3s) through the teacher,
    collect the teacher's average logit for digit 3, and set the student's
    output bias for digit 3 to match it.

    This is a post-hoc surgical correction, not retraining.
    """
    print(f"\n  Correcting output bias for digit {digit}...")

    model.eval()
    teacher.eval()

    teacher_logit_sum = 0.0
    student_logit_sum = 0.0
    count = 0

    with torch.no_grad():
        for images, labels in full_train_loader:
            images = images.to(DEVICE)

            # Only look at examples OF the held-out digit
            # to measure the true expected logit for that class
            mask = (labels == digit)
            if mask.sum() == 0:
                continue

            digit_images = images[mask]
            teacher_logits = teacher(digit_images)
            student_logits = model(digit_images)

            teacher_logit_sum += teacher_logits[:, digit].sum().item()
            student_logit_sum += student_logits[:, digit].sum().item()
            count += mask.sum().item()

    if count == 0:
        print("  No examples found for bias correction (check data loader).")
        return

    teacher_mean = teacher_logit_sum / count
    student_mean = student_logit_sum / count
    correction   = teacher_mean - student_mean

    print(f"  Teacher mean logit for digit {digit}: {teacher_mean:.4f}")
    print(f"  Student mean logit for digit {digit}: {student_mean:.4f}")
    print(f"  Bias correction applied:              {correction:+.4f}")

    # Directly modify the bias parameter of the output layer (fc3)
    # model.fc3.bias is a tensor of shape [10], one entry per class.
    # We add the correction to the entry for the held-out digit only.
    with torch.no_grad():
        model.fc3.bias[digit] += correction
        # torch.no_grad() is required here too — we're directly modifying
        # a parameter tensor, not going through a forward pass.


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_detailed(model, full_loader, digit_loader, other_loader, digit):
    """
    Runs three evaluations:
      1. Full test set — overall accuracy
      2. Held-out digit only — did dark knowledge transfer?
      3. All other digits — did learning them suffer from the absence of 3?

    Also prints a per-class breakdown so you can see exactly where
    the model succeeds and fails.
    """
    model.eval()

    def count_correct(loader):
        correct = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
        return correct

    # --- Overall ---
    total    = len(full_loader.dataset)
    correct  = count_correct(full_loader)
    errors   = total - correct
    print(f"\n  Overall  : {correct}/{total} correct | {errors} errors "
          f"({100*correct/total:.1f}%)")

    # --- Held-out digit ---
    d_total   = len(digit_loader.dataset)
    d_correct = count_correct(digit_loader)
    d_errors  = d_total - d_correct
    print(f"  Digit {digit} only : {d_correct}/{d_total} correct | {d_errors} errors "
          f"({100*d_correct/d_total:.1f}%) ← never seen during training")

    # --- All other digits ---
    o_total   = len(other_loader.dataset)
    o_correct = count_correct(other_loader)
    o_errors  = o_total - o_correct
    print(f"  Other digits : {o_correct}/{o_total} correct | {o_errors} errors "
          f"({100*o_correct/o_total:.1f}%)")

    # --- Per-class breakdown ---
    print(f"\n  Per-class breakdown:")
    print(f"  {'Digit':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*38}")

    with torch.no_grad():
        class_correct = torch.zeros(10)
        class_total   = torch.zeros(10)
        for images, labels in full_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            for c in range(10):
                mask = (labels == c)
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    for c in range(10):
        tag = " ← held out" if c == digit else ""
        acc = 100 * class_correct[c] / class_total[c]
        print(f"  {c:<8} {int(class_correct[c]):>8} {int(class_total[c]):>8} "
              f"{acc:>9.1f}%{tag}")

    return errors, d_errors, d_correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    print(f"Device : {DEVICE}")
    print(f"Held-out digit : {HELD_OUT_DIGIT}")
    print(f"Temperature    : {TEMPERATURE}\n")

    # -----------------------------------------------------------------------
    # Step 1: Load the pretrained teacher
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading pretrained teacher")
    print("=" * 60)

    teacher = TeacherNet().to(DEVICE)
    load_model(teacher, f"{SAVE_DIR}/teacher.pt")
    teacher.eval()
    # Freeze the teacher permanently — we never update its weights in this script.
    # Setting eval() disables dropout so it gives deterministic soft targets.

    # Also get a full train loader (includes 3s) — needed for bias correction later
    full_train_data = datasets.MNIST(
        "./data", train=True, download=True, transform=get_transform()
    )
    full_train_loader = DataLoader(full_train_data, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------------------------
    # Step 2: Build the filtered training set (no 3s)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"STEP 2: Building transfer set — digit {HELD_OUT_DIGIT} removed")
    print("=" * 60)

    transfer_loader = get_train_loader_without_digit(HELD_OUT_DIGIT)

    # -----------------------------------------------------------------------
    # Step 3: Build test loaders
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 3: Preparing test sets")
    print("=" * 60)

    full_test_loader, digit_test_loader, other_test_loader = get_test_loaders(HELD_OUT_DIGIT)

    # -----------------------------------------------------------------------
    # Step 4: Train student on soft targets — no 3s, no hard labels
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"STEP 4: Training student on soft targets (digit {HELD_OUT_DIGIT} withheld)")
    print("=" * 60)
    print(f"  Loss: pure soft-target KL divergence (alpha=1.0)")
    print(f"  The student will never see a single image of digit {HELD_OUT_DIGIT}.\n")

    student = StudentNet().to(DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=LR)

    for epoch in range(1, STUDENT_EPOCHS + 1):
        train_student_soft_only(student, teacher, transfer_loader, optimizer, epoch)

    # -----------------------------------------------------------------------
    # Step 5: Evaluate BEFORE bias correction
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"STEP 5: Evaluation BEFORE bias correction")
    print("=" * 60)
    print("  (Student's output bias for digit 3 is miscalibrated —")
    print("   it learned to suppress class 3 since it never appeared as a target)\n")

    evaluate_detailed(
        student, full_test_loader, digit_test_loader, other_test_loader, HELD_OUT_DIGIT
    )

    # -----------------------------------------------------------------------
    # Step 6: Apply bias correction for the held-out digit
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"STEP 6: Bias correction for digit {HELD_OUT_DIGIT}")
    print("=" * 60)
    print("  Hinton notes the output bias for the held-out digit must be")
    print("  adjusted after training. We compute the gap between the teacher's")
    print("  average logit for digit 3 and the student's, then add it directly")
    print("  to the student's output bias weight for class 3.\n")

    correct_bias_for_held_out_digit(student, teacher, full_train_loader, HELD_OUT_DIGIT)

    # -----------------------------------------------------------------------
    # Step 7: Evaluate AFTER bias correction
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"STEP 7: Evaluation AFTER bias correction")
    print("=" * 60)

    _, d_errors_after, d_correct_after = evaluate_detailed(
        student, full_test_loader, digit_test_loader, other_test_loader, HELD_OUT_DIGIT
    )

    # -----------------------------------------------------------------------
    # Step 8: Save the trained student
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("STEP 8: Saving student checkpoint")
    print("=" * 60)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = f"{SAVE_DIR}/student_no_{HELD_OUT_DIGIT}s.pt"
    torch.save({
        "model_state_dict"    : student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch"               : STUDENT_EPOCHS,
        "held_out_digit"      : HELD_OUT_DIGIT,
        "digit_correct_after" : d_correct_after,
        "digit_errors_after"  : d_errors_after,
        "temperature"         : TEMPERATURE,
    }, save_path)
    print(f"  Saved to {save_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY — What just happened")
    print("=" * 60)
    print(f"  A student was trained on every digit EXCEPT {HELD_OUT_DIGIT}.")
    print(f"  It never saw a single training image of digit {HELD_OUT_DIGIT}.")
    print(f"  Its only exposure to class {HELD_OUT_DIGIT} was the small residual")
    print(f"  probability the teacher assigned to class {HELD_OUT_DIGIT} when")
    print(f"  predicting other digits — Hinton's 'dark knowledge'.")
    print()
    print(f"  After bias correction, the student correctly classified")
    print(f"  {d_correct_after} / {len(digit_test_loader.dataset)} test examples")
    print(f"  of digit {HELD_OUT_DIGIT} it had never seen.")
    print()
    print(f"  Paper's claim: the student learns digit {HELD_OUT_DIGIT} almost")
    print(f"  entirely from the structure encoded in the teacher's wrong-class")
    print(f"  probability assignments — not from ever seeing a {HELD_OUT_DIGIT}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
