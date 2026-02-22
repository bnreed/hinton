"""
Hinton et al. (2015) - "Distilling the Knowledge in a Neural Network"
MNIST Experiment Implementation — Annotated for PyTorch novices

Paper specifications:
  Teacher : 2 hidden layers x 1200 ReLU units, dropout(0.2 input, 0.5 hidden)
  Student : 2 hidden layers x 800  ReLU units, no dropout
  Data    : MNIST, images jittered up to 2px in any direction
  Loss    : alpha * L_soft(T) + (1-alpha) * L_hard, scaled by T^2

Usage:
  pip install torch torchvision
  python hinton_mnist_distillation_annotated.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os                                  # for creating directories
import torch                              # the core PyTorch library
import torch.nn as nn                     # building blocks for neural networks (layers, etc.)
import torch.nn.functional as F           # stateless functions: relu, softmax, cross_entropy, etc.
import torch.optim as optim               # optimizers: Adam, SGD, etc.
from torchvision import datasets, transforms  # standard datasets (MNIST) and image transforms
from torch.utils.data import DataLoader   # wraps a dataset and serves batches during training


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE      = 128     # how many images to process at once during training
TEACHER_EPOCHS  = 100     # full passes through the training data for the teacher
STUDENT_EPOCHS  = 100     # full passes through the training data for the student
LR              = 1e-3    # learning rate — how large a step the optimizer takes each update
TEMPERATURE     = 8.0     # T in Equation 1; higher = softer probability distributions
ALPHA           = 0.9     # weight on soft loss; (1-ALPHA) goes to hard loss
                          # paper recommends close to 1.0 so soft targets dominate
SEED            = 42      # random seed for reproducibility
SAVE_DIR        = "./saved_models"  # directory where trained models will be written
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                          # use a GPU if one is available, otherwise fall back to CPU
                          # torch.cuda.is_available() returns True if CUDA GPU is detected


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloaders():
    """
    Loads MNIST and returns DataLoader objects for training and testing.

    A DataLoader is an iterator that feeds the model data in batches.
    Each call to next() on a DataLoader returns (images, labels) for one batch.

    Transforms are applied to each image before it enters the network:
      - RandomAffine: randomly shifts the image up to 2 pixels (the "jitter" from the paper)
      - ToTensor: converts a PIL image (H x W, values 0-255) to a
                  PyTorch tensor (C x H x W, values 0.0-1.0)
      - Normalize: shifts pixel values to have mean=0.1307, std=0.3081
                   (precomputed statistics for MNIST)
                   normalized inputs train faster and more stably
    """

    # training images get the jitter augmentation; test images do not
    # (you never augment test data — that would change what you're measuring)
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),
        # translate=(2/28, 2/28) means shift up to 2/28 of image width/height
        # = 2 pixels on a 28x28 image, matching the paper exactly
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # download=True fetches the data from the internet on first run
    # subsequent runs use the cached copy in ./data
    train_data = datasets.MNIST("./data", train=True,  download=True, transform=train_transform)
    test_data  = datasets.MNIST("./data", train=False, download=True, transform=test_transform)

    # shuffle=True randomizes the order each epoch so the model doesn't
    # memorize the sequence of batches
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------

class TeacherNet(nn.Module):
    """
    The "big net" from the paper: 2 hidden layers of 1200 ReLU units.
    Dropout: 0.2 on the input layer, 0.5 after the first hidden layer.

    All PyTorch models inherit from nn.Module. The two methods you must
    define are:
      __init__  : declare all layers as attributes
      forward   : describe how data flows through those layers

    Layers declared in __init__ are automatically registered as model
    parameters, so the optimizer knows what to update.
    """
    def __init__(self):
        super().__init__()
        # Dropout randomly zeroes a fraction of activations during training.
        # p=0.2 means 20% of inputs are zeroed; p=0.5 means 50%.
        # This forces the network to learn redundant representations
        # and is Hinton's main regularization tool here.
        self.drop_input  = nn.Dropout(p=0.2)

        # nn.Linear(in, out) is a fully connected layer.
        # It computes: output = input @ weight.T + bias
        # where weight has shape [out, in] and bias has shape [out].
        # 784 = 28*28 pixels flattened; 1200 = hidden units per layer.
        self.fc1         = nn.Linear(784, 1200)
        self.drop_hidden = nn.Dropout(p=0.5)
        self.fc2         = nn.Linear(1200, 1200)

        # Output layer: 10 units, one per digit class (0-9).
        # No activation here — we output raw logits (z_i in the paper).
        # The softmax is applied inside the loss function, not the model.
        self.fc3         = nn.Linear(1200, 10)

    def forward(self, x):
        # x arrives with shape [batch_size, 1, 28, 28]
        # (batch of grayscale 28x28 images)

        x = x.view(-1, 784)
        # .view() reshapes the tensor without copying data.
        # -1 means "infer this dimension" = batch_size.
        # Result shape: [batch_size, 784]

        x = self.drop_input(x)
        # Dropout only has an effect during training.
        # During evaluation (.eval() mode), it passes data through unchanged.

        x = F.relu(self.fc1(x))
        # F.relu clamps all negative values to zero: relu(z) = max(0, z)
        # This is the non-linearity that lets the network learn complex functions.
        # Without it, stacking linear layers would still just be one linear layer.

        x = self.drop_hidden(x)
        x = F.relu(self.fc2(x))

        return self.fc3(x)
        # Returns raw logits — shape [batch_size, 10]
        # These are the z_i values from Equation 1 in the paper.
        # No softmax here; it's applied inside the loss functions below.


class StudentNet(nn.Module):
    """
    The "small net" from the paper: 2 hidden layers of 800 ReLU units.
    No dropout — the student is trained to mimic the teacher,
    and the soft targets already act as a form of regularization.

    Parameter count comparison (weights only):
      Teacher: 784*1200 + 1200*1200 + 1200*10 = ~2.4M parameters
      Student: 784*800  + 800*800   + 800*10  = ~1.3M parameters
      Student is ~53% the size of the teacher.
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
        return self.fc3(x)    # raw logits, shape [batch_size, 10]


# ---------------------------------------------------------------------------
# Loss function — the core of distillation
# ---------------------------------------------------------------------------

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Implements the combined loss from Section 2 of the paper:

        L = alpha * L_soft + (1 - alpha) * L_hard

    Args:
        student_logits : raw logits from the student network  [batch_size, 10]
        teacher_logits : raw logits from the teacher network  [batch_size, 10]
                         these are treated as fixed targets — no gradients flow
                         through them (enforced by torch.no_grad() at call site)
        labels         : ground-truth digit class [batch_size], integers 0-9
        T              : temperature — controls softness of the distributions
        alpha          : blending weight; paper recommends close to 1.0
    """

    # --- Soft target loss (L_soft) ---
    # This is the distillation signal: train the student to match the
    # teacher's full probability distribution, not just its top prediction.

    soft_student = F.log_softmax(student_logits / T, dim=1)
    # F.log_softmax computes log(softmax(x)) in a numerically stable way.
    # Dividing by T first is exactly Equation 1 from the paper.
    # dim=1 means softmax is computed across the 10 class scores for each image.
    # Shape: [batch_size, 10], values are log-probabilities (negative numbers)

    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    # F.softmax converts teacher logits to a probability distribution at temperature T.
    # These are the "soft targets" p_i from the paper.
    # Shape: [batch_size, 10], values are probabilities (sum to 1 per row)

    L_soft = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    # F.kl_div computes KL divergence: sum(p_teacher * log(p_teacher / p_student))
    # which is equivalent to cross-entropy here since teacher targets are constant.
    # It expects: first arg = log-probabilities (student), second = probabilities (teacher)
    #
    # reduction="batchmean" divides by batch size, giving the average loss per image.
    #
    # * (T ** 2): the T^2 scaling Hinton describes after Equation 3.
    # Without this, the soft-target gradients shrink as 1/T^2 when T is large,
    # making the soft loss negligible compared to the hard loss.
    # Multiplying by T^2 keeps the two loss terms on the same scale
    # regardless of what T you choose.

    # --- Hard label loss (L_hard) ---
    # Standard training signal: did the student predict the correct digit?
    # This always uses T=1 (no temperature) — we pass raw logits directly.

    L_hard = F.cross_entropy(student_logits, labels)
    # F.cross_entropy combines softmax + negative log likelihood in one step.
    # For the correct class c: loss = -log(softmax(student_logits)[c])
    # It penalizes the student for putting low probability on the right answer.
    # labels contains integers (0-9), not one-hot vectors.

    # --- Combine ---
    return alpha * L_soft + (1 - alpha) * L_hard
    # alpha=0.9 means 90% of the gradient signal comes from matching the teacher,
    # 10% from getting the hard label right.
    # The hard label term acts as a regularizer to keep the student anchored
    # to correct predictions when soft targets don't fully constrain the output.


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_teacher(model, loader, optimizer, epoch):
    """
    Standard supervised training — no distillation involved.
    The teacher is just trained to classify digits correctly.
    """
    model.train()
    # .train() switches the model to training mode.
    # This enables dropout (which is disabled during evaluation).
    # BatchNorm layers (not used here) also behave differently in train vs eval.

    total_loss = 0
    for images, labels in loader:
        # loader yields (images, labels) pairs one batch at a time.
        # images shape: [batch_size, 1, 28, 28]
        # labels shape: [batch_size], integers 0-9

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # .to(DEVICE) moves tensors to GPU if available.
        # Both the model and its inputs must be on the same device.

        optimizer.zero_grad()
        # PyTorch accumulates gradients by default — we must clear them
        # at the start of each batch, otherwise they compound across batches.

        logits = model(images)
        # Forward pass: runs images through the network.
        # Calls model.forward(images) under the hood.
        # logits shape: [batch_size, 10]

        loss = F.cross_entropy(logits, labels)
        # Compute how wrong the predictions are.

        loss.backward()
        # Backpropagation: computes the gradient of loss with respect to
        # every parameter in the model using the chain rule.
        # Gradients are stored in each parameter's .grad attribute.

        optimizer.step()
        # Updates each parameter by taking a step in the negative gradient direction:
        #   param = param - lr * param.grad   (simplified; Adam is more complex)

        total_loss += loss.item()
        # .item() converts a single-element tensor to a plain Python float.

    avg_loss = total_loss / len(loader)
    print(f"  [Teacher] Epoch {epoch:3d} | Loss: {avg_loss:.4f}")


def train_student(student, teacher, loader, optimizer, epoch):
    """
    Distillation training — student learns from teacher's soft targets.
    The teacher's weights are frozen; only the student's weights update.
    """
    student.train()   # student in training mode (enables any dropout if present)
    teacher.eval()    # teacher in eval mode — we're not training it, just querying it
                      # .eval() disables dropout so teacher outputs are deterministic

    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        student_logits = student(images)
        # Forward pass through student — gradients will flow through this.

        with torch.no_grad():
            teacher_logits = teacher(images)
        # torch.no_grad() is a context manager that disables gradient tracking.
        # We don't need gradients through the teacher because we're not updating
        # its weights. This also saves memory and speeds up the forward pass.

        loss = distillation_loss(
            student_logits, teacher_logits, labels,
            T=TEMPERATURE, alpha=ALPHA
        )
        loss.backward()   # gradients flow through student_logits only
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"  [Student] Epoch {epoch:3d} | Loss: {avg_loss:.4f}")


def train_student_baseline(student, loader, optimizer, epoch):
    """
    Baseline: train student directly on hard labels, no teacher involved.
    This is what the paper compares against — the 146-error result.
    The distilled student (74 errors) should outperform this significantly.
    """
    student.train()
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = student(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, label="Model"):
    """
    Runs the model on the test set and reports accuracy and error count.
    No gradients needed here — we're just measuring performance.
    At inference, temperature is always 1 (standard softmax).
    """
    model.eval()
    # .eval() disables dropout so every forward pass gives the same result.
    # If you forget this, dropout will randomly zero activations during evaluation,
    # giving you different (and worse) results each time you run it.

    correct = 0
    with torch.no_grad():
        # Wrap evaluation in no_grad to skip gradient computation entirely.
        # Saves memory and runs faster — we only need the forward pass output.
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)          # shape: [batch_size, 10]
            preds = logits.argmax(dim=1)    # pick the class with the highest logit
                                            # argmax along dim=1 = across the 10 classes
                                            # shape: [batch_size]
            correct += (preds == labels).sum().item()
            # (preds == labels) is a boolean tensor [batch_size]
            # .sum() counts the True values = number of correct predictions
            # .item() converts to a Python int

    errors = len(loader.dataset) - correct
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"  [{label}] Accuracy: {accuracy:.2f}% | Errors: {errors}/{len(loader.dataset)}")
    return errors


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------

def save_model(model, optimizer, epoch, errors, filename):
    """
    Saves a model checkpoint to disk so training can be resumed later
    or the model can be reloaded without retraining.

    PyTorch convention is to save a "checkpoint" dict containing:
      - model state dict   : all the learned weights and biases
      - optimizer state    : momentum/variance terms so training can resume
                             smoothly from exactly where it left off
      - metadata           : anything else useful (epoch, errors, config)

    Args:
        model     : the trained PyTorch model (nn.Module)
        optimizer : the optimizer used during training
        epoch     : how many epochs were trained (for resuming)
        errors    : test error count (for reference)
        filename  : full path to save to, e.g. "./saved_models/teacher.pt"
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # os.makedirs creates the directory (and any parent dirs) if it doesn't exist.
    # exist_ok=True means don't raise an error if it already exists.

    checkpoint = {
        "model_state_dict"    : model.state_dict(),
        # state_dict() returns an OrderedDict of all parameters and buffers.
        # Keys are layer names like "fc1.weight", "fc1.bias", etc.
        # Values are tensors containing the actual learned values.
        # This is what you need to restore the model's learned behavior.

        "optimizer_state_dict": optimizer.state_dict(),
        # Saves the optimizer's internal state (e.g. Adam's running averages).
        # Only needed if you want to resume training — not needed for inference.

        "epoch"               : epoch,
        "test_errors"         : errors,
        "temperature"         : TEMPERATURE,
        "alpha"               : ALPHA,
    }

    torch.save(checkpoint, filename)
    # torch.save uses Python's pickle under the hood to serialize the dict.
    # .pt and .pth are both conventional extensions for PyTorch checkpoints.

    print(f"  Saved to {filename}")


def load_model(model, filename, optimizer=None):
    """
    Restores a saved model from a checkpoint file.

    Args:
        model     : an instantiated (but untrained) model of the correct architecture.
                    The architecture must match what was saved — PyTorch saves weights,
                    not the architecture itself, so you must recreate the structure first.
        filename  : path to the .pt checkpoint file
        optimizer : optional — pass in if you want to resume training.
                    Skip if you only need the model for inference.

    Returns:
        The checkpoint dict (contains epoch, errors, etc.)
    """
    checkpoint = torch.load(filename, map_location=DEVICE)
    # map_location=DEVICE handles the case where the model was saved on GPU
    # but is being loaded on CPU (or vice versa).

    model.load_state_dict(checkpoint["model_state_dict"])
    # Copies the saved weights back into the model layer by layer.
    # After this call, model behaves identically to how it did when saved.

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Restores optimizer state so training can resume without a "warm-up" period.

    print(f"  Loaded from {filename} (epoch {checkpoint['epoch']}, "
          f"errors {checkpoint['test_errors']})")
    return checkpoint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    # Setting the seed makes random operations (weight initialization,
    # dropout, data shuffling) reproducible across runs.

    print(f"Device: {DEVICE}")
    print(f"Temperature: {TEMPERATURE} | Alpha: {ALPHA}\n")

    train_loader, test_loader = get_dataloaders()

    # -----------------------------------------------------------------------
    # Step 1: Train teacher on hard labels
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Training Teacher (1200-1200 units, dropout)")
    print("=" * 60)

    teacher = TeacherNet().to(DEVICE)
    # Instantiate the model and move all its parameters to the target device.

    teacher_opt = optim.Adam(teacher.parameters(), lr=LR)
    # optim.Adam is an adaptive optimizer that adjusts the learning rate
    # per-parameter based on gradient history. Generally a good default.
    # teacher.parameters() returns an iterator over all learnable tensors
    # in the model (weights and biases of all layers).

    for epoch in range(1, TEACHER_EPOCHS + 1):
        train_teacher(teacher, train_loader, teacher_opt, epoch)

    print("\nTeacher evaluation:")
    teacher_errors = evaluate(teacher, test_loader, "Teacher")
    save_model(teacher, teacher_opt, TEACHER_EPOCHS, teacher_errors,
               f"{SAVE_DIR}/teacher.pt")

    # -----------------------------------------------------------------------
    # Step 2: Train student WITH distillation (the paper's main result)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Training Student WITH Distillation (800-800 units)")
    print("=" * 60)

    student_distilled = StudentNet().to(DEVICE)
    student_dist_opt  = optim.Adam(student_distilled.parameters(), lr=LR)
    # Note: optimizer is created for student_distilled's parameters only.
    # The teacher's parameters are not passed in, so they will never be updated.

    for epoch in range(1, STUDENT_EPOCHS + 1):
        train_student(student_distilled, teacher, train_loader, student_dist_opt, epoch)

    print("\nDistilled student evaluation:")
    print("(temperature set to 1 at inference — standard softmax)")
    distilled_errors = evaluate(student_distilled, test_loader, "Distilled Student")
    save_model(student_distilled, student_dist_opt, STUDENT_EPOCHS, distilled_errors,
               f"{SAVE_DIR}/student_distilled.pt")

    # -----------------------------------------------------------------------
    # Step 3: Train student WITHOUT distillation (the baseline to beat)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Training Student WITHOUT Distillation (baseline)")
    print("=" * 60)

    student_baseline = StudentNet().to(DEVICE)
    # Fresh StudentNet — same architecture, random initialization.
    # This isolates the effect of distillation: same model, same data,
    # different training signal.

    student_base_opt = optim.Adam(student_baseline.parameters(), lr=LR)

    for epoch in range(1, STUDENT_EPOCHS + 1):
        train_student_baseline(student_baseline, train_loader, student_base_opt, epoch)
        if epoch % 10 == 0:
            print(f"  [Baseline] Epoch {epoch:3d}")

    print("\nBaseline student evaluation:")
    baseline_errors = evaluate(student_baseline, test_loader, "Baseline Student")
    save_model(student_baseline, student_base_opt, STUDENT_EPOCHS, baseline_errors,
               f"{SAVE_DIR}/student_baseline.pt")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
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
    # This block only runs when the script is executed directly,
    # not when it's imported as a module by another script.
    main()
