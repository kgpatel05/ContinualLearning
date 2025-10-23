# train_cifar10.py
"""
Wire-up script for the MVP:
- builds CIFAR-10 class-IL stream (5 experiences)
- ResNet-18 model (10-way head)
- ER buffer
- Fabric learner
- trains & prints per-experience accuracy + AA
"""

from lightning.fabric import Fabric

from ContinualLearning import (
    build_cifar10_cil_stream,
    build_resnet18,
    ERBuffer,
    Learner,
)

def main():
    fabric = Fabric(accelerator="auto", devices=1, precision="bf16-mixed")  # if bf16 unsupported, try "16-mixed" or omit
    fabric.launch()

    stream = build_cifar10_cil_stream(data_root="./data", n_experiences=5, seed=0)
    model = build_resnet18(num_classes=10, pretrained=False)
    buffer = ERBuffer(capacity=2000)

    learner = Learner(
        model=model,
        fabric=fabric,
        buffer=buffer,
        lr=0.03,
        weight_decay=5e-4,
        momentum=0.9,
        replay_ratio=0.5,
        batch_size=128,
        epochs=1,
        num_workers=2,
        pin_memory=True,
    )

    report = learner.fit(stream)
    fabric.print(f"\nPer-exp acc: {report.per_exp_acc}")
    fabric.print(f"Average acc: {report.avg_acc:.2f}%")
    if fabric.global_rank == 0:
        fabric.print(f"Checkpoints: {report.checkpoints}")

if __name__ == "__main__":
    main()
