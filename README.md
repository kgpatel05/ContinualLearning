# ContinualLearning

Metrics from Jenell (Terminal Results):
(base) jenells@dhcp-10-5-223-69 ContinualLearning-2 % python experiments/run_cl_experiment.py --algo er --dataset cifar10 --n-experiences 5 --seed 0
Using bfloat16 Automatic Mixed Precision (AMP)

============================================================
Algorithm: ER
Dataset: CIFAR10 (5 experiences)
Buffer capacity: 2000
Replay ratio: 0.5
============================================================

/opt/anaconda3/lib/python3.13/site-packages/torch/utils/data/dataloader.py:684: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
[Exp 0] classes=[7, 8] acc=86.35% AA=86.35% updates=79 time=34.4s buffer=2000
[Exp 1] classes=[1, 5] acc=88.80% AA=87.57% updates=79 time=28.1s buffer=2000
[Exp 2] classes=[3, 4] acc=70.20% AA=81.78% updates=79 time=28.3s buffer=2000
[Exp 3] classes=[2, 0] acc=81.05% AA=81.60% updates=79 time=29.4s buffer=2000
[Exp 4] classes=[9, 6] acc=84.25% AA=82.13% updates=79 time=29.8s buffer=2000

============================================================
FINAL RESULTS
============================================================
Average accuracy: 82.13%
Average forgetting: 64.87%
Total training time: 150.0s
Total updates: 395

Per-task forgetting: ['86.35', '88.65', '70.20', '79.15', '0.00']

Accuracy matrix:
  After exp 0: ['86.35']
  After exp 1: ['5.15', '88.80']
  After exp 2: ['12.10', '2.30', '70.20']
  After exp 3: ['0.00', '0.00', '0.00', '81.05']
  After exp 4: ['0.00', '0.15', '0.00', '1.90', '84.25']

Metrics saved to metrics_seed0.csv
(base) jenells@dhcp-10-5-223-69 ContinualLearning-2 % python experiments/run_cl_experiment.py --algo er --dataset cifar10 --n-experiences 5 --seed 1
Using bfloat16 Automatic Mixed Precision (AMP)

============================================================
Algorithm: ER
Dataset: CIFAR10 (5 experiences)
Buffer capacity: 2000
Replay ratio: 0.5
============================================================

/opt/anaconda3/lib/python3.13/site-packages/torch/utils/data/dataloader.py:684: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
[Exp 0] classes=[6, 8] acc=93.00% AA=93.00% updates=79 time=32.4s buffer=2000
[Exp 1] classes=[9, 7] acc=64.55% AA=78.78% updates=79 time=28.3s buffer=2000
[Exp 2] classes=[5, 3] acc=55.90% AA=71.15% updates=79 time=34.8s buffer=2000
[Exp 3] classes=[0, 4] acc=85.30% AA=74.69% updates=79 time=30.2s buffer=2000
[Exp 4] classes=[1, 2] acc=78.40% AA=75.43% updates=79 time=29.7s buffer=2000

============================================================
FINAL RESULTS
============================================================
Average accuracy: 75.43%
Average forgetting: 59.09%
Total training time: 155.5s
Total updates: 395

Per-task forgetting: ['92.55', '64.55', '55.90', '82.45', '0.00']

Accuracy matrix:
  After exp 0: ['93.00']
  After exp 1: ['12.95', '64.55']
  After exp 2: ['12.75', '3.15', '55.90']
  After exp 3: ['0.00', '0.00', '0.00', '85.30']
  After exp 4: ['0.45', '0.00', '0.00', '2.85', '78.40']

Metrics saved to metrics_seed1.csv
(base) jenells@dhcp-10-5-223-69 ContinualLearning-2 % python experiments/run_cl_experiment.py --algo er --dataset cifar10 --n-experiences 5 --seed 2
Using bfloat16 Automatic Mixed Precision (AMP)

============================================================
Algorithm: ER
Dataset: CIFAR10 (5 experiences)
Buffer capacity: 2000
Replay ratio: 0.5
============================================================

/opt/anaconda3/lib/python3.13/site-packages/torch/utils/data/dataloader.py:684: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
[Exp 0] classes=[5, 9] acc=83.20% AA=83.20% updates=79 time=33.2s buffer=2000
[Exp 1] classes=[3, 4] acc=61.05% AA=72.12% updates=79 time=29.2s buffer=2000
[Exp 2] classes=[6, 7] acc=78.30% AA=74.18% updates=79 time=27.8s buffer=2000
[Exp 3] classes=[2, 8] acc=70.15% AA=73.17% updates=79 time=28.6s buffer=2000
[Exp 4] classes=[1, 0] acc=75.05% AA=73.55% updates=79 time=27.7s buffer=2000

============================================================
FINAL RESULTS
============================================================
Average accuracy: 73.55%
Average forgetting: 58.54%
Total training time: 146.7s
Total updates: 395

Per-task forgetting: ['83.20', '61.05', '78.30', '70.15', '0.00']

Accuracy matrix:
  After exp 0: ['83.20']
  After exp 1: ['11.65', '61.05']
  After exp 2: ['7.15', '0.00', '78.30']
  After exp 3: ['0.00', '0.00', '0.00', '70.15']
  After exp 4: ['0.00', '0.00', '0.00', '0.00', '75.05']

Metrics saved to metrics_seed2.csv
(base) jenells@dhcp-10-5-223-69 ContinualLearning-2 % 