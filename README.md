# ContinualLearning

Metrics from Jenell (Terminal Results):
(avalanche_py310) jenells@dhcp-10-5-223-69 ContinualLearning-2 % python experiments/run_cl_experiment.py \
  --algo er \
  --dataset cifar10 \
  --n-experiences 5 \
  --epochs 1 \
  --seed 0
Using bfloat16 Automatic Mixed Precision (AMP)
/opt/anaconda3/envs/avalanche_py310/lib/python3.10/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
[Exp 0] acc=78.00% AA=78.00% updates=79 time=31.0s buffer=2000
[Exp 1] acc=87.50% AA=82.75% updates=79 time=29.1s buffer=2000
[Exp 2] acc=65.55% AA=77.02% updates=79 time=30.7s buffer=2000
[Exp 3] acc=81.25% AA=78.08% updates=79 time=30.9s buffer=2000
[Exp 4] acc=80.15% AA=78.49% updates=79 time=27.7s buffer=2000

FINAL RESULTS
Average accuracy: 78.49%
Average forgetting: 62.46%

Metrics saved to metrics_seed0.csv
(avalanche_py310) jenells@dhcp-10-5-223-69 ContinualLearning-2 % python experiments/run_cl_experiment.py --algo er --dataset cifar10 --n-experiences 5 --epochs 1 --seed 1
Using bfloat16 Automatic Mixed Precision (AMP)
/opt/anaconda3/envs/avalanche_py310/lib/python3.10/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
[Exp 0] acc=94.25% AA=94.25% updates=79 time=31.8s buffer=2000
[Exp 1] acc=74.75% AA=84.50% updates=79 time=28.3s buffer=2000
[Exp 2] acc=55.70% AA=74.90% updates=79 time=40.1s buffer=2000
[Exp 3] acc=84.15% AA=77.21% updates=79 time=31.9s buffer=2000
python(8855) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8858) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8864) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8867) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8872) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8875) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8885) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8888) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8893) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8896) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8907) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8910) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8920) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
python(8927) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.
[Exp 4] acc=70.75% AA=75.92% updates=79 time=35.8s buffer=2000

FINAL RESULTS
Average accuracy: 75.92%
Average forgetting: 61.77%

Metrics saved to metrics_seed1.csv
(avalanche_py310) jenells@dhcp-10-5-223-69 ContinualLearning-2 % python experiments/run_cl_experiment.py --algo er --dataset cifar10 --n-experiences 5 --epochs 1 --seed 2
Using bfloat16 Automatic Mixed Precision (AMP)
/opt/anaconda3/envs/avalanche_py310/lib/python3.10/site-packages/torch/utils/data/dataloader.py:692: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, device pinned memory won't be used.
  warnings.warn(warn_msg)
[Exp 0] acc=62.45% AA=62.45% updates=79 time=30.8s buffer=2000
[Exp 1] acc=70.00% AA=66.22% updates=79 time=28.1s buffer=2000
[Exp 2] acc=72.85% AA=68.43% updates=79 time=27.8s buffer=2000
[Exp 3] acc=78.05% AA=70.84% updates=79 time=29.0s buffer=2000
[Exp 4] acc=75.75% AA=71.82% updates=79 time=29.7s buffer=2000

FINAL RESULTS
Average accuracy: 71.82%
Average forgetting: 55.86%

Metrics saved to metrics_seed2.csv
(avalanche_py310) jenells@dhcp-10-5-223-69 ContinualLearning-2 % 