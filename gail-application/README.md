# Reproducing coinrun expriements in Colab: Recommended way. 

```
import os
del os.environ['LD_PRELOAD']
!apt-get remove libtcmalloc*

!apt-get update
!apt-get install mpich build-essential qt5-default pkg-config

!git clone https://github.com/openai/coinrun

!pip install -r coinrun/requirements.txt

# alter sys path instead of pip install -e to avoid colab import issues
import sys
sys.path.insert(0, 'coinrun')

from google.colab import drive
drive.mount('/content/gdrive')

%cp -r gdrive/My\ Drive/codes/ .
%cd codes/
%cd corgail/
%cp -r /content/coinrun/coinrun .
!unzip gail_experts.zip

!git clone https://github.com/openai/baselines.git
%cd baselines
!pip install -e .
%cd ..
!pip uninstall torch
!pip install torch==1.4.0
```

## FTNPL GAIL (in codes it is referred to as corGAIL)
```
!python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --cor-gail --use-gae --log-interval 10 --num-processes 8 --lr 3e-5 --m_lr 3e-4 --entropy-coef 0.0 --num-steps 32 --ppo-epoch 10 --gail-epoch 1 --queue-size 5 --embed-size 2 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 500000 --save-interval 10 --use-linear-lr-decay --use-proper-time-limits --seed 0
```
## GAIL
```
!python main.py --env-name CoinRun --num-levels 0 --high-difficulty False --algo ppo --gail --use-gae --log-interval 10 --num-processes 8 --lr 3e-5 --entropy-coef 0.02 --num-steps 32 --ppo-epoch 10 --gail-epoch 5  --gamma 0.99 --gae-lambda 0.95 --num-env-steps 500000 --save-interval 10 --use-linear-lr-decay --use-proper-time-limits --seed 0
```

# For classic environment
## FTNPL GAIL (in codes it is referred to as corGAIL )
```
!python main.py --env-name MountainCarContinuous-v0 --algo ppo --cor-gail --use-gae --save-interval 100 --log-interval 10 --num-steps 64 --num-processes 8 --lr 3e-6 --m_lr 3e-4 --entropy-coef 0 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 5 --embed-size 1 --num-env-steps 2000000 --seed 0 --log-dir logs/car/corgail-0/

!python main.py --env-name Pendulum-v0 --algo ppo --cor-gail --use-gae --save-interval 100 --log-interval 10 --num-steps 32 --num-processes 8 --lr 3e-5 --m_lr 3e-4 --entropy-coef 0.0 --value-loss-coef 1 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 5 --embed-size 2 --num-env-steps 4000000 --seed 0 --log-dir logs/pend/corgail-0
```

## FTPL GAIL (in codes it is also referred to as noregret gail)
```
!python main.py --env-name MountainCarContinuous-v0 --algo ppo --no-regret-gail --use-gae --save-interval 100 --log-interval 10 --num-steps 64 --num-processes 8 --lr 3e-6 --entropy-coef 0.02 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 5 --num-env-steps 2000000 --seed 0 --log-dir logs/car/noregretgail-0/

!python main.py --env-name Pendulum-v0 --algo ppo --no-regret-gail --use-gae --save-interval 100 --log-interval 10 --num-steps 32 --num-processes 8 --lr 3e-6 --entropy-coef 0.02 --value-loss-coef 1 --ppo-epoch 10 --gail-epoch 1 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --queue-size 5  --num-env-steps 4000000 --seed 0 --log-dir logs/pend/noregretgail-0
```
## wGAIL
```
!python main.py --env-name MountainCarContinuous-v0 --algo ppo --gail --use-gae --save-interval 100 --log-interval 10 --num-steps 64 --num-processes 8 --lr 3e-6 --entropy-coef 0.02 --ppo-epoch 10 --gail-epoch 5 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 2000000 --seed 0 --log-dir logs/car/gail-0/

!python main.py --env-name Pendulum-v0 --algo ppo --gail --use-gae --save-interval 100 --log-interval 10 --num-steps 32 --num-processes 8 --lr 3e-6 --entropy-coef 0.02 --value-loss-coef 1 --ppo-epoch 10 --gail-epoch 5 --num-mini-batch 8 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 4000000 --seed 0 --log-dir logs/pend/gail-0
 ```

#To reproduce locally (has been only tested on Mac, python3) with the same arguments and codes as above. 
## install coinrun env from https://github.com/openai/coinrun

```
unzip gail_experts.zip
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
pip install -r requirements.txt
```

## Visualizing results

### Pendulum: Point the visualize_training_dynamics.py to the logs/pendulum/
### MountainCarContinuous: for visualization, point the visualize_training_dynamics.py to the logs/car/
### Coinrun:  Coinrun automatically saves the results in a directory. The directory name gets printed at the beginning. You have to put all the directories in any arbitrary directory with the results directory saved up with seeds-number at the end. For example `coinrun-0' `coinrun-1' `coinrun-2' in a parent directory `coinrun'. Then point the visualize_training_dynamics.py to the logs/coinrun/
