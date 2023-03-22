# Code for computing the number of parameters for flow models.
'''Original''':https://github.com/rtqichen/ffjord
## For FFJORD on CIFAR10 DATASET

python /experments/ffjord/train_cnf.py --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 4 --layer_type concat --multiscale True --rademacher True

'''NB of parameters on CIFAR 10: 2717764'''

## For Flowification
'''Original:''' https://github.com/balintmate/flowification
### Environment setup:
conda create --name flowification python=3.8
conda activate flowification
conda install pytorch==1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install wandb nflows

python /experiments/image_generation.py --data cifar --architecture conv2

'''NB of parameters on CIFAR 10: 9420455'''

## For Sinusoidal Flow: A Fast Invertible Autoregressive Flow
'''Original:''' https://github.com/weiyumou/ldu-flow
### Environment setup:
cd ldu-flow
pip install -e .

python -m project.experiments.ldu.count_ldu_parameters --dataset cifar10 --test_checkpoints

'''NB of parameters on CIFAR 10: 17458632'''

