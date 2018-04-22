# Deep Convolutional For Tiny Images

## Requirments:

+ Python 2.7

+ Tensorflow

## Dataset

+ Dataset: CIFAR-10, CIFAR-100 ([Original dataset link](https://www.cs.toronto.edu/~kriz/cifar.html))

+ Dataset use for training and testing is available [here](https://drive.google.com/file/d/14h-sRyIXzNZTUMNn1GMXN8pJOZAxPJcg/view?usp=sharing)

## Training

```bash
    python train.py --dataset cifar10 --learning_rate 0.001 --regularization_rate 0.005 --batch_size 128 --num_epoch 100
```

+ For more detail:

```bash
    python train.py -h
```

## Testing

```bash
    python test.py --dataset cifar10  --batch_size 128
```

+ For more detail:

```bash
    python test.py -h
```

## Citation

Please cite our paper in your publications if it helps your research:

```latex
@conference{indeed18,
author={Thanh-Dat Truong and Vinh-Tiep Nguyen and Minh-Triet Tran},
title={Lightweight Deep Convolutional Network for Tiny Object Recognition},
booktitle={Proceedings of the 7th International Conference on Pattern Recognition Applications and Methods - Volume 1: INDEED,},
year={2018},
pages={675-682},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0006752006750682},
isbn={978-989-758-276-9},
}
```


## Author:

### Thanh-Dat Truong

+ University of Science, Vietnam National University, Ho Chi Minh City

+ Email: ttdat@selab.hcmus.edu.vn
