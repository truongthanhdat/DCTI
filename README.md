# Deep Convolutional For Tiny Images

## Requirments:

+ Python 2.7

+ Tensorflow

## Dataset

+ Dataset: CIFAR-10, CIFAR-100 ([Original dataset link](https://www.cs.toronto.edu/~kriz/cifar.html))

+ Dataset use for training and testing is available [here]

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


## Author:

### Thanh-Dat Truong

+ University of Science, Vietnam National University, Ho Chi Minh City

+ Email: thanhdattrg@gmail.com

### Vinh-Tiep Nguyen

+ University of Information Technology, Vietnam National University, Ho Chi Minh City

+ Email: tiepnv@uit.edu.vn

### Minh-Triet Tran

+ University of Science, Vietnam National University, Ho Chi Minh City

+ Email: tmtriet@fit.hcmus.edu.vn
