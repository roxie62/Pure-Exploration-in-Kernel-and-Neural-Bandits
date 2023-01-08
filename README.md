# Pure exploration in Kernel and Neural Bandits

This repository is the official implementation of [Pure exploration in Kernel and Neural Bandits](https://openreview.net/pdf?id=X_jSy6seRj)

## Requirements

To install requirements:

>Packages used in this folder include: numpy, functools, scipy, sklearn, math, sys, logging, torch, itertools, pickle, gzip.

## Training and Evaluation

This paper includes results on two sythentic datasets and two real datasets.

To run the model(s) in the paper, run this command:

##### Yahoo dataset

To load the data, please run process_yahoo.py first. After running, we will see two generated numpy files: yahoo_features.npy and yahoo_targets.npy.

To reproduce the result in Section 6, use the following command:

method_list = [neural_elim, kernel_elim, linear_elim]

For method in method_list:

```train and evaluate
python run_yahoo.py method
```

For example, to run Alg.2 NeuralEmbedding, we use:

```train and evaluate
python run_yahoo.py neural_elim
```

##### MNIST dataset

mnist.pkl contains the raw data of the MNIST dataset.

```train and evaluate
- python run_minst.py method
```

##### Linear dataset

```train and evaluate
- python run_linear_data.py method
```

##### Nonlinear dataset

```train and evaluate
- python run_nonlinear_data.py method
```


## Results

Our model achieves the following performance on [Mnist Dataset](http://yann.lecun.com/exdb/mnist/) and [Yahoo Dataset](https://webscope.sandbox.yahoo.com/?guccounter=1  style="zoom:50%;" />)

##### Sample complexity
- MNIST dataset:
<img src="https://github.com/roxie62/Pure-Exploration-in-Kernel-and-Neural-Bandits/blob/master/plots/bar_plot_ready_mnist.png" alt="drawing" width="500"/>

- Yahoo dataset:
<img src="https://github.com/roxie62/Pure-Exploration-in-Kernel-and-Neural-Bandits/blob/master/plots/bar_plot_ready_yahoo.png" alt="drawing" width="500"/>

##### Success rate

|               | Neural Elimination | Kernel Embedding | Linear Embedding | RAGE | Action Elimination |
| ------------- | ------------------ | ---------------- | ---------------- | ---- | ------------------ |
| MNIST Dataset | 98%                | 100%             | 100%             | 100% | 100%               |
| Yahoo Dataset | 100%               | 98%              | 88%              | 90%  | 100%               |

## Acknowledgement

We would like to thank [Sequential Experimental Design for Transductive Linear Bandits](https://proceedings.neurips.cc/paper/2019/file/8ba6c657b03fc7c8dd4dff8e45defcd2-Paper.pdf) for open-source code.

## Citations

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.
```citations
@article{zhu2021pure,
  title={Pure Exploration in Kernel and Neural Bandits},
  author={Zhu, Yinglun and Zhou, Dongruo and Jiang, Ruoxi and Gu, Quanquan and Willett, Rebecca and Nowak, Robert},
  journal={arXiv preprint arXiv:2106.12034},
  year={2021}
}
```
