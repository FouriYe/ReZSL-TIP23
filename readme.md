# Official PyTorch implementation of "Rebalanced Zero-shot Learning" [(TIP 2023 Paper)](https://arxiv.org/abs/2210.07031) #

<div align="center">
    <img width="800" alt="banner" src="banner.pdf"/>
</div>

Zero-shot learning (ZSL) aims to identify unseen classes with zero samples during training.
Broadly speaking, present ZSL methods usually adopt class-level semantic labels and compare them with instance-level semantic predictions to infer unseen classes.
However, we find that such existing models mostly produce imbalanced semantic predictions, i.e. these models could perform precisely for some semantics, but  may not for others. To address the drawback, we aim to introduce an imbalanced learning framework into ZSL. However, we find that imbalanced ZSL has two unique challenges: (1) Its imbalanced predictions are highly correlated with the value of semantic labels rather than the number of samples as typically considered in the traditional imbalanced learning; (2) Different semantics follow quite different error distributions between classes. To mitigate these issues, we first formalize ZSL as an imbalanced regression problem  which offers empirical evidences to interpret how semantic labels lead to imbalanced semantic predictions. We then propose a re-weighted loss termed Re-balanced Mean-Squared Error (ReMSE), which tracks the mean and variance of error distributions, thus ensuring rebalanced learning across classes. As a major contribution, we conduct a series of analyses showing that ReMSE is theoretically well established. Extensive experiments demonstrate that the proposed method effectively alleviates the imbalance in semantic prediction and outperforms many state-of-the-art ZSL methods.

The code would be available within ONE MONTH.

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for additional details.

## Bibtex ##
Cite our paper using the following bibtex item:

```
@ARTICLE{
ye2023rebalanced,
title={Rebalanced Zero-shot Learning},
author={Zihan Ye and Guanyu Yang and Xiaobo Jin and Youfa Liu and Kaizhu Huang},
journal={IEEE transactions on image processing},
doi={10.1109/TIP.2023.3295738}},
publisher={IEEE},
year={2023}
}
```
