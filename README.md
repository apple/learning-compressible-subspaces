# Learning Compressible Subspaces

This is the official code release for our publication, [LCS: Learning Compressible Subspaces for Adaptive Network Compression at Inference Time](https://arxiv.org/abs/2110.04252). Our code is used to train and evaluate models that can be compressed in real-time after deployment, allowing for a fine-grained efficiency-accuracy trade-off.

This repository hosts code to train compressible subspaces for structured sparsity, unstructured sparsity, and quantization. We support three architectures: cPreResNet20, ResNet18, and VGG19. Training is performed on the CIFAR-10 and ImageNet datasets.

Default training configurations are provided in the `configs` folder. Note that they are automatically altered when different models and datasets are chosen through flags. See `training_params.py`. The following training parameter flags are available to all training regimes:

- `--model`: Specifies the model to use. One of cpreresnet20, resnet18, or vgg19.
- `--dataset`: Specifies the dataset to train on. One of cifar10 or imagenet.
- `--imagenet_dir`: When using imagenet dataset, the directory to the dataset must be specified.
- `--method`: Specifies the training method. For unstructured sparsity, one of target_topk, lcs_l, lcs_p. For structured sparsity, one of lec, ns, us, lcs_l, lcs_p. For quantized models, one of target_bit_width, lcs_l, lcs_p.
- `--norm`: The normalization layers to use. One of IN (instance normalization), BN (batch normalization), or GN (group normalization).
- `--epochs`: The number of epochs to train for.
- `--learning_rate`: The optimizer learning rate.
- `--batch_size`: Training and test batch sizes.
- `--momentum`: The optimizer momentum.
- `--weight_decay`: The L2 regularization weight.
- `--warmup_budget`: The percentage of epochs to use for the training method warmup phase.
- `--test_freq`: The number of training epochs to wait between test evaluation. Will also save models at this frequency.

The "lcs_l" training method refers to the "LCS+L" method in the paper. In this setting, we train a linear subspace where one end is optimized for efficiency, while the other end prioritizes accuracy. The "lcs_p" training method refers to the "LCS+P" in the paper and trains a degenerate subspace conditioned to perform at arbitrary sparsity rates in the unstructured and structured sparsity settings, or bit widths in the quantized setting.

## Structured Sparsity

In the structured sparsity setting, we support five training methods:

1. "lcs_l" -- This refers to the LCS+L method where one end of the linear subspace performs at high sparsity rates while the other performs at zero sparsity.
2. "lcs_p" -- This refers to the LCS+P method where we train a degenerate subspace conditioned to perform at arbitrary sparsity rates.
3. "lec" -- This refers to the method introduced in "Learning Efficient Convolutional Networks through Network Slimming" by Liu et al. (2017). We do not perform fine-tuning, as described in our paper.
4. "ns" -- This refers to the method introduced in "Slimmable Neural Networks" by Yu et al. (2018). We use a single BatchNorm to allow for evaluation at arbitrary width factors, as decribed in our paper.
5. "us" -- This refers to the method introduced in "Universally Slimmable Networks and Improved Training Techniques" by Yu & Huang (2019). We do not recalibrate BatchNorms (to facilitate on-device compression to arbitrary widths), as described in our paper.

Training a model in the structured sparsity setting can be accomplished by running the following command:

> python train_structured.py

By default, the command above will train the cPreResNet20 architecture on CIFAR-10 using instance normalization layers with the LCS+L method. To specify the model, dataset, normalization, and training method, the flags `--model`, `--dataset`, `--norm`, `--method` can be used. The following command

> python train_structured.py --model resnet18 --dataset imagenet --norm IN --method lcs_p --imagenet_dir <dir>

will train a ResNet18 point subspace (LCS+P) on ImageNet using instance normalization layers and the parameters from our paper.

In addition to the global flags above, the structured setting also has the following:

- `--width_factors_list`: When training using the "ns" method, this sets the width factors at which the model will be trained.
- `--width_factor_limits`: When training using the "us", "lcs_l", or "lcs_p" methods, sets the lower and upper width factor limits.
- `--width_factor_samples`: When training using the "us", "lcs_l", or "lcs_p" methods, sets the number of samples to use for the sandwich rule. Two of these will be the samples from the width factor limits.
- `--eval_width_factors`: Sets the width factors to evaluate the model for all training methods.

The command

> python train_structured.py --model cpreresnet20 --dataset cifar10 --norm BN --method ns --width_factors_list 0.25,0.5,0.75,1.0

will train a cPreResNet20 architecture on CIFAR-10 via the NS method.

## Unstructured Sparsity

In the unstructured sparsity setting, we support three training methods:

1. "lcs_l" -- This refers to the LCS+L method where one end of the linear subspace performs at high sparsity rates while the other performs at zero sparsity.
2. "lcs_p" -- This refers to the LCS+P method where we train a degenerate subspace conditioned to perform at arbitrary sparsity rates.
3. "target_topk" -- this will train a network optimized to perform well at a specified TopK target.

Training a model in the unstructured sparsity setting can be accomplished by running the following command:

> python train_unstructured.py

By default, the command above will train the cPreResNet20 architecture on CIFAR-10 using group normalization layers with the LCS+L method and the parameters used described in our paper. To specify the model, dataset, normalization, and training method, the flags `--model`, `--dataset`, `--norm`, `--method` can be used. The following command

> python train_unstructured.py --model resnet18 --dataset imagenet --norm GN --method lcs_p --imagenet_dir <dir>

will train a ResNet18 point subspace (LCS+P) on ImageNet using group normalization layers again using the parameters from our paper.

The command

> python train_unstructured.py --model resnet18 --dataset imagenet --method target_topk --topk 0.5 --imagenet_dir <dir>

will train a VGG19 architecture optimized to perform at a TopK value of 0.5.

In addition to the global flags above, the unstructured setting also has the following:

- `--topk`: When training using the "target_topk" method, this sets the target TopK value.
- `--eval_topk_grid`: Will evaluate the model at these TopK values.
- '--topk_lower_bound': The lower bound TopK value (1-sparsity) to be used for training. For linear subspaces, one end of the line will be optimized for sparsity 1-topk_lower_bound which corresponds to the high accuracy endpoint. Note: If specified, eval_topk_grid must be specified as well.
- '--topk_upper_bound': The upper bound TopK value (1-sparsity) to be used for training. For linear subspaces, one end of the line will be optimized for sparsity 1-topk_upper_bound which corresponds to the high efficiency endpoint. Note: If specified, eval_topk_grid must be specified as well.

The following command

> python train_unstructured.py --model cpreresnet20 --dataset cifar10 --norm GN --method lcs_p --topk_lower_bound 0.005 --topk_upper_bound 0.05 --eval_topk_grid 0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05

will train a point subspace with high sparsity.

## Quantization

In the quantized setting, we support three training methods:

1. "lcs_l" -- This refers to the LCS+L method where one end of the linear subspace performs at a low bit width while the other performs at a hight  bit width.
2. "lcs_p" -- This refers to the LCS+P method where we train a degenerate subspace conditioned to perform at arbitrary bit widths in a range.
3. "target_bit_width" -- This trains a network optimized to perform at a specified bit width.

Training a model in the structured sparsity setting can be accomplished by running the following command:

> python train_quantized.py

By default, the command above will train the cPreResNet20 architecture on CIFAR-10 using group normalization layers with the LCS+L method with a bit range [3,8]. To specify the model, dataset, normalization, and training method, the flags `--model`, `--dataset`, `--norm`, `--method` can be used. The following command

> python train_quantized.py --model vgg19 --dataset imagenet --norm GN --method lcs_p --imagenet_dir <dir>

will train a ResNet18 point subspace (LCS+P) on ImageNet using group normalization layers.

In addition to the global flags above, the quantized setting also has the following:

- `--bit_width`: When training using the "target_bit_width" method, this sets the target bit width.
- `--eval_bit_widths`: Will evaluate models at these bit widths.
- `--bit_width_limits`: This sets the upper and lower bit width bounds to use for training.

The following command

> python train_quantized.py --model cpreresnet20 --dataset cifar10 --norm GN --method lcs_l --bit_width_limits 3,8 --eval_bit_widths 3,4,5,6,7,8

will train a linear subspace cPreResNet20 model with GN layers on the ImageNet dataset and will be optimized so that one end of the line performs at 3 bits, and the other at 8.
