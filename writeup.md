Optimizers Qualitatively Alter Solutions And We Should Leverage This (Pasanu et al. 2025)

The above work examines how different families of optimization algorithms affect the final solution that training converges to.
In particular, they examine SGD based methods and second-order methods (like Shampoo and Muon). 
They found that models optimized via second-order methods converged to lower-rank solutions, i.e. the SVD of weight matrices trained with second-order methods had lower rank.

This led to an interesting observation: models trained with second-order methods seemed to forget less and generalize better.
Their set up involved training 1 and 2 layer MLPs on MNIST images in the following paradigm: train on images showing (0, 1), then (2, 3), and on until a model has seen all classes in separate epochs.
There was _no mixing of classes between epochs_.
Models trained with SGD and AdamW would "forget" -- they'd be unable to classify any digits outside of the most recent epoch.
Models trained with second-order methods, however, did not experience as dramatic a drop in performance.

The two above observations leads to the following: optimization methods qualitatively affect learned solutions, and more exploration should be done to see if we can design optimizers that bias towards solutions with qualities we want (like broad generalization).

This is to contrast with existing work exploring and understanding optimizers as to how they affect convergence rate.
e.g. Wen et al. (2025) do broad sweeps across many kinds of optimizers to show how optimizers affect convergence rates at different model sizes and batch sizes.
They found that second-order optimizers are modestly more efficient and can converge ~10% faster than AdamW.

---

The exploration in this repo is to confirm Pasanu et al.'s observations at larger scales and for different modalities.
We want to confirm how this trend holds up when training models beyond 1 or 2 layer MLPs and across OOMs of parameter counts.
We want to further the understanding of how different optimizers affect solution quality for different domains.

Early POC of this line of inquiry seems exciting.
Small Vision Transformers (ViTs) trained on CIFAR-10 demonstrate similar trends to the 1 and 2 layer MLPs trained on MNIST.
ViTs trained with second-order optimizers have, on average, lower-rank weight matrices than their counterparts trained on SGD and AdamW.
We also examine intermediate activations of these models (this may be more meaningful to look at due to attention having unique inter-layer dynamics and a residual stream to pass information between them).
We find that ViTs trained with second-order optimizers also exhibit increased sparsity compared to models trained with SGD and AdamW.

We also confirm that lower-rank solutions seem to generalize better.
Our Muon-ViT achieved a test accuracy of ~37%, compared to ~29% accuracy of AdamW-ViT.
This result needs to be taken with a grain of salt; we have not tuned hyperparameters for every optimizer and analyzed training dynamics across multiple seeds.

TOOD:
- tune hparams for all optimizers
- train across different seeds; confirm trend is robust and not spurious
- train for longer; checkpoint model between epochs and examine how rank and activation distribution changes across epochs for every optimizer
- add covariance plot of intermediate activations
  - start with computing vectors prior to the final layer in the ViT
  - maybe extend to all layers to observe differences between layers
- examine rank and activation sparsity across model sizes trained
- examine rank and activation sparsity on different difficulty of tasks (e.g. MNIST vs CIFAR-10 vs CIFAR-100)
- better understand theoretical foundations of first-order and second-order optimizations

## covariance notes:

```
In the covariance/cosine similarity heatmap (Figure 4, right), they're computing similarity between class-specific representations. Here's the process:

Take the hidden layer activations (the intermediate representations in the MLP)
For each class, collect all activations for samples from that class
Compute a representative vector per class (likely the mean activation across all samples of that class)
Calculate pairwise cosine similarity between these class representatives
```
```
```


For a ViT, I'd recommend:

Primary choice: CLS token or mean-pooled patch tokens from the last layer before the classification head
Alternative: All patch token activations from the final transformer block

```
```
# For ViT - Option 1: CLS token
cls_tokens = model.get_cls_tokens()  # [batch, hidden_dim]
cov = torch.cov(cls_tokens.T)

# For ViT - Option 2: All patch tokens
patch_tokens = model.get_patch_tokens()  # [batch, num_patches, hidden_dim]
acts_flat = patch_tokens.reshape(-1, hidden_dim)
cov = torch.cov(acts_flat.T)
```


