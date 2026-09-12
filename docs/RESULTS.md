# Results & Reproducibility

The repository does not hard-code a fabricated MNIST accuracy number.

Run the actual benchmark after downloading MNIST:

```bash
python examples/mnist_mlp.py --epochs 10
```

For a quick smoke benchmark:

```bash
python examples/mnist_mlp.py --epochs 2 --train-limit 10000 --test-limit 2000
```

The script prints train/test loss and accuracy for every epoch and saves the
model checkpoint under `checkpoints/`.

This makes the README honest and reproducible instead of presenting an
unverified benchmark.
