# autoresearch

This is an experiment to have the LLM do its own research for BERT-based Kaggle competition classification.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `train.py` — the file you modify. Model architecture, optimizer, hyperparameters, training loop.
   - `prepare.py` — fixed constants, data loading, evaluation, tokenizer. Do not modify.
4. **Verify data exists**: Check that `data/processed/` contains train/val/test splits. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single device (CPU, CUDA, or MPS). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, classifier head, pooling strategy, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains data loading, tokenizer, and evaluation logic.
- Install new packages or add dependencies. You can only use what's already available.
- Modify the evaluation harness. The evaluation metrics in `prepare.py` are the ground truth.

**The goal is simple: get the highest val_f1_macro.** Since the classes may be imbalanced, F1 macro is the primary metric. Everything is fair game: change the classifier head, the optimizer, the hyperparameters, the batch size, the pooling strategy. The only constraint is that the code runs without crashing.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_f1_macro improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_f1_macro improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Hyperparameters (train.py)

All hyperparameters are at the top of `train.py` and can be modified:

### Model Architecture
```python
DROPOUT = 0.1                    # Dropout probability
HIDDEN_DIM = 256                 # Classifier hidden dimension
USE_HIDDEN_LAYER = True          # Use hidden layer in classifier
POOL_STRATEGY = "cls"            # Pooling: "cls", "mean", "max"
```

### Optimization
```python
BATCH_SIZE = 16                  # Training batch size
LEARNING_RATE = 2e-5             # Peak learning rate
WEIGHT_DECAY = 0.01              # L2 regularization
ADAM_BETAS = (0.9, 0.999)        # Adam momentum
WARMUP_RATIO = 0.1               # LR warmup fraction
```

### Training Control
```python
MAX_EPOCHS = 10                  # Maximum epochs
EARLY_STOPPING_PATIENCE = 3      # Patience for early stopping
GRADIENT_CLIP = 1.0              # Gradient clipping (0 = off)
USE_CLASS_WEIGHTS = True         # Balance classes
```

### Regularization
```python
LABEL_SMOOTHING = 0.0            # Label smoothing
FREEZE_EMBEDDINGS = False        # Freeze BERT embeddings
FREEZE_ENCODER_LAYERS = 0        # Layers to freeze
```

## Output format

Once the script finishes it prints a summary like this:

```
==================================================
BERT Training Script
==================================================
Device: cuda
...
Epoch  1/10 | Train Loss: 1.2345 | Val Loss: 0.9876 | Val F1: 0.7234 | Time: 45.2s
  -> New best model saved! F1: 0.7234
...

>>> val_f1_macro = 0.8456
```

You can extract the key metric from the log file:

```
grep "val_f1_macro" run.log | tail -1
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_f1_macro	epochs	status	description
```

1. git commit hash (short, 7 chars)
2. val_f1_macro achieved (e.g. 0.8456) — use 0.0000 for crashes
3. epochs completed (e.g. 5) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_f1_macro	epochs	status	description
a1b2c3d	0.8234	10	keep	baseline
b2c3d4e	0.8456	8	keep	increase LR to 3e-5
c3d4e5f	0.8012	10	discard	switch to max pooling
d4e5f6g	0.0000	0	crash	double hidden dim (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr6`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "val_f1_macro\|epochs" run.log | tail -2`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_f1_macro improved (higher), you "advance" the branch, keeping the git commit
9. If val_f1_macro is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (including startup and eval overhead). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

## Tips for Optimization

1. **Learning rate** is often the most impactful hyperparameter
2. **Class weights** help with imbalanced data
3. **Pooling strategy** can significantly affect results:
   - `cls` - fast, often sufficient
   - `mean` - captures more context
   - `max` - emphasizes strong signals
4. **Hidden layer** adds capacity but may overfit on small data
5. **Freezing layers** can help with limited data
6. **Label smoothing** can improve generalization

## Constraints

- **Device**: CPU, CUDA, or MPS (Apple Silicon)
- **Base model**: DistilBERT (fixed in prepare.py)
- **Labels**: 4 classes — CLASSIC ML, LLM/NLP, CV, Other
- **Primary metric**: val_f1_macro (higher is better)
