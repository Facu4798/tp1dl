---
name: code-style
description: Review and fix AI-bloat style issues (e.g. excessive comments) in Python code Claude just wrote. Use once after finishing a coding task, as a final pass over the new/changed Python lines — not after every edit, and not for non-Python code.
---

# Code Style (Python)

Reviews Python code Claude just wrote for common "AI bloat" patterns — habits models default to that read as over-engineered or defensive in a context (university projects, static environments) where they aren't warranted — and fixes them in place.

## When to use this skill

- After finishing a coding task — once all edits for the task are in place, before reporting completion to the user. Not after every individual edit; run it as a final pass over everything written during the task.
- Applies regardless of task size: a one-line fix gets the same pass as a multi-file feature.
- Review only the new/changed lines Claude itself wrote — do not flag pre-existing style issues in surrounding code that was merely touched or left alone.
- Do NOT trigger for non-code tasks (docs, config-only changes with no logic, pure research/explanation turns).
- Scoped to Python code — other languages are out of scope for now.

## Checks

For each new/changed line of Python, check for and fix:

1. **Excessive comments.** Remove comments that restate what the code already says. Keep a comment only if it explains a non-obvious *why* (a workaround, a subtle constraint) — not what a well-named line already communicates.
2. **Overlong or unnecessary docstrings.** Simple functions (short, single-purpose, self-evident from name/signature) get no docstring at all. Complex functions get at most two or three lines — no multi-paragraph docstrings, no full Args/Returns/Raises blocks.
3. **Defensive `try`/`except`.** Remove try/except blocks added defensively "just in case." These are university/static-environment projects — inputs and environment are fixed, so an unhandled failure is a bug in the code to fix, not a runtime condition to guard against. Only keep a try/except if the assignment/task explicitly calls for error handling (e.g. validating user input, a stated edge case).
4. **Excessive prints.** Remove decorative print scaffolding — separator banners (`print("="*60)`), boxed/framed output, print statements that just narrate what's about to happen. A print is warranted only in two cases: reporting an important result (e.g. a model's metrics), or a lightweight progress marker during a heavy/long-running loop (e.g. `print(f"epoch {i}")`). Everything else gets removed.
5. **Unnecessary intermediate variables.** If a variable is just a simple, fast one-line derivation (e.g. `x_2 = x_1.some_method()`), inline it at each call site instead (`x_1.some_method() + 1`) rather than naming it — even if that expression appears more than once. Repetition alone does not justify a variable. Keep a named variable only when it holds a config value, a long/complex expression, or the result of a heavy (slow/expensive) operation worth naming and not recomputing (e.g. `pandas_df = spark_df.toPandas()`).
6. **Unnecessary `if __name__ == "__main__":` / `main()` wrapper.** For a short single-purpose script that just runs top-to-bottom once, don't wrap it in a `main()` function plus a `if __name__ == "__main__":` guard. Only use this pattern when the file is genuinely meant to be both imported as a module and run directly.
7. **Redundant validation (`assert`, `isinstance`, etc.).** Don't add assert statements or `isinstance()` type checks to guard against conditions the static environment already guarantees. Same reasoning as the try/except rule — inputs and types are fixed by the assignment, so an unhandled failure is a bug to fix, not something to defensively check for.

## Examples

**Example 1 — comments, docstring, try/except, intermediate variable**

Before:
```python
def compute_average(values):
    """
    Computes the average of a list of numbers.

    Args:
        values: A list of numbers to average.

    Returns:
        The average as a float.
    """
    try:
        # sum up all the values
        total = sum(values)
        # divide by the number of values to get the average
        avg = total / len(values)
        return avg
    except ZeroDivisionError:
        print("Error: empty list")
        return None
```

After:
```python
def compute_average(values):
    return sum(values) / len(values)
```

**Example 2 — prints and isinstance validation**

Before:
```python
def load_dataset(path):
    assert isinstance(path, str), "path must be a string"
    print("="*60)
    print("Loading dataset from:", path)
    print("="*60)
    df = pd.read_csv(path)
    print("Dataset loaded successfully")
    return df
```

After:
```python
def load_dataset(path):
    return pd.read_csv(path)
```

**Example 3 — main wrapper on a single-shot script**

Before:
```python
def main():
    df = pd.read_csv("data.csv")
    model = train_model(df)
    print(model.score(df))

if __name__ == "__main__":
    main()
```

After:
```python
df = pd.read_csv("data.csv")
model = train_model(df)
print(model.score(df))
```

**Example 4 — simple expression used more than once**

Repetition alone doesn't justify naming it — `row["age"]` is simple and fast, so it stays inlined even though it's used twice:

Before:
```python
age = row["age"]
if age > 18:
    process_adult(age)
else:
    process_minor(age)
```

After:
```python
if row["age"] > 18:
    process_adult(row["age"])
else:
    process_minor(row["age"])
```

**Example 5 — heavy operation and progress prints kept**

This case should stay mostly as-is — the named variable avoids recomputing an expensive operation, and the loop print is a legitimate progress marker:
```python
pandas_df = spark_df.toPandas()  # heavy operation, worth naming

for epoch in range(num_epochs):
    train_one_epoch(model, pandas_df)
    print(f"epoch {epoch} done")

print(f"final accuracy: {accuracy:.4f}")  # important result, kept
```