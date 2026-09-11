Yep :') The previous answer is **way more complicated than it needs to be**. Let's translate it into normal-human language.

## First: what are we even trying to do?

You have **Kepler's equation**:

$$
E-e\sin E-M=0
$$

You currently use **Newton-Raphson** to find \(E\).

Your teacher basically said:

> "Don't just use basic Newton. Look at newer Newton variants from recent literature and compare them."

So you're looking for **newer algorithms that solve equations like Newton, but hopefully faster/better.**

---

# 1. The headline finding

> **"No 2024+ paper applies a genuinely new Newton-type variant to Kepler's equation itself."**

Meaning:

I couldn't find a recent paper saying:

> "Here is our new Newton method, and we specifically use it to solve Kepler's equation."

Instead, recent Kepler papers mostly do things like:

* find a **better starting guess** for Newton
* develop completely different methods
* improve existing solvers

So we have to take **recent Newton variants from general numerical-analysis papers** and apply them to Kepler ourselves.

---

# 2. What is a "with-memory" method?

This is the important idea.

Normal Newton does:

```text
current guess
     ↓
calculate stuff
     ↓
next guess
```

A **with-memory** method says:

> "Why throw away information from the previous iteration?"

So it does:

```text
current iteration
      +
information from previous iteration
      ↓
better next guess
```

The old information is basically **reused** to make the method converge faster.

That's why it's called **with-memory**.

---

# 3. What's NWM11?

NWM = **Newton-type method with memory**.

The particular NWM11 method claims approximately:

$$
\boxed{\text{order }10.7446}
$$

Don't panic about this number.

### What does "order" mean?

Roughly:

> **How quickly does the error shrink as the iterations continue?**

Newton:

$$
\text{order}=2
$$

Higher-order methods can converge much faster when you're already close to the solution.

So:

```text
Newton       → order 2
Halley       → order 3
NWM11        → order ~10.7
```

Higher order sounds fantastic, **but** there's a catch.

---

# 4. What's the catch?

Higher-order methods often require more calculations.

For example, imagine:

```text
Newton:
1 expensive sin/cos evaluation
→ good progress

Method X:
3 expensive sin/cos evaluations
→ even better progress
```

Method X might have higher mathematical order but still be **slower in practice**.

That's why your team was talking about:

> **convergence order per transcendental evaluation**

Basically:

> "How much convergence do I get for each expensive `sin()`/`cos()` calculation?"

---

# 5. Why are sin/cos expensive here?

Your equation is:

$$
f(E)=E-e\sin E-M
$$

Newton needs:

$$
f'(E)=1-e\cos E
$$

So at the same \(E\):

```text
sin(E) → gives f
cos(E) → gives f'
```

That's actually convenient.

And you also have:

$$
f''(E)=e\sin E
$$

$$
f'''(E)=e\cos E
$$

So once you have `sin(E)` and `cos(E)`, you basically get **all these derivatives for free**.

That's a special property of Kepler's equation.

---

# 6. What does NWM11 do?

The method generates several points:

```text
sₖ → vₖ → tₖ → next s
```

At each stage it gets closer to the answer.

Instead of calculating expensive derivatives everywhere, it uses **previously calculated values** to estimate some derivatives.

So conceptually:

```text
sₖ
 ↓
calculate f and f'
 ↓
vₖ
 ↓
reuse information to estimate derivative
 ↓
tₖ
 ↓
reuse information again
 ↓
sₖ₊₁
```

The clever part is:

> **It gets higher-order convergence without calculating new derivatives everywhere.**

---

# 7. What does "Hermite interpolation" mean?

This sounds scary but isn't.

Suppose you know:

```text
f(x₁)
f(x₂)
f(x₃)
```

and maybe some derivatives.

You can construct a polynomial that approximately represents the function between those points.

That's **Hermite interpolation**.

In this method, that interpolation is used to estimate information like:

$$
f'(x)
$$

without actually calculating the derivative again.

So:

```text
Instead of:
calculate expensive derivative

Do:
use information we already have
→ estimate derivative
```

---

# 8. What does "zero extra evaluations" mean?

This is important.

Suppose the base algorithm already needs:

```text
sin(x1)
sin(x2)
sin(x3)
```

The memory modification doesn't require another:

```text
sin(x4)
```

It reuses old information.

So:

> **Higher convergence without additional function evaluations.**

That's why the paper is interesting for your project.

---

# 9. Why is NWM11 potentially good for Kepler?

Because Kepler has this special structure:

$$
f(E)=E-e\sin E-M
$$

$$
f'(E)=1-e\cos E
$$

At a given \(E\):

```text
sin(E) + cos(E)
       ↓
f(E), f'(E), f''(E), f'''(E)
```

So derivatives aren't particularly painful.

NWM11 also tries to avoid repeatedly calculating derivatives.

Therefore:

> **It could potentially be a good match for Kepler.**

BUT—and this is very important—

**the NWM11 paper did NOT test it on Kepler's equation.**

You would be doing that yourself.

---

# 10. What is the disadvantage?

NWM11 evaluates the function at **three different points**.

Something like:

```text
sₖ
 ↓
vₖ
 ↓
tₖ
```

Every new point means you need new trigonometric calculations.

So even though it has order ~10.7, it might not actually beat Newton for every value of \(e\).

That's something **your experiment needs to find out.**

---

# 11. What's NWM10?

Basically:

> Same idea as NWM11, but slightly simpler.

It has:

$$
\text{order}=10
$$

instead of:

$$
10.7446
$$

Both use the same general **memory idea**.

So:

```text
NWM10 → one memory parameter
NWM11 → two memory parameters
```

More memory information → higher order.

---

# 12. What's NWM9?

Same family again.

It has approximately:

$$
\text{order}=8.9
$$

So conceptually:

```text
NWM9
  ↓
~8.9 order

NWM10
  ↓
10 order

NWM11
  ↓
~10.7 order
```

They're basically progressively more sophisticated versions of the same idea.

---

# 13. What's the 2024 bi-parametric method?

Another **with-memory Newton method**.

It has approximately:

$$
\text{order}=10.52
$$

But it needs more expensive evaluations.

For Kepler, roughly:

```text
NWM11:
1 sin+cos pair
+
2 sin evaluations

Other method:
2 sin+cos pairs
+
1 sin evaluation
```

So NWM11 may be more attractive because **cos is needed less often**.

---

# 14. What's the Kumar et al. 2024 method?

Another recent **with-memory method**.

Its claimed order is:

$$
\boxed{11}
$$

That's very high.

The basic idea is again:

```text
Use current information
+
reuse previous iteration information
↓
get faster convergence
```

The problem is that the exact implementation details weren't easily available from the source you found.

So:

> **Don't use this one as your main experimental method unless you can get the actual algorithm.**

---

# 15. What is Napier 2024?

This one is **different**.

It isn't really a Newton variant.

Instead, it asks:

> "Can we give Newton a much better starting point?"

Normally:

```text
Newton:
E₀ → E₁ → E₂ → E₃ → solution
```

With a better initial guess:

```text
Better E₀ → E₁ → E₂ → solution
```

So you save iterations.

Napier uses **symbolic regression / machine learning-like techniques** to find good formulas for the starting guess.

---

# 16. What is symbolic regression?

Very simply:

Instead of giving a computer data and asking:

> "Predict a number."

you ask:

> "Find me a mathematical formula that predicts this number."

For example, it might discover something like:

$$
E_0 = M + 0.8e\sin(M)
$$

or some more complicated formula.

So it's basically:

```text
data
 ↓
search for mathematical formula
 ↓
good initial guess
```

---

# 17. Why is Napier useful?

Because you can combine it with **any solver**.

For example:

```text
Napier starting guess
        ↓
Newton
```

or:

```text
Napier starting guess
        ↓
NWM11
```

So it isn't really a competitor to Newton.

It's more like a **booster** that you can put before the solver.

---

# 18. The REALLY important part for your project

This paragraph from the answer:

> "the efficiency index everyone reports treats every evaluation as equally expensive."

means:

Most numerical-analysis papers say:

```text
evaluation = evaluation
```

They don't care whether you're calculating:

```python
x + 2
```

or

```python
math.sin(x)
```

They count both as "one evaluation."

But **for your project, that's not realistic**.

For Kepler:

```text
x + 2             → cheap
x*x               → cheap
sin(x)            → expensive
cos(x)            → expensive
```

So you should measure the expensive operations separately.

---

# 19. Your project could therefore measure

For each method:

### Accuracy

```text
How close is E to the true solution?
```

### Iterations

```text
How many iterations were needed?
```

### Function evaluations

```text
How many f(E) calculations?
```

### Sin evaluations

```text
How many sin(E) calls?
```

### Cos evaluations

```text
How many cos(E) calls?
```

### Runtime

```text
How long did the algorithm actually take?
```

---

# 20. Why does eccentricity matter?

Kepler has:

$$
E-e\sin E=M
$$

where \(e\) is eccentricity.

You can test different values:

```text
e = 0.1
e = 0.5
e = 0.8
e = 0.9
e = 0.99
e = 0.999
```

As \(e\) gets close to 1, solving the equation can become more difficult.

So you might discover:

```text
e < 0.8
→ Newton is already great

e ≈ 0.9
→ NWM11 starts helping

e ≈ 0.99
→ another method becomes better
```

**That's an actual experimental result.**

---

# 21. What does the final "novelty" suggestion mean?

The previous answer said:

> "None of these papers exploit \(f''=e\sin E\), \(f'''=e\cos E\)."

Meaning:

The recent NWM papers were designed for **generic equations**.

They don't specifically say:

> "Hey, Kepler gives us derivatives almost for free!"

You could potentially take a recent Newton-type method and **adapt it specifically for Kepler**.

For example:

```text
Generic NWM method
       ↓
adapt it to Kepler
       ↓
exploit exact sin/cos derivative relationships
       ↓
compare against Newton
```

That could become an interesting project angle.

---

# 22. So what should YOU actually take from all this?

Don't try to understand every equation in that previous answer yet.

Your current shortlist can simply be:

| Method             | What it does                                  |
| ------------------ | --------------------------------------------- |
| **Newton-Raphson** | Your baseline                                 |
| **NWM11 (2025)**   | Newton + memory → very high-order convergence |
| **NWM9 (2024)**    | Simpler Newton + memory                       |
| **Napier (2024)**  | Gives Newton a better starting guess          |

And your experiment becomes:

```text
For many values of e and M:

        ↓

Newton
NWM9
NWM11
Napier + Newton

        ↓

Compare

iterations
runtime
sin calls
cos calls
accuracy
```

### The one sentence I'd tell your teacher:

> **"Based on recent 2024–2025 literature, we selected Newton-type methods with memory as modern high-order alternatives to classical Newton, and we will evaluate whether their higher convergence order actually translates into lower computational cost for Kepler's equation, whose dominant cost comes from repeated trigonometric evaluations."**

That's the core of the whole thing. Everything else is details.
