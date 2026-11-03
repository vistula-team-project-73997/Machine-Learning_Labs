# Lab 5 study guide

Read this alongside the manual, then practise explaining the worked example without looking at the answer.

## Prescribed reading map

Your supplied July 31, 2025 textbook is `E:\introduction-to-machine-learning-systems_compress.pdf`. Use the verified page ranges below. The PDF viewer page is the printed page plus 52. The manual calls section 7.4 "Framework Architecture"; your edition labels it "Framework Components". These are the same assigned section number. The [online AI Frameworks chapter](https://mlsysbook.ai/contents/core/frameworks/frameworks.html) has revised headings.

| Manual section | Reading focus |
|---|---|
| 7.3 Fundamental Concepts | Graphs express dependencies; differentiation propagates local derivatives; tensors and execution modes determine how calculations run. |
| 7.4 Framework Architecture | Tensor operations, layers and training APIs expose different abstraction levels over the same computational machinery. |
| 7.6 System Integration | Hardware, libraries, serving and monitoring must fit together; deployment performance depends on the surrounding system. |
| 7.7 Major Framework Platform Analysis | Compare PyTorch's experimentation emphasis, TensorFlow's deployment ecosystem and JAX's composable transformations. |
| 7.8 Deployment Environment-Specific Frameworks | Cloud, edge, mobile and microcontrollers impose different memory, latency, energy and interoperability requirements. |

| Supplied textbook section | Printed pages | PDF viewer pages |
|---|---:|---:|
| 7.3 Fundamental Concepts | 310-350 | 362-402 |
| 7.4 Framework Components (manual: Architecture) | 350-353 | 402-405 |
| 7.6 System Integration | 356-359 | 408-411 |
| 7.7 Major Frameworks | 359-363 | 411-415 |
| 7.8 Framework Specialization | 363-369 | 415-421 |

Boundary pages can contain the end of one section and the start of the next.

These are reading prompts, not a substitute for the assigned sections. The experiments below connect those ideas to the lab's small examples.

## A chain-rule example to work by hand

Let a=2, b=-3, c=1. Define u=a*b=-6, v=u+c=-5, q=v**2=25 and L=ReLU(q)=25. A scalar's sum is itself. With vector inputs, construct one scalar expression per coordinate, then add the outputs.

Start with dL/dL=1. Because q>0, dL/dq=1. The square contributes 2v=-10, so dL/dv=-10. Addition sends -10 to both u and c. Multiplication sends b*(-10)=30 to a and a*(-10)=-20 to b. Therefore the gradient is **(30, -20, -10)**.

For a reused input, let u=x*x and L=u+u. At x=3, u=9 and L=18. The two uses of u contribute 1+1=2. The two operand positions in x*x each contribute 3*2=6. The final x gradient is 12. Overwriting either contribution produces the wrong answer.

The implementation stores parents as a tuple, preserving repeated operands. Topological traversal visits each node once; each operator still adds one contribution per operand position. Parents precede their users in forward order, so reversing this list ensures all users contribute before a parent propagates its own gradient. Iterative traversal also avoids Python's recursion-depth limit.

## Understand the engine's contract

Each Value stores a Python float, an accumulated gradient, parents and a local backward closure. Forward arithmetic immediately computes data and creates graph nodes. The closure saves the local derivative needed later. Addition contributes the upstream gradient; multiplication scales it by the other operand; a constant power p contributes p*x**(p-1); ReLU passes it for positive inputs and uses zero at x=0.

In this engine, `backward()` clears all reachable gradients before computing a new derivative. `backward(clear=False)` instead preserves leaf gradients across calls, while resetting intermediates. This is an explicit API choice. In training, each forward builds a fresh graph and each step starts with cleared parameter gradients. `retain_graph=False` releases the graph after the pass; rebuild it before another pass. Avoid changing parameter data between forward and backward.

The lab expression ReLU((a*b+c)**2) never supplies a negative input to ReLU. Consequently, that expression alone cannot establish that the negative ReLU branch is implemented correctly; the independent tests cover it.

## Numerical checks

The central difference for coordinate i is `(f(theta+h*e_i)-f(theta-h*e_i))/(2*h)`. Evaluate it with plain numeric arithmetic, independently of the engine. Compare both this estimate and PyTorch's derivative to the engine gradient. The required metric is maximum **absolute** difference, strictly below 1e-6.

A very large h introduces approximation error; a very small h causes cancellation. Float64 and moderate input scales make this check reliable. At ReLU's kink, central differences need not match the implementation's selected subgradient. Probe smooth points numerically and check the zero-input convention separately.

## PyTorch points to recognize

PyTorch creates autograd history during forward execution. `grad_fn` provides an entry point into the backward graph; leaf parameters receive `.grad` values. `optimizer.zero_grad()` prevents unintended accumulation between optimization steps. Evaluation mode changes layer behavior; `torch.inference_mode()` disables gradient recording for this lab's inference measurements. These are separate controls. See [Autograd mechanics, PyTorch 2.3](https://docs.pytorch.org/docs/2.3/notes/autograd.html).

The FX graph shows forward tensor operations. The autograd graph shows derivative work. The traced graph records example execution as TorchScript. Inspect all three saved text files and locate the two linear layers, ReLU and sigmoid in the forward representations.

## Execution and timing practice

Eager execution runs operations as Python reaches them. Compilation may capture and optimize regions, but has setup and runtime overhead. The lab measures both setup observations and steady-state execution; it does not count initial compilation inside warmed latency. Use the same model, values, dtype, device and inference settings for each variant. Alternate timing order and retain raw repetitions so one unusual sample does not decide the result.

Tracing records operations exercised by example inputs. A tensor-dependent Python branch may be fixed to the observed path, so an apparently fast trace can be incorrect for another input. This model has a fixed operation sequence. See [torch.jit.trace, PyTorch 2.3](https://docs.pytorch.org/docs/2.3/generated/torch.jit.trace.html). For compilation background, see the [official torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial).

Practise computing break-even: if setup costs 2 s and each optimized call saves 20 microseconds, more than 100,000 calls are needed to repay setup. If optimized execution is slower, there is no positive break-even under those measurements. Use the actual report before writing your short note.

## Tensor inspection exercise

Shape gives axis lengths. Dtype gives numeric representation and bytes per element. Strides describe storage steps per axis in units of elements. A view can reuse storage while changing indexing. With backing shape (1024,2048), `backing[:, ::2]` has shape (1024,1024), stride (2048,2); its contiguous copy has stride (1024,1).

The two arrays contain equal values, enabling a controlled timing comparison. The benchmark also times copying separately. A copy adds work and allocation, so it may only repay itself after reuse. Stride effects depend on the operation; do not generalize one measurement to all non-contiguous tensors. `.view()` can reject incompatible strides; `.reshape()` may allocate a copy.

## Before leaving for class

- Run `lab05/main.py` and read the measured report.
- Explain why x*x needs two contributions and why backward order matters.
- Explain a gradient check, a trace limitation and compilation break-even.
- Open the digital manual in `references/` or print it.
- Review the prescribed textbook sections yourself and arrive on time for the opening implementation instructions.
