# Ivann

[![Build Status](https://travis-ci.com/icivann/ivann.svg?branch=master)](https://travis-ci.com/icivann/ivann)

[Ivann](https://icivann.github.io/ivann/) is a **visual** tool for building neural networks with [PyTorch](pytorch.org), and generating code in Python in order to train and test them.

It is aimed at researchers who wish to see the bigger picture by visualising the flow behind their models. You can build through the UI by hand (Ivann aims to have the same feature set as PyTorch) and then modify the generated Python code, or build with a mixture of handwritten and generated code.

## What can I make with ivann?

Ivann is engineered to build neural networks. You will find
we put extra effort in building your typical CNN.

But we also put a lot of effort in basically supporting all
of PyTorch, and we allow for 'inlining' python code in an otherwise fully
UI-oriented developing experience. So it is up to you: you can either stick
to using PyTorch's layers, or make your own basic building blocks and use
them as bricks in your model. We have found that a match and mix of both works best.


## Building ivann locally
#### Setup dependencies
```bash
npm install
```

#### Produce production artifacts
```
npm run build
```

#### Test
```
npm run test:unit
```

