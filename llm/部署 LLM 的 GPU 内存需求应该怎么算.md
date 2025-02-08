$$M = \frac{(P \times 4B)}{(32/Q)} \times 1.2$$

## Symbols and Descriptions

| Symbol | Description |
|--------|-------------|
| M      | GPU memory expressed in Gigabyte |
| P      | The amount of parameters in the model. E.g. a 7B model has 7 billion parameters. |
| 4B     | 4 bytes, expressing the bytes used for each parameter |
| 32     | There are 32 bits in 4 bytes |
| Q      | The amount of bits that should be used for loading the model. E.g. 16 bits, 8 bits or 4 bits. |
| 1.2    | Represents a 20% overhead of loading additional things in GPU memory |

GPU memory required for serving Llama 70B

Let's try it out for Llama 70B that we will load in 16 bit. The model has 70 billion parameters.

$$\frac{70 \times 4\text{ bytes}}{32/16} \times 1.2 = 168\text{GB}$$

That's quite a lot of memory. A single A100 80GB wouldn't be enough, although 2x A100 80GB should be enough to serve the Llama 2 70B model in 16 bit mode.
