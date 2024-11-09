# Quantization vs Pruning vs Distillation: Optimizing NNs for Inference

[Quantization vs Pruning vs Distillation: Optimizing NNs for Inference (youtube.com)](https://www.youtube.com/watch?v=UcwDgsMgTu4)

[NLP Deep Dives - YouTube](https://www.youtube.com/playlist?list=PLc7il9kHHib2FOazrkCUBqGdQviq-thQu)

Bai Li <https://luckytoilet.wordpress.com/>

----

Hi everyone in today's video I'm going to talk about how to compress and optimize your model.

Let's say that you have trained or fine-tuned the model and now you're ready to deploy it for everybody to use.

But you find that the latency is too slow and you want to make your model faster.

Well today I will show you four ways of making your model faster:

- Quantization
- Pruning
- Knowledge distillation
- Engineering optimizations.

My name is Bai. I am machine learning engineer and a PhD in natural language processing.

## Quantization

Without further ado, let's get started with Quantization.

Neural networks are large and take up a lot of space because they have millions or billions of parameters.

By default when you train your neural network, usually the parameters are stored in FP32,  which means that each parameter takes up 32 bits.

The idea of quantization is to reduce the precision of the parameters into a format that takes up less space.

For example 16-bit floating point or INT8.

If we store all of the parameters in INT8 format, that means all of them are represented by
integers between 0 and 255, then that means we will save four times as much space compared to the original Nntwork in FP32 format.

### Zero-point Quantization

![Zero-point_Quantization](Zero-point_Quantization.png)

There are several ways you can turn floating points into integers, and the most common way is called zero-point quantization.

I'm going to step through an example of how this works.

The reason this is called zero-point quantization is all the zeros in the original matrix are mapped to zero in the quantized version.

We will see later why this is useful for sparse neural networks.

Next we take the maximum absolute value element and map it either to negative 128 or 127.

In this case, the maximum absolute value element is negative 51.5, so this gets mapped to negative 128.

The quantization has to be a linear transformation, so with two elements determined the rest of the elements are determined as well.

Finally, to get the INT8 representation, we add 128 to each element so all of the elements are positive.

### Weight vs Activation Quantization

Two ways of doing quantization are quantizing the weights and quantizing the activations.

![Weight_vs_Activation_Quantization](Weight_vs_Activation_Quantization.png)

In weight quantization, we store all of the weights of the neural network INT8 format, and we dequantize the weights into FP32 when we run it, so that all of the data remains in FP32 format throughout the network.

Since everything is being done in FP32, it is not going to be faster than the original model, but this is still useful because it saves space.

For example, in mobile devices, where making the model four times smaller is a significant mprovement.

On the other hand, in activation quantization, we convert all of the inputs into INT8, and all of the computations are also performed in INT8.

This is faster than weight quantization because on most hardware INT8 computations are faster than FP32.

But one challenge is we don't know the inputs of of the neural network when we quantize the model.

So in order to determine the scale factors for each layer, we will need a calibration set that represents what kind of data we expect to see during inference time.

If you ever come across the terms static or dynamic quantization these refer to different ways of determining the scale factors of activations.

If calibration is not done properly, you will encounter clipping in the network, because the quantization is only able to handle floating points in a certain range, and anything outside of the range will clip to the max or minvalues.

![A10](A10.png)

To determine which type of quantization to use, it helps to look at the specifications of the hardware that you intend to do the inference on.

Here I have pulled up the data sheet for the Nvidia A10 GPU, which is a popular choice for inference.

According to the specification sheet, the FP32 performance of this GPU is 31 TerraFlops.

Whereas the INT8 performance is a lot faster at 250 tensor operations per second.

This is thanks to its tensor core capabilities. But not all GPUs have this capability.

So on some older GPUs you might find that the FP32 has the same performance as INT8.

### LLM.INT8: Mixed decompostition (PTQ)

One more thing that you should be aware of is the effect of outliers on quantization.

One recent paper called [LLM.INT8()](https://arxiv.org/abs/2208.07339) found that in large language models with over 6 billion parameters, quantization doesn't work because of outlier features, that caused the performance of the model to fall to close to zero.

![quantization_doesnt_work](quantization_doesnt_work.png)

To understand why this is the case, consider what will happen if you have an outlier in the weights.

What happens is the buckets become very large because there is only 256 buckets to cover all of the values between the minimum and maximum values, including the outlier.

To solve this problem, they proposed a mixed decomposition scheme where the outliers are handled separately from the majority of the data.

This is not necessary when you're running smaller models, but useful to know if you ever plan to quantize larger language models.

## Pruning

Now let's move on to the second method, pruning.

The basic idea of pruning is you want to remove some of the connections in your neural network.

![pruning](pruning.png)

This leaves you with what is called a sparse network.

And in terms of the matrix computation, a lot of the values in The matrix get set to zero, which makes it cheaper to store and faster to compute.

### Magnitude pruning(Unstructured pruning)

Once again, there are many different algorithms you can use to do your pruning.

And in this video, I will only talk about the simplest one magnitude pruning(幅度剪枝).

![magnitude_pruning](magnitude_pruning.png)

In magnitude pruning, you first pick a pruning factor X, which denotes what proportion of the connections you would like to remove.

Then in each layer of the network, you set the lowest X percent of the weights by absolute value to zero.

The idea being that the lowest weights by absolute value, so the ones closest to zero, are the least important for the the network to function.

By removing some of the connections, your model will experience some degradation in accuracy.

So as an optional third step, you may want to retrain your model for a few more iterations while keeping the removed weights fixed at zero, and this is to recover some of the accuracy.

Now it's important to note that just setting some of the matrix values to zero doesn't actually save space or make it go any faster.

Because zeros take just as much space to store and just as much time to process as non-zero values.

So if you're doing pruning, you need to combine that with some sort of sparse execution
engine that can take advantage of a sparsified neural network structure.

Let me give you an example of what I mean by this.

In general, when your GPU performs matrix multiplication, it iterates over slices of your two
matrices.

And for each pair of slices, it accumulates an out-of-product matrix.

And the sum of all of this is the matrix multiplication.

But even if you have zeros in the slices, it does not affect how long this operation will take.

![sparse_matrix_mul](sparse_matrix_mul.png)

Compare this, on the other hand, with an algorithm that's specifically designed to multiply sparse matrices.

The sparse matrix multiplication algorithm has a special trick that skips over all of the zero entries in a vector, so that the more zeros you have in a matrix, the faster the multiplication will be.

### N:M Sparsity(structured pruning)

The last thing I will talk about is structured pruning.

If you simply remove connections from a network without any further pattern that is called unstructured pruning.

But structured pruning is when you enforce more structure on which weights are you allowed to set to zero.

![ 2-4_structured_sparcity](2-4_structured_sparcity.png)

One type of structured pruning is the 2-4 structured sparcity pattern.

What this means is for each block of four consecutive matrix values, only two of them are allowed to be non-zero.

And this allows you to store the matrix in a compressed format where only the non-zero values are stored along with indices for which values are represented in which positions.

As well, Nvidia's Tensor Core GPUs are able to execute this type of structured sparsity with greater efficiency.

So we see that for pruning neural networks, we need to design the pruning algorithm with the hardware in mind.

Which pruning algorithm you should use will depend on which type of sparsity runs fast on the hardware that you intend to deploy your neural network on.

## Distillation

The third method of making our model more efficient is knowledge distillation, or sometimes called model distillation.

![knowledge_distillation](knowledge_distillation.png)

So what is knowledge distillation? In knowledge distillation, we first use the data to train a teacher network.

After the teacher network has been trained, we then start training the student network to predict the outputs of teacher network.

Well, you might ask, why is it more helpful to have the student predict the outputs of the teacher network instead of just training the student network from the labels?

And the reason is basically, the output of the teacher network contains more information, so it is faster and easier for the student network to learn from it.

Assuming you're doing some kind of classification model, then the training data only has one label per training instance.

But the output of the teacher network gives you a probability distribution over all possible labels, which is a lot more information to learn from.

![distillation_advantagesNdisadvantages](distillation_advantagesNdisadvantages.png)

Knowledge distillation has several advantages and disadvantages compared to other methods of optimizing your model.

One advantage of knowledge distillation is you can modify the architecture of the student model to be different from the teacher model.

For example, if your teacher model has 12 transformer layers, that doesn't mean your student model has to have 12 transformer layers.

It might have six or two or something like that, and this sort of architectural change is
not really possible with quantization or pruning.

Therefore knowledge distillation has the biggest potential gain in speed compared to all of the other methods that we've seen.

But the disadvantage is, it's relatively more difficult to set it up, because you need to set up the training data, which might be billions of tokens.

And if the teacher model is a big model, then running inference over it can be a challenge.

So overall, knowledge distillation is relatively expensive.

In my previous experience, this takes maybe 5-10% of the total compute or GPU hours needed  to train the teacher model from scratch.

Here is one example. **DistilBERT** is a model trained with knowledge distillation, where BERT is a teacher model.

In this model, they reduced the size of the BERT base model by 40% while retain 97% of its accuracy.

And the authors tell us how many GPUs and for how long they had to train this model.

DistilBERT was trained on 8 GPUs for about 90 hours, so in total about 700 hours of
GPU time.

In comparison, the RoBERTa model, which is similar to The BERT model, required one day of training on 1000 GPUs, which is about 24,000 hours of GPU time or around 20 times bigger.

So we see that in the DistilBERT example, training a model using knowledge distillation is a lot faster than training from scratch, but still requires a significant amount of compute.

## Engineering Optimizations

The last category of optimizations are what I will group together and call them
all engineering optimizations.

At some point, you need to decide whether you want to run your model on CPUs or GPUs.

In either case, making it run efficiently requires doing some integration between the hardware and the software.

What I mean by that is your hardware might have the physical capability of running a model quickly, but at the same time, the software needs to know how to use the hardware capabilities.

For example, vectorized operations to multiply large matrices in a parallel manner.

GPUs are of course very good at this, and but CPUs actually can do vectorized operations as well using some of the newer instruction sets like AVX2 and AVX512(five-twelve).

Newer CPUs and GPU models have the ability to perform reduced precision and mixed precision operations faster than full precision, like INT8 format, and this is useful for running inference on quantized models quickly.

As well, some GPUs have the hardware capability to run **sparse kernels**, which is necessary to have a gain in speed when running pruned neural networks.

Another type of optimization is **fused kernels**.

![pytorch_scaled-dot-product](pytorch_scaled-dot-product.png)

For example, PyTorch has a function called scaled dot product attention, and what this function does is it combines all of these operations that are typically seen together in a transformer architecture, but it does it very quickly.

In a transformer architecture, we often do a sequence of operations in a fixed order.

For example, multiply the query and key matrices, and then take a softmax, take a square root, and then apply dropout.

And if we combine all of these operations into one single operation that is executed in on the GPU, then this is a lot faster than if we executed each instruction in sequence.

One popular way of implementing this is called Flash Attention.

![FlashAttention](FlashAttention.png)

Not only does it fuse together these operations, but flash attention also does some tiling, and some optimization according to the gpu's memory hierarchy to further reduce the amount of time needed to perform this operation.

And you can see on the chart on the right here that the fused flash attention is a lot faster than naively doing all of the operations sequentially in PyTorch.

All of this might sound a little bit overwhelming, but really it's not that complicated in practice.

Because all you have to do in practice is convert your model that you have trained into some format, that is executable by an  inference engine that is optimized for whatever hardware that you in intend to deploy on.

![trainingNinference](trainingNinference.png)

The reason why you often need to use separate frameworks for training and  inference, is because the requirements for training a neural network is often quite different from the requirements during inference.

When you're training a model, you need a library that can do things that are relevant during training, like loading the data from disk, pre-processing it, doing gradient descent and back propagation, running evaluations, saving checkpoints, and so on.

But none of that is really required during inference.

When you're deploying a model for inference, however, the requirements tends to be quite different.

The model needs to be small and fast and needs to run efficiently on a hardware that's probably different from what you trained the model on.

So it is often better to use a different library for inference.

Two of the most popular libraries for inference are ONNX Runtime, which can run models that are stored in the ONNX format on a variety of different hardware.

And another one is TensorFlowLite, if you prefer the TensorFlow ecosystem.

## Conclusion

Let's summarize what we have covered in this video so far.

![conclusion_table](conclusion_table.png)

First, Quantization.

Quantization uses less precise data formats to reduce the model size and latency.

When you're reducing the format from FP32 to INT8, this results in a reduction of 4x.

It is best used in combination with a reduced position execution engine that is able to execute reduced position formats faster.

And a drawback is it can potentially result in a loss of accuracy, although hopefully not too much.

Pruning is setting some of the weights of un networks to zero to save space and compute.

And in order for this to work at all, it requires an execution engine that's capable of executing sparse neural networks.

And similar to quantization, it can potentially result in a loss of accuracy.

Knowledge distillation is the only method we covered where you're able to modify the model's architecture.

So the impact of this is varied depending on how you modify the architecture, but can potentially be much larger than any other method.

The downside of know distillation is it's relatively expensive to train.

And finally, engineering optimizations.

These should be used in combination with all of the above methods.

And you should expect no loss in accuracy when employing engineering optimizations because the output should be identical.

![trade-off](trade-off.png)

Ultimately, all of these methods make a trade-off between development cost, inference cost, and model accuracy.

Quantization and to some extent, model pruning are two ways of reducing the model's latency and inference cost without being too difficult.

But for both of them, you will potentially incur a slight loss in model accuracy.

Knowledge distillation has potential to reduce your model size a lot further, but it is also more complicated and expensive to train, especially for larger models that are trained with lots of GPUs.

Thank you for watching and I hope you will use these techniques in your own projects.

If you found this video helpful please don't forget to like And subscribe to my channel and get notified when I make new and helpful machine learning related videos.

It will help me out a lot.

Goodbye.
