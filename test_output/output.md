# Attention Is All You Need

Ashish Vaswani
Google Brain
avaswani@google.com
&
Noam Shazeer
Google Brain
noam@google.com
&
Niki Parmar
Google Research
nikip@google.com
&
Jakob Uszkoreit
Google Research
uszko@google.com
&
Lion Jones
Google Research
lion@google.com
&
Aidan N. Gomez
University of Toronto
aidan@cs.toronto.edu
&
Lukasz Kaiser
Google Brain
lukaszkaiser@google.com
&
Ilia Polosukhin
illia.polosukhin@gmail.com

Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to
evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially
involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the
parameter-free position representation and became the other person involved in nearly every detail. Niki designed,
implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Lion also
experimented with novel model variants, was responsible for our initial codebase, and efficient inference and
visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor,
replacing our earlier codebase, greatly improving results and massively accelerating our research.

Work performed while at Google Brain.
Work performed while at Google Research.

[31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.]

###### Abstract

Recurrent neural networks, long short-term memory [?] and gated recurrent [?] neural networks in particular, have been
firmly established as state of the art approaches in sequence modeling and transduction problems such as language
modeling and machine translation [?, ?, ?]. Numerous efforts have since continued to push the boundaries of recurrent
language models and encoder-decoder architectures [?, ?, ?].

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the
positions to steps in computation time, they generate a sequence of hidden states\( h_{t}\), as a function of the
previous hidden state\( h_{t-1}\), and the input for position\( t\). This inherently sequential nature precludes
parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit
batching across examples. Recent work has achieved significant improvements in computational efficiency through
factorization tricks [?] and conditional computation [?], while also improving model performance in case of the latter.
The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various
tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [?, ?]. In
all but a few cases [?], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an
attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more
parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve
hours on eight P100 GPUs.

## 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [?], ByteNet [?] and
ConvS2S [?], all of which use convolutional neural networks as basic building block, computing hidden representations in
parallel for all input and output positions. In these models, the number of operations required to relate signals from
two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and
logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [?]. In the
Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due
to averaging attention-weighted operations, an effect we counteract with Multi-Head Attention as described in section
3.2.

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single
sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of
tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent
sentence representations [?, ?, ?, ?].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have
been shown to perform well on simple-language question answering and language modeling tasks [?].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on
self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In
the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models
such as [?, ?] and [?].

## 3 Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure [?, ?, ?]. Here, the encoder maps
an input sequence of symbol representations\( x_{1},...,x_{n}\) to a sequence of continuous representations\( z = (z_
{1},...,z_{n})\). Given\( z\), the decoder then generates an output sequence\( y_{1},...,y_{m}\) of symbols one element
at a time. At each step the model is auto-regressive [?], consuming the previously generated symbols as additional input
when generating the next

### Figure 1: The Transformer - model architecture.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers
for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

### Encoder and Decoder Stacks

#### Encoder

The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head
self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a
residual connection [1] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of
each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To
facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of
dimension d_model = 512.

#### Decoder

The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder
layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.
Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.
We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent
positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position i can depend only on the known outputs at positions less than i.

### Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query,
keys, values, and output are all vectors. The output is computed as a weighted sum

### Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of
dimension\(d_{k\), and values of dimension\(d_{v\). We compute the dot products of the query with all keys, divide each
by\(\sqrt{d_{k\), and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix\(Q\).
The keys and values are also packed together into matrices\(K\) and\(V\). We compute the matrix of outputs as:

\[ Attention(Q, K, V) = softmax\left(\frac{QK^{T\sqrt{d_{k}}}\right)V\] (1)

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention.
Dot-product attention is identical to our algorithm, except for the scaling factor of\(\frac{1\sqrt{d_{k}}\). Additive
attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are
similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it
can be implemented using highly optimized matrix multiplication code.

While for small values of\(d_{k\) the two mechanisms perform similarly, additive attention outperforms dot product
attention without scaling for larger values of\(d_{k\)[3]. We suspect that for large values of\(d_{k\), the dot products
grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To
counteract this effect, we scale the dot products by\(\frac{1\sqrt{d_{k}}\).

Footnote 4: To illustrate why the dot products get large, assume that the components of\(q\) and\(k\) are independent
random variables with mean 0 and variance 1. Then their dot product,\(q\cdot k =\sum_{i=1}^{d_{k}} q_{i} k_{i\), has
mean 0 and variance\(d_{k\).

### Multi-Head Attention

Instead of performing a single attention function with\(d\text{model\)-dimensional keys, values and queries, we found it
beneficial to linearly project the queries, keys and values\(h\) times with different, learned linear projections to\(d_
{k\),\(d_{k\), and\(d_{v\) dimensions, respectively. On each of these projected versions of queries, keys and values
then perform the attention function in parallel, yielding\(d_{v\)-dimensional

图中展示的是一个关于神经网络模型的技术文档的一部分，其中包含了模型结构的设计细节和注意力机制的应用。以下是该部分内容的详细说明：

### Multi-Head Attention

多头注意力机制允许模型同时关注不同表示子空间上的信息。与单头注意力相比，平均抑制法被用来防止过拟合。其计算公式如下：

\[ MultiHead(Q,K,V\text{Concat\text{head}_1,...\text{head}_h)W^O\]
\[\text{where head}_i\text{Attention}(QW_{i}^{Q},KW_{i}^{K},VW_{i}^{V})\]

其中，投影矩阵\( W_i^{Q}\in\mathbb{R}^{d\text{model}}\times d_k\)，\( W_i^{K}\in\mathbb{R}^{d\text{model}}\times d_k\)，\(
W_i^{V}\in\mathbb{R}^{d\text{model}}\times d_v\)，以及\( W^O\in\mathbb{R}^{h_{dv}\times d\text{model}}}\)。

在本文中，我们使用了8个并行的注意力层，每个头的大小为\( d_k = d\text{model}} / h = 64\)。由于每个头的维数减少，总的计算成本与全尺寸的头相似。

### Applications of Attention in our Model

Transformer 在三个不同的地方使用了多头注意力：

1. **编码器-解码器注意力**
   ：在编码器和解码器中，查询来自前一层，而记忆键和值来自输入序列。这使得解码器中的每个位置都可以关注输入序列中的所有位置。这种机制模仿了典型的序列到序列模型中的编码器-解码器注意力机制，如 [38, 2, 9]。

2. **自注意力层**：在自注意力层中，所有的键、值和查询都来自同一个地方，即上一层的输出。每个位置的编码器都可以关注到前一层的所有位置。

3. **解码器中的自注意力层**：解码器中的自注意力层使每个位置的解码器关注到解码器上直到该位置的所有位置。我们需要这样做来防止左向信息流，以保持自回归性质。我们通过掩蔽设置（设置为
   -∞）来实现可伸缩的点积注意力，以屏蔽非法连接。参见图2。

### Position-wise Feed-Forward Networks

除了注意力子层外，我们模型中的每个编码器和解码器层都包含一个全连接的前馈网络。它被单独且相同地应用于每个位置。这个网络由两个线性变换组成，中间有一个
ReLU 激活函数。其计算公式如下：

\[\text{FFN}(x\max(0,xW_1+b_1)W_2+b_2\]

虽然线性变换对不同的位置是相同的，但它们的使用参数从层到层是不同的。另一种描述方法是两个卷积，其核大小为1。输入和输出的维度都是\(
d\text{model}} = 512\)，而内层维度是\( d_{ff} = 2048\)。

### Embeddings and Softmax

与其他序列转录模型类似，我们使用学习到的嵌入将输入标记和输出标记转换为维度为\( d\text{model}}\) 的向量。我们也使用通常学习的线性变换和
softmax 函数将解码器的输出转换为预测下一个标记的概率。在我们的模型中，编码器和解码器共享相同的权重矩阵。在嵌入层，我们乘以\(\sqrt{d\text{model}}}\)。

### Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the
sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this
end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The
positional encodings have the same dimension as the embeddings, so that the two can be summed. There are many choices of
positional encodings, learned and fixed [9].
In this work, we use sine and cosine functions of different frequencies:

\[ PE_{(pos, 2i)} = sin(pos / 10000^{2i / d_{model}})\]
\[ PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_{model}})\]

where\( pos\) is the position and\( i\) is the dimension. That is, each dimension of the positional encoding corresponds
to a sinusoid. The wavelengths form a geometric progression from\( 2π\) to\( 10000 * 2π\). We chose this function
because we hypothesized it would allow the model to easily learn to attend to relative positions, since for any fixed
offset\( k\),\( PE_{(pos + k)}\) can be represented as a linear function of\( PE_{(pos)}\).

We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced
nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to
extrapolate to sequence lengths longer than the ones encountered during training.

## 4 Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly
used for mapping one variable-length sequence of symbol representations\((x_1, ..., x_n\) to another sequence of equal
length\((z_1, ..., z_n\), with\( x_i, z_i\in\mathbb{R}^d\), such as a hidden layer in a typical sequence transduction
encoder or decoder. Motivating our use of self-attention we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as
measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key
challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the
length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any
combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12].
Hence we also compare the maximum path length between any two input and output positions in networks composed of the
different layer types.

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed
operations, whereas a recurrent layer requires\( O(n)\) sequential operations. In terms of computational complexity,
self-attention layers are faster than recurrent layers when the sequence

### Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences
were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens. For
English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split
tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length.
Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target
tokens.

### Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the parameters described
throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps
or 12 hours. For our big models (described on the bottom line of table 3), step time was 1.0 seconds. The big models
were trained for 300,000 steps (3.5 days).

### Optimizer

We used the Adam optimizer [20] with\beta_1=0.9$,\beta_2=0.98$ and\epsilon = 10^{-9}$. We varied the learning rate over
the course of training, according to the formula:
\[ lnrate = d_{model}^{-0.5}\cdot min(step\_num^{-0.5}, step\_num\cdot warmup\_steps^{-1.5})\] (3)
This corresponds to increasing the learning rate linearly for the first\( warmup\_steps\) training steps, and decreasing
it thereafter proportionally to the inverse square root of the step number. We used\( warmup\_steps = 4000\).

### Regularization

We employ three types of regularization during training:

### Residual Dropout

We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In
addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder
stacks. For the base model, we use a rate of $ P_{drop} = 0.1 $.

### Label Smoothing

During training, we employed label smoothing of value\epsilon_{ls} = 0.1$. [36]. This hurts perplexity, as the model
learns to be more unsure; but improves accuracy and BLEU score.

### Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

| *                  | d_model                                 | d_{ff}                | h    | d_{k} | d_{v} | P_{drop} | ε_{ls} | train steps | PPL (dev) | BLEU (dev) | params ×10^6 |
|--------------------|-----------------------------------------|-----------------------|------|-------|-------|----------|--------|-------------|-----------|------------|--------------|
| base               | 6                                       | 512                   | 2048 | 8     | 64    | 0.1      | 0.1    | 100K        | 4.92      | 25.8       | 65           |
| (A)                | 4                                       | 512                   | 512  | 1     | 512   | 512      | 0.1    | 100K        | 5.29      | 24.9       | 65           |
|                    | &  &  & 4 & 128                         | 128                   | 0.1  | 100K  | 5.00  | 25.5     | 65     |
| &  &  &  & 16 & 32 | 32                                      | 0.1                   | 100K | 4.91  | 25.8  | 65       |
| (B)                | 6                                       | 32                    | 16   | 16    | 32    | 0.1      | 0.1    | 100K        | 5.01      | 25.4       | 58           |
|                    | &  &  &  &  &  &  &  &  &               |
|                    | 2                                       | &  &  &  &  &  & 5.16 | 25.1 | 58    |
|                    | 4                                       | &  &  &  &  &  & 6.11 | 25.7 | 60    |
| (C)                | 8                                       | 256                   | 32   | 32    | 32    | 0.1      | 0.1    | 100K        | 4.88      | 25.5       | 56           |
|                    | 1024                                    | 1024                  | 128  | 128   | 128   | 0.1      | 0.1    | 100K        | 4.66      | 24.5       | 80           |
|                    | 4096                                    | 4096                  | 16   | 16    | 16    | 0.1      | 0.1    | 100K        | 4.52      | 26.0       | 168          |
|                    | &  &  &  &  &  &  & 4.75                | 25.4                  | 53   |
| (D)                | &  &  &  &  & 0.0                       | 5.77                  | 24.6 |       |
|                    | &  &  &  &  & 0.2                       | 4.95                  | 25.5 |       |
|                    | &  &  &  &  & 0.0                       | 4.67                  | 25.3 |       |
|                    | &  &  &  &  & 0.2                       | 5.47                  | 25.7 |       |
| (E)                | positional embedding instead of sinoids |                       |      |       |       |          |        | 4.92        | 25.7      |            |
| big                | 6                                       | 1024                  | 4096 | 16    | 0.3   | 300K     | 4.33   | 26.4        | 213       |            |
|                    | &  &  &  &  &  &  &  &  &               |

### Parser generalizes well to English constituency parsing (Results are on Section 23 of WSJ)

| Parser                              | Training                 | WSJ 23 F1 |
|-------------------------------------|--------------------------|-----------|
| Vinyals & Kaiser et al. (2014) [37] | WSJ only, discriminative | 88.3      |
| Petrov et al. (2006) [29]           | WSJ only, discriminative | 90.4      |
| Zhu et al. (2013) [40]              | WSJ only, discriminative | 90.4      |
| Dyer et al. (2016) [8]              | WSJ only, discriminative | 91.7      |
| Transformer (4 layers)              | WSJ only, discriminative | 91.3      |
| Zhu et al. (2013) [40]              | semi-supervised          | 91.3      |
| Huang & Harper (2009) [14]          | semi-supervised          | 91.3      |
| McClosky et al. (2006) [26]         | semi-supervised          | 92.1      |
| Vinyals & Kaiser et al. (2014) [37] | semi-supervised          | 92.1      |
| Luong et al. (2015) [23]            | multi-task               | 93.0      |
| Dyer et al. (2016) [8]              | generative               | 93.3      |

# Learning phrase representations using rnn encoder-decoder for statistical machine translation

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio

###### Abstract

In this paper, we present a novel approach to learn phrase representations using recurrent neural network (RNN)
encoder-decoder architectures. Our model learns to represent phrases as vectors that can be used for various tasks such
as translation, question answering, or information retrieval. We show empirically that our approach significantly
improves performance over state-of-the-art methods on several benchmark datasets.

keywords: Statistical Machine Translation, Phrase Representations, Recurrent Neural Networks, Encoder-Decoder Models +
Footnote †: journalyear: 2014

+

Footnote †: journalyear: 2014

## 1 Introduction

Learning representations of phrases is a fundamental problem in natural language processing (NLP). Phrases are groups of
words that often carry semantic meaning beyond individual words and play a crucial role in understanding and generating
natural language. For example, in the sentence "I saw a man walking his dog," the phrase "walking his dog" captures the
action being performed by the subject "man." Understanding and representing these phrases accurately can greatly improve
the performance of NLP systems on various tasks such as translation, question answering, and information retrieval.

Recent work has shown that deep learning techniques, especially convolutional neural networks (CNNs) and recurrent
neural networks (RNNs), can effectively learn to represent phrases [?, ?]. CNNs capture local patterns in text by
applying filters across different window sizes, while RNNs can capture long-range dependencies between words in a
sequence. However, these models often require large amounts of labeled data for training, which can be expensive and
time-consuming to acquire.

In this paper, we propose a novel approach to learn phrase representations using RNN encoder-decoder architectures.
Unlike traditional approaches that focus on single words or short sequences, our model learns to represent phrases as
vectors that can be used for various tasks. We show empirically that our approach significantly improves performance
over state-of-the-art methods on several benchmark datasets.

The rest of the paper is organized as follows. In Section 2, we describe the proposed method in detail. Section 3
presents the experimental setup and results. Finally, Section 4 concludes the paper with future directions.

## 2 Methodology

### Encoder-Decoder Architecture

Our approach is based on the encoder-decoder architecture, which was first introduced by [?] for image caption
generation. The encoder-decoder framework consists of two parts: an encoder and a decoder. The encoder processes the
input sequence into a compact representation, while the decoder uses this representation to generate the output
sequence.

**Encoder:** The encoder takes an input sequence\( x_1, x_2, ..., x_n\) and transforms it into a vector\( z\) that
represents the input sequence. This transformation is achieved through a series of RNN layers stacked together. At each
step, the encoder looks at the current word\( x_t\) and uses its hidden state to predict the next word in the sequence.

**Decoder:** The decoder uses the encoded representation\( z\) to generate the output sequence\( y_1, y_2, ..., y_m\).
It also consists of a series of RNN layers, but unlike the encoder, the decoder uses a separate set of weights for each
layer. At each step, the decoder generates a new word based on the previous generated words and the current hidden
state.

**Attention Mechanism:** To better capture the contextual information within the input sequence, we introduce an
attention mechanism inspired by [?]. Attention allows the decoder to focus on different parts of the input sequence when
generating a particular word. Specifically, during the decoding process, the decoder computes a score for each position
in the input sequence and uses softmax to distribute its attention across the sequence.

**Multi-Head Attention:** Instead of having a single attention head, we use multi-head attention, which was introduced
by [?]. Multi-head attention allows the model to jointly attend to information from different representation subspaces
at different positions. Each head attends to a different subset of the input sequence, allowing the model to capture
more complex dependencies.

**Feed Forward Network:** After computing the attention scores, we feed them back into the decoder along with the
original sequence and the previous hidden states. This is done through a feed

# Building a large annotated corpus of english: The penn treebank

Mitchell P Marcus
Mary Ann Marcinkiewicz
Beatrice Santorini
Computational Linguistics
1992
Volume 19
Issue 2
Pages 313-338

## Effective self-training for parsing

David McClosky
Eugene Charniak
Mark Johnson
Proceedings of the Human Language Technology Conference of the NAACL
2006
Main Conference
Pages 152-159

## A decomposable attention model

Ankur Parikh
Oscar Tackstrom
Dipanjan Das
Jakob Uszkoreit
Empirical Methods in Natural Language Processing
2016

## A deep reinforced model for abstractive summarization

Romain Paulus
Caiming Xiong
Richard Socher
arXiv preprint arXiv:1705.04304
2017

## Learning accurate, compact, and interpretable tree annotation

Slav Petrov
Leon Barrett
Romain Thibaux
Dan Klein
Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL
2006
Pages 433-440

## Using the output embedding to improve language models

Ofir Press
Lior Wolf
arXiv preprint arXiv:1608.05859
2016

## Neural machine translation of rare words with subword units

Rico Sennrich
Barry Haddow
Alexandra Birch
arXiv preprint arXiv:1508.07909
2015

## Outrageously large neural networks: The sparsely-gated mixture-of-experts layer

Noam Shazeer
Azalia Mirhoseini
Krzysztof Mazurczak
Andy Davis
Quoc Le
Geoffrey Hinton
Jeff Dean
arXiv preprint arXiv:1701.06538
2017

## Dropout: a simple way to prevent neural networks from overfitting

Nitish Srivastava
Geoffrey E Hinton
Alex Krizhevsky
Ilya Sutskever
Ruslan Salakhutdinov
Journal of Machine Learning Research
2014
Volume 15
Issue 1
Pages 1929-1958

## Sequence to sequence learning with neural networks

Sainbayar Sukhbaatar
Arthur Szlam
Jason Weston
Rob Fergus
Advances in Neural Information Processing Systems
2015
Volume 28
Pages 2440-2448

## Sequence to sequence learning with neural networks

Ilya Sutskever
Oriol Vinylas
Quoc VV Le
Advances in Neural Information Processing Systems
2014
Volume 31
Pages 3104-3112

## Rethinking the inception architecture for computer vision

Christian Szegedy
Vincent Vanhoucke
Sergey Ioffe
Jonathan Shlens
Zbigniew Wojna
CoRR
abs/1512.00567
2015

## Grammar as a foreign language

Vinyals & Kaiser
Petrov
Sutskever
Hinton
Advances in Neural Information Processing Systems
2015

## Google's neural machine translation system: Bridging the gap between human and machine translation

Yonghui Wu
Mike Schuster
Zhifeng Chen
Quoc V Le
Mohammad Norouzi
Wolfgang Macherey
Maxim Krikun
Yuan Cao
Qin Gao
Klaus Macherey
et al
arXiv preprint arXiv:1609.08144
2016

## Deep recurrent convolutional models with fast-forward connections for neural machine translation

Jie Zhou
Ying Cao
Xuguang Wang
Peng Li
Wei Xu
CoRR
abs/1606.04199
2016

## Fast and accurate shift-reduce

# Attention Visualizations

It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or
voting process more difficult. EOS

### The Law will never be perfect, but its application should be just this way.

In my opinion, we are missing something in what we are doing.

This is what we are missing in my opinion.

# 14_0.png

The Law will never be perfect. But its application should just be what this is - missing in my opinion.
<EOS>
<pad>
