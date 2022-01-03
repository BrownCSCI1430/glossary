# Course Content

> Note: the headers below are numbered by lecture number, then PDF number.

---

### 1.1 Introduction to Computer Vision

#### Recognition

This refers to the process of attaching **_semantic_ category labels** to objects, scenes, events, and activities in images. [\[1\]][1]

#### Reconstruction

Traditionally, this involves the **recovery of three-dimensional geometry** from images.\
More broadly, it can be interpreted as **"inverse graphics"**: estimating shape, spatial layout, reflectance, and illumination. [\[1\]][1]

#### Reorganization

This refers to the **grouping/segmentation** of visual elements.<br>It is the computer vision analog of _perceptual organization_ from Gestalt psychology. [\[1\]][1]

[1]: https://www.sciencedirect.com/science/article/pii/S0167865516000313

### 1.2 What is an Image?

#### Signal

A signal is a **function** of some variable(s), often time or space.\
Its inputs and outputs typically have physical meaning &mdash; for instance, an image can be understood as a signal: brightness as a function of position.

#### Quantization

This is the process of mapping values from a **larger, continuous, and/or infinite set** to values in a **smaller, discrete, and/or finite set**.<br>It is the basis of discrete signal and image processing.

#### Resolution

For our purposes ([slide 18](https://drive.google.com/file/d/15rw06o8WOnqjfQMjPjdSkrHJ1Qhn-4R9/view)):\
**Spatial resolution** refers to the linear spacing of a measurement, which, in the context of images, corresponds to the physical separation represented by a pixel &mdash; this could be an angle or a distance.\
**Geometric resolution** is closer in meaning to [angular resolution](https://en.wikipedia.org/wiki/Angular_resolution), which relates to [resolving power](https://en.wikipedia.org/wiki/Angular_resolution#Definition_of_terms), the Rayleigh criterion, and how "blurry" images are.

---

### 2.1 Image Filtering

#### Filter

As generally as possible, a filter is a **function of the local neighborhood** which operates on a signal.\
This function can be continuous or discrete, linear or non-linear, ... etc, and the "local neighborhood" can even be infinite in extent.

#### Image Filtering

It is the computation of a **function of the local neighborhood** of an image **at each position**.\
We use filtering to enhance, extract information from, and detect patterns in images.

#### Convolution

As generally as possible, convolution is an **operation** which takes two functions and produces a third function whose value at some point X is the _integral_ of the _product_ of the first function and the second function _flipped in the x direction then offset by X_. Whew!\
\
This is relevant to us because the convolution of a given **image** and **filter** is the result of **image filtering**!\
Note: In the context of digital image processing, _integrals_ are just sums over matrix elements, _products_ are Hadamard products, and _flipping_ is just rotating by 180°.\
\
`Todo: check if this is a necessary and sufficient explanation`

#### Correlation

Correlation is the same as convolution, but without the flipping: if you can do one, you can do the other, simply by doing the flipping yourself.\
**_Correlation is not commutative, but convolution is_**.

#### Kernel

In the context of digital image processing, a kernel is a **2D matrix** which acts as a (linear) filter when **convolved** with another 2D matrix, typically an image.

#### Separability

When used to describe a kernel, separability refers to the kernel's ability to be **factored out** as the **product of two 1D kernels** (one row and one column vector).\
Given a separable kernel `K` which factors out into `R` and `C`, and an image `I`:\
`K * I == R * (C * I) == C * (R * I)`, where `*` represents the convolution operator.

#### Linearity

When used to describe a kernel, linearity refers to... `Todo`

#### Correlation and Convolution

`Todo`

### 2.2 Thinking in Frequency I

---

### 3.1 Thinking in Frequency II

---

### 4.1 Thinking in Frequency III

### 4.2 Edge Detection

---

### 5.1 Interest Points and Corners

### 5.2 Local Image Features

---

### 6.1 Feature Matching

### 6.2 Light and Color

---

### 7.1 Camera Geometry

---

### 8.1 Camera Calibration

### 8.2 Stereo Vision

---

### 9.1 Epipolar Geometry, Stereo Disparity Matching, and RANSAC

---

### 10.1 Reconstruction and Depth Cameras

---

### 11.1 Machine Learning: Unsupervised Learning

---

### 12.1 Machine Learning: Supervised Learning

---

### 13.1 Recognition, Bag of Features, and Large-scale Instance ### Recognition

---

### 14.1 Large-scale Scene Recognition and Advanced Feature Encoding

### 14.2 Detection with Sliding Windows: Dalal Triggs

---

### 15.1 Detection with Sliding Windows: Viola Jones

### 15.2 Descriptor Failure and Big Data

---

### 16.1 Neural Networks and Convolutional Neural Networks

---

### 17.1 Training Neural Networks

---

### 18.1 What do CNNs learn?

---

### 19.1 Architectures: ResNets, R-CNNs, FCNs, and UNets

---

### 20.1 Social Good and Dataset Bias

---
