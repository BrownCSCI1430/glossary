# Course Content

> Notes:
>
> - the headers below are numbered by lecture number, then PDF number.
> - we have attempted to be as _precise_ and _accurate_ as possible, but there may be errors. feel free to correct them.
> - it may not be necessary to understand all these concepts with as much _detail_ as is listed below.
> - `ctrl + F` / `cmd + F` is your friend!

`Todo: find a place to define low and high pass filters, preferably after introducing frequency`

---

## 1.1 Introduction to Computer Vision

### Recognition

This refers to the process of attaching **_semantic_ category labels** to objects, scenes, events, and activities in images. [\[1\]][1]

### Reconstruction

Traditionally, this involves the **recovery of three-dimensional geometry** from images.\
More broadly, it can be interpreted as **"inverse graphics"**: estimating shape, spatial layout, reflectance, and illumination. [\[1\]][1]

### Reorganization

This refers to the **grouping/segmentation** of visual elements.<br>It is the computer vision analog of _perceptual organization_ from Gestalt psychology. [\[1\]][1]

[1]: https://www.sciencedirect.com/science/article/pii/S0167865516000313

## 1.2 What is an Image?

### Signal

A signal is a **function** of some variable(s), often time or space.\
Its inputs and outputs typically have physical meaning &mdash; for instance, an image can be understood as a signal: brightness as a function of position.

### Quantization

This is the process of mapping values from a **larger, continuous, and/or infinite set** to values in a **smaller, discrete, and/or finite set**.<br>It is the basis of discrete signal and image processing.

### Resolution

For our purposes ([slide 18](https://drive.google.com/file/d/15rw06o8WOnqjfQMjPjdSkrHJ1Qhn-4R9/view)):\
**Spatial resolution** refers to the linear spacing of a measurement, which, in the context of images, corresponds to the physical separation represented by a pixel &mdash; this could be an angle or a distance.\
**Geometric resolution** is closer in meaning to [angular resolution](https://en.wikipedia.org/wiki/Angular_resolution), which relates to [resolving power](https://en.wikipedia.org/wiki/Angular_resolution#Definition_of_terms), the Rayleigh criterion, and how "blurry" images are.

---

## 2.1 Image Filtering

### Filter

As generally as possible, a filter is a **function of the local neighborhood** which operates on a signal.\
This function can be continuous or discrete, linear or non-linear, ... etc, and the "local neighborhood" can be any size: from a single point to an infinite extent.

### Image Filtering

Image filtering is the computation of a **function of the local neighborhood** of an image **at each position**.\
It may be used to _enhance_, _extract information_ from, or _detect patterns_ in images.\
\
**Image filtering** can be implemented by **convolving** an **image** and a **filter**.\

### Convolution

As generally as possible, convolution is an **operation** which takes two functions and produces a third function, one whose value at some point X is the **integral** of the **product** of the first function and the second function **flipped in the x direction then offset by X**. Whew!\
\
In the context of digital image processing, "integrals" are just sums over matrix elements, "products" refer to Hadamard products, and "flipping" is just rotating a 2D matrix by 180°. `Todo: add link to animation`\
\
**Important: _convolution is both commutative and associative._**

### Correlation

Correlation is the same as convolution, but **without the flipping**.\
Observe that if you are using a 180°-rotationally-symmetric kernel, then convolution and correlation are identical.\
\
**Important: _correlation is neither commutative nor associative._**

### Kernel

In the context of digital image processing, a kernel is a **2D matrix** which acts as a filter when **convolved** with another 2D matrix, typically an image.

### Separability

When used to describe a kernel, separability refers to the kernel's ability to be **factored out** as the **product of two 1D kernels** (one row and one column vector).\
\
Given a separable kernel `K` which factors out into `R` and `C`, and an image `I`:\
`K * I == R * (C * I) == C * (R * I)`, where `*` represents the convolution operator.

### Linearity and Shift Invariance

These are properties of operations.\
\
Suppose you have some operation `T` such that `y(t) = T( x(t) )`.\
If `T` is linear: `T( a * x1(t) + b * x2(t) ) = a * y1(t) + b * y2(t)`\
If `T` is shift-invariant: `y(t - s) = T( x(t - s) )`\
\
Any operation which is **both linear and shift-invariant** can be represented as a `convolution`.\
Convolution itself is also linear and shift-invariant.

## 2.2 Thinking in Frequency I

### Aliasing

Aliasing refers to when a signal becomes **indistinguishable** from a different signal, due to **sampling**.\
\
An example is when car wheels appear to spin the wrong way in videos &mdash; the orientation of the wheel with time is the signal, and the video's frames are the samples. If your sampling rate is too low, a fast clockwise rotation can look exactly like a counter-clockwise one.

### Nyquist-Shannon Sampling Theorem

The Nyquist-Shannon Sampling Theorem provides a rule for sampling which prevents aliasing:\
\
_When sampling a signal **at discrete intervals**, the sampling frequency must be `≥ 2 * f_max`, where `f_max`is the absolute maximum frequency of the input signal._\
\
If this rule is followed, it is possible reconstruct the original signal _perfectly_ from its samples &mdash; lossless compression.

### Preventing Aliasing

Following the Nyquist-Shannon sampling theorem, you can either:\
a. Increase the sampling rate, or\
b. Decrease the maximum frequency of the input signal. This can be done via Gaussian filtering.

### Hybrid Images

These images are formed by combining the high-frequency components of one image with the low-frequency components of another image.\
The result is a third "hybrid" image that looks like either the first or the second image, depending on the scale at which it is viewed.

---

## 3.1 Thinking in Frequency II

### Template Matching

This refers to image filtering when viewed as "comparing an **image of what you want to find** (as the filter) against another image".\
\
This involves zero-centering your selected filter by subtracting the mean of its pixels, then correlating the filter with the image (or equivalently, convolving the flipped filter with the image).

### Fourier Theorem

_Any univariate function can be rewritten as a weighted sum of sines and cosines of different frequencies._

### Fourier Transform, Decomposition, Series

The Fourier **transform** of a function is the representation of that function as a weighted sum of **Fourier basis functions**. The process of breaking a function into its Fourier basis functions is known as Fourier **decomposition**.\
\
The Fourier **series** is similar to the Fourier transform, but it is used exclusively for _periodic_ functions. So, unless you're taking some strange photos, you'll probably want the Fourier transform instead.

### Fourier Basis Functions

These are simply sines and cosines of different (1) **amplitudes** (weights) and (2) **frequencies**.\
\
In 2D Fourier decomposition (of 2D images, say), we use 2D sinusoids: amplitude (and phase) are scalar values just as with 1D sinusoids, but frequency is now a 2D vector, since you need to account for rate of change in both directions.

### Amplitude-Phase Form

The amplitude-phase form aims to encode the Fourier transform of an image.\
\
Recall that the sum of a sine and a cosine function, each with some amplitude but the same frequency, is simply a third sinusoid with some **phase offset**. Thus, we can represent every term of a Fourier decomposition with three values: (1) amplitude, (2) frequency, and (3) phase.\
\
The amplitude-phase form encodes this information in two signals: (A) amplitude as a function of frequency, and (B) phase as a function of frequency.\
Because we're using 2D sinusoids, frequency is 2D: therefore, signals (A) and (B) are typically represented as images, where the position of each pixel is the frequency, and the intensity of that pixel is the corresponding amplitude/phase of the term with that frequency.

### Spatial Domain vs Fourier/Frequency Domain

Recall that an image can be thought of as brightness as a function of 2D **position**. Since the input is a point in space, we can say that this image is in the spatial domain.\
An amplitude-phase form image, on the other hand, is amplitude/phase as a function of 2D **frequency**, and is thus in the Fourier/frequency domain.

---

## 4.1 Thinking in Frequency III

### Duality, or, The Convolution Theorem

Convolution in the spatial domain is equivalent to (element-wise) multiplication in the frequency domain.\
Consequently, the **Fourier transform** of the convolution of two functions is the product of their Fourier transforms.

### Image Filtering in the Frequency Domain

Recall that image filtering is implemented by the convolution of an **image** and a **filter**. Now that we understand **the convolution theorem**, we can view image filtering as the product of the Fourier transforms of the image and the filter.

### Box / Sinc Dual

The Fourier transform of a box function is a sinc function, and vice versa.\
This is slightly troublesome: a box function in the frequency domain would be an ideal low-pass filter, but to implement it, you'd need a filter that looks like a sinc function in the spatial domain.\
Unfortunately, sinc functions are infinite in extent, and we do not have infinitely-wide filters.

### Artifacts

What if you tried to blur an image with a **box filter**?\
\
You'd get artifacts. The Fourier transform of a box filter is a sinc, which has non-zero components in the high-frequency range. Consequently, your output image will retain any existing high-frequency components.\
`Todo: add link to example image`

### Ringing Artifacts / Gibbs Phenomenon

What if you tried to blur an image with an **approximation of a sinc filter**?\
\
You'd still get artifacts. Because the approximation is imperfect, the Fourier transform of this filter will be an imperfect box with **overshoots near the discontinuities** (see: [Gibbs phenomenon](https://en.wikipedia.org/wiki/Gibbs_phenomenon)). Consequently, your output image will exhibit "ringing" artifacts near edges.\
`Todo: add link to example image`

## 4.2 Edge Detection

### Edges

An edge is a place of rapid change in the image intensity function.\
This change can be in overall brightness, or in colors &mdash; consider a sudden jump from #FF0000 (red) to ##00FF00 (blue).\
\
Edges are "high-frequency content" &mdash; that is, they correspond to the image's high frequency components.

### Edge Detection via Taking the Derivative

Because of their association with _high rate of change_, edges correspond directly to **extrema in the first derivative of image signals**.\
However, because of the presence of noise, we can't simply take the derivative of an image &mdash; it must first be smoothed (e.g. with a Gaussian filter).

### 1D Gaussian Filter

A Gaussian filter is a filter whose shape in the spatial domain is a Gaussian function.\
Interestingly, its Fourier transform (i.e. its shape in the frequency domain) is simply another Gaussian (with inverted sigma). \
**It is thus useful as a low-pass filter, e.g. for _blurring/smoothing_**.\
\
Even though the Gaussian function is technically of infinite extent, in practice, it is effectively zero three standard deviations away from the mean, which is why we can approximate it fairly well with discrete Gaussian kernels.

### 2D Gaussian Filter

The above holds true for the 2D Gaussian filter.\
2D Gaussian filters are **separable**.

### The 1st Derivative of a Gaussian

Recall that in order to find edges, we need to (1) **blur** (convolve by a Gaussian), then (2) **differentiate** (convolve by a kernel which achieves differentiation).\
Since convolution is [**differentiable**](https://en.wikipedia.org/wiki/Convolution#Differentiation), we can combine these two steps, and instead convolve our image by the 1st derivative of a Gaussian.

### Detection and Localization

These are qualities of edge detectors.\
\
**Detection** refers to the detector's ability to find all real edges, ignoring noise and other artifacts.\
**Localization** refers to the detector's ability to return a single-point output close to the true edge.

### Canny Edge Detector

The [**Canny edge detector**](https://ieeexplore.ieee.org/document/4767851) is "probably the most widely-used edge detector in computer vision".\
Please refer to the slides for steps!

### Non-Maximum Suppression

This refers to the reduction of multi-pixel wide "ridges" to single-pixel wide lines, which is achieved by discarding pixels which are not the local maximum along the direction of most change (the gradient).

### Hysteresis Thresholding

Regular thresholding involves discarding areas below a certain fixed threshold value.\
This can easily be improved by setting, say, two thresholds corresponding to "weak" and "strong" edges.\
**Hysteresis** thresholding further improves on this, by discarding "weak" edges which are not connected to "strong" edges. Refer to the [`scikit-image` documentation](https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html) for a visual example.

---

## 5.1 Interest Points and Corners

## 5.2 Local Image Features

---

## 6.1 Feature Matching

## 6.2 Light and Color

---

## 7.1 Camera Geometry

---

## 8.1 Camera Calibration

## 8.2 Stereo Vision

---

## 9.1 Epipolar Geometry, Stereo Disparity Matching, and RANSAC

---

## 10.1 Reconstruction and Depth Cameras

---

## 11.1 Machine Learning: Unsupervised Learning

---

## 12.1 Machine Learning: Supervised Learning

---

## 13.1 Recognition, Bag of Features, and Large-scale Instance ### Recognition

---

## 14.1 Large-scale Scene Recognition and Advanced Feature Encoding

## 14.2 Detection with Sliding Windows: Dalal Triggs

---

## 15.1 Detection with Sliding Windows: Viola Jones

## 15.2 Descriptor Failure and Big Data

---

## 16.1 Neural Networks and Convolutional Neural Networks

---

## 17.1 Training Neural Networks

---

## 18.1 What do CNNs learn?

---

## 19.1 Architectures: ResNets, R-CNNs, FCNs, and UNets

---

## 20.1 Social Good and Dataset Bias

---
