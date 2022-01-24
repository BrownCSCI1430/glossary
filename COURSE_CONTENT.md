# Course Content

> Notes:
>
> - the headers below are numbered by lecture number, then PDF number.
> - we have attempted to be as _precise_ and _accurate_ as possible, but there may be errors. feel free to correct them.
> - it may not be necessary to understand all these concepts with as much _detail_ as is listed below.
> - `ctrl + F` / `cmd + F` is your friend!

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
It may be used to _enhance_, _extract information_ from, or _detect patterns_ in images.

**Image filtering** can be implemented by **convolving** an **image** and a **filter**.

### Low-pass Filter

A low-pass filter, when applied to an image, averages out the changes in pixel intensity, creating a **blurring** or **smoothing** effect. For each pixel, its value will be replaced with some average of its 8 immediate neighbors.

### High-pass Filter

A high-pass filter, when applied to an image, emphasizes the details and contrasts between pixel intensities, creating a **sharpening** effect. If one pixel is brighter than its immediate neighbors, it gets boosted.

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
_When sampling a signal **at discrete intervals**, the sampling frequency must be `≥ 2 * f_max`, where `f_max` is the absolute maximum frequency of the input signal._\
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

### The Convolution Theorem

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

### Features

These are parts of an image which are distinctive and likely useful for computing similarities between images.\
A feature typically consists of (1) a **key/interest point**, which has some position, and (2) a **feature descriptor**, which encodes some information about the point's immediate neighborhood / surrounding patch in the image. Either (1) or (2) might also include scale or orientation information, among other things.

### Detection, Description, and Matching

These are the three steps involved in identifying feature point correspondences between multiple images.\
\
**Detection** refers to the process of finding **key points**.\
**Description** refers to the process of extracting **feature descriptors** from the areas around each key point.\
**Matching** refers to the process of comparing the **features** of two or more images, and figuring out which features in each correspond to features in the others.

### Distinctiveness, Repeatability, and Compactness/Efficiency

These are qualities of feature representations.\
\
**Distinctiveness** refers to the ability to uniquely identify a point, which may be challenging, regardless of representation, in images with repeated elements. This can be\
**Repeatability** refers to the ability to locate the same feature in multiple images despite _geometric_ and _photometric_ differences.\
**Compactness/Efficiency** refers to the ability of the representation to be as small (compact) as possible, for performance.

### Geometric Transformations

These refer to transformations in translation, rotation, scale, and perspective, etc.

### Photometric Transformations

These refer to transformations in reflectance and illumination, etc.

### Corners

To get _distinctive_ and _repeatable_ features, we want to look for points which are stable in appearance, with respect to small variations in position.\
Thus, we choose to look for **corners**: when looking at them through a small **window**, shifting away from them in _any direction_ will cause a large change in the window's overall intensity.

### Corner Detection by Auto-Correlation

In this case, auto-correlation refers to our method of looking through a window, evaluating intensity, and comparing it with the intensity resulting from a shifted window. We are effectively correlating (a part of) the image with (another part of) itself, hence auto-correlation.\
\
We would like to do this for every point in the image, and keep the points where the auto-correlation function looks like a strong **peak**. You might say that such points have a high "corner-ness" score.\
\
Unfortunately, this requires a lot of computation, far too costly for our purposes.

### Harris Corner Detection

The Harris corner detector solves the above problem by way of approximations and linear-algebraic manipulations, ultimately reducing the cost of computing a "cornerness" score for each point (see the entry for the _Harris Cornerness Score_ below).\
Please refer to the slides for steps!

### Taylor Series Expansion

The Taylor series of a function is an infinite series of terms that are expressed in terms of the function's derivatives at a single point. Most functions are equal to the sum of their Taylor series near that point.\
\
Taylor series expansion refers to the process of finding these terms for a given function.\
It was one of tools used to arrive at the Harris corner detector algorithm, in approximating the computation of the auto-correlation function.

### Second Moment Matrix (M)

This is a 2x2 matrix with the following elements:\
`⎡ Σ( I_x ^ 2 ), Σ( I_x * I_y ) ⎤`\
`⎣ Σ( I_x * I_y ), Σ( I_y ^ 2 ) ⎦`\
Where `I_x` and `I_y` refer to the image derivatives, at some point, in the x and y directions respectively. These summations are over the window described earlier.\
\
This matrix is square, [symmetric](https://en.wikipedia.org/wiki/Symmetric_matrix), and [diagonalizable](https://en.wikipedia.org/wiki/Diagonalizable_matrix).\
It has two eigenvalues `λ_1` and `λ_2` &mdash; please refer to the slides for their visual interpretation!

### Harris Cornerness Score

This is equal to `C`, where `C = λ_1 * λ_2 - α * (λ_1 + λ_2) ^ 2` and `α` is some constant around 0.04 - 0.06.\
\
With the second moment matrix, its determinant is equal to the product of its eigenvalues, and its trace is the sum of its eigenvalues, so this equation can be rewritten as: `C = det(M) - α * trace(M) ^ 2`, thereby avoiding even having to calculate the eigenvalues.

### Invariance and Covariance

Loosely, invariance = "does not change with", and covariance = "changes with".\
Ideally, we'd like features to be _invariant_ to **photometric** transformations and _covariant_ to **geometric** ones.\
\
Harris corner locations are covariant wrt translation and rotation, **but not scaling(!)**. They are also only partially invariant to affine intensity changes, due to the effects of thresholding.

## 5.2 Local Image Features

### Templates and Histograms

These are two ways to represent image features. They can be used as **feature descriptors**.\
\
**Templates** are basically smaller images (intensities, gradients) that can be compared against the local region of a feature point.\
**Histograms** are simply counts or bins of the presence of certain "sub"-features, like particular colors or textures (e.g. oriented gradients), again in a window around the key point.

### Scale-Invariant Feature Transform (SIFT)

SIFT is an algorithm used to detect, describe, and match local features in images, invented by David Lowe in 1999. Here's [the paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).

### SIFT Descriptors

These are the feature descriptors used in SIFT &mdash; each feature point is represented by a vector (of length 128) containing information about orientations in its vicinity.\
Gradients of a 16x16 area around the point are found; then, this area is broken into 16 4x4 areas, and the gradients are binned into 8 orientations.\
The algorithm also performs:\
\- trilinear interpolation to smooth the output vector,\
\- gaussian weighting of gradient magnitudes to prioritize orientations closest to the center of the 16x16 region,\
\- some normalization/clamping of the output vector to reduce the effect of illumination,\
\- dominant orientation estimation to gain invariance to rotation.

---

## 6.1 Feature Matching

### Euclidean Distance

This is a measure of the **magnitude of the difference** between two vectors.\
Given two vectors `p` and `q`, it is equal to `sqrt( Σ( (p[i] - q[i])^2 ) )`.

### Cosine Similarity / Cosine Distance

This is a measure of the similarity between two non-zero(!) vectors in terms of the **cosine of the angle between them**.\
Given two vectors `p` and `q`, it is equal to `dot(p, q) / norm(p) / norm(q)`, where `dot` produces a dot product, and `norm` gets the magnitude of the vector.

### Nearest Neighbor Distance Ratio (NNDR)

In the context of feature matching, and given some notion of distance between two features, the NNDR is equal to `NN1`, the distance between a point and its nearest neighbor, divided by `NN2`, the distance between that same point and its second closest neighbor.

### SURF, Shape Context, Geometric Blur, Self-Similarity, etc...

These are all other ways to get local image description.\
\
**SURF** is a fast approximation of SIFT, which uses 2D box filters & integral images.\
**Shape context** bins points instead of orientations, and uses log-polar binning.\
... and so on. (refer to the slides if you're interested!)

## 6.2 Light and Color

### Frequency Spectrum

Light can be _completely described_ physically by its **spectrum**, the intensity of light (per unit time) for each wavelength.\
You've probably seen this yourself: a rainbow, or a prism or diffraction grating, can split light into its separate wavelengths and reveal its spectrum.

### Cones and Rods

These are photoreceptors (light-receiving cells) that can be found in the retina of your eye.\
\
**Cones** detect light around a specific color, or to be precise, wavelength. Most humans are trichromats: we have cones corresponding to the 564–580 nm, 534–545 nm, and 420–440 nm wavelengths, which provides our color vision.\
**Rods** detect light of a much wider range of wavelengths (for humans, that's the visible spectrum). They are much more sensitive than cones and thus contribute significantly to our low-light vision, but are also "color-blind".

### Color Cameras

Cameras produce images with color by encoding three separate color channels.\
At its simplest, a color camera has three sensors, one per color channel: light comes in, gets split based on its color, and hits the appropriate sensor. The requisite splitting can be achieved by an arrangement of prisms.

### Bayer Filter

Bayer filters are a cheaper and more compact solution to color imaging &mdash; instead of using optical splitting and three sensors, a grid of **color filters** is placed over a square grid of sensors. Twice as many "green" sensors are used than red and blue, both to approximate human spectral sensitivity, and make it a _lot_ easier to tile 3 colors into a 2x2 pattern unit.

---

## 7.1 Camera Geometry

### Cameras

A camera can be understood as a **dimension reduction machine**, which maps a 3D world to a 2D image.

### Parametric Global Transformations

A parametric global transformation is one which is the same for all points p (global), and can be described by a few numbers (parameters).

### Classes of Transformations

\- **Displacement**: preserves distances, oriented angles, and handedness. _Translations_ only.\
\- **_Proper_-Rigid**: preserves distances, **non-oriented** angles, and handedness. _Rotations_, plus the above.\
\- **Rigid/Euclidean**: preserves distances and non-oriented angles. _Reflections_, plus the above.\
\- **Similarity**: preserves **ratios** of distances, and non-oriented angles. _Scaling_, plus the above.\
\- **Affine**: preserves ratios of distances, and the **parallel-ness** of lines. _Skewing/shearing_, plus the above.\
\- **Projective**: preserves the **straightness** of lines (collinearity). _Projective warps_, plus the above.

### Linear Transformations

These are transformations which can be represented as **matrices**, such that applying the transformation on a point is just a matter of matrix multiplication.\
Note that **translation** for a point in an n-dimensional space is **_not linear_**, unless you look at the transformation from an (n+1)-dimensional space (see: homogenous coordinates).

### Homogenous Coordinates
Converting to homogeneous coordinates requires adding a dimension. Converting from homogenous coordinates requires dividing the original dimension values by the extra dimension value that has been added when conversion. Scale invariance between homogeneous coordinates and cartesian coordinates.

### Pinhole Cameras
Involves sensor, pinhole or also known as camera center. Camera center is at the optical center of the camera where all light rays converge, which is the origin of camera coordinates. Focal length is the distance between the sensor and the camera center. Virtual image is projected at one focal length from the camera center towards the real object.

### Aperture
Aperture refers to the opening of a lens's diaphragm through which light passes. It expands and shrinks to allow more or less light through to a camera's sensor. It is a pinhole in camera. Varies in size in most cameras.

### Lenses
The lens captures the image and delivers it to the image sensor in the camera. Lens will vary in optical quality and price, the lens used determines the quality and resolution of the captured image. Depending on the lense you use, there may be radial distortion of the image.

### Depth of Field
Depth of field is one of the essential concepts in photography. Depth of field in a photo refers to the distance between the closest and farthest objects that appears acceptably sharp. Depth of field differs based on camera type, aperture, and focusing distance. The depth of field increases with a narrower aperture and thus increases focus on an object.

### Field of View, Zoom
The field of view is the extent of the observable world that is seen at any given moment. The field of view depends of focal length
where a smaller field of view = larger focal length. Thus, to zoom in, you would have a larger field of view and a smaller focal length.

### Chromatic Aberration
Chromatic aberration, also known as color fringing, is a color distortion that creates an outline of unwanted color along the edges of objects in a photograph. Often, it appears along metallic surfaces or where there's a high contrast between light and dark objects, such as a black wall in front of a bright blue sky.

### Barrel and Pin-cushion Distortion
Both are a type of fadial distortion where straight lines curve around the image center. Barrel refers to a negative distortion while pin-cushion refers to positive distortion

### Camera Projection Matrix
A matrix which describes the mapping of a pinhole camera from 3D points in the world to 2D points in an image.

### Extrinsic Matrix
Defines everything outside of the camera. Defines the coordinate space transformation between the two coordinate systems.

### Intrinsic Matrix
Defines everything internal of the camera, e.g. focal length, center of projection image

### Orthographic Projection
Orthographic projection (Also called “parallel projection”) where the distance from the COP to the image plane is infinite. It is a kind of parallel projection where the projecting lines emerge parallelly from the object surface and incident perpendicularly at the projecting plane.

---

## 8.1 Camera Calibration

### Estimating A Camera Matrix with Known Points

### Linear Least-Squares Regression
The line that makes the vertical distance from the data points to the regression line as small as possible. It's called a “least squares” because the best line of fit is one that minimizes the variance (the sum of squares of the errors).

### Total Least-Squares Regression


## 8.2 Stereo Vision

### Disparity
The difference in distance between the object location seen by the two cameras in stereo vision, resulting from the horizontal separation of the cameras.

---

## 9.1 Epipolar Geometry, Stereo Disparity Matching, and RANSAC

### Triangulation, aka Estimating Depth With Calibrated Stereo Cameras + Point Correspondences

### Finding (Good) Point Correspondences

### Epipolar Constraints

### Baseline
The line that connects two cameras at their camera centres, in stereo vision
### Epipoles
Lies along the baseline, right where the baseline meets either camera planes. A projection of the other camera's image centre.

### Epipolar Plane
The plane created by an epipolar line and the baseline.

### Epipolar Lines
The straight line of intersection of the epipolar plane with the image plane.

### Essential Matrix
Relates corresponding pairs of normalised homogeneous image points across pairs of images.
Used when finding the geometry of the camera system when the amera is calibrated.
Given `t` as translation matrix and `R` as rotation matrix, the essential matrix can be found by `t x R`

### Fundamental Matrix

### Estimating A Fundamental Matrix with Stereo Point Correspondences

### 8-Point Algorithm

---

## 10.1 Reconstruction and Depth Cameras

### Epipolar Relation

### Random Sample Consensus (RANSAC)

### Lambertian Reflectance

### Rectification

### Correspondence Problem

### Dense Correspondence Search

### Similarity Cost

### Sum of Squared Differences (SSD)

### Normalized Correlation

### Disparity vs Depth

### Uniqueness Constraint

### Occlusion

### Disparity Gradient Constraint

### Ordering Constraint

### Active Stereo with Structured Light

### LiDAR

### Iterative Closest Points (ICP) Algorithm

---

## 11.1 Machine Learning: Unsupervised Learning

### Unsupervised Learning

### ImageNet

### Dimension Reduction

### Principal Component Analysis

### Eigenfaces

### Clustering

### Segmentation

### K-Means Clustering

### Generative, Discriminative

### Agglomerative Clustering

### Mean Shift Clustering

### Spectral Clustering

---

## 12.1 Machine Learning: Supervised Learning

### Supervised Learning

### Training, Validation, and Testing Sets

### Coverage, Concision, Directness

### Classification

### The Machine Learning Framework

### Nearest Neighbor Classifier

### Voronoi Cells/Partitioning

### K-Nearest Neighbor Classifier

### Linear Classifier

### Naïve Bayes

### Logistic Regression

### Support Vector Machines

### Linear SVMs

### Nonlinear SVMs

### Kernel Trick

### One vs Others, One vs One

---

## 13.1 Recognition, Bag of Features, and Large-scale Instance Recognition

### Classification Algorithm Ideals

### Generalization

### Generalization Error

### Bias/Variance Trade-Off

### Underfitting

### Overfitting

### Generative and Discriminative Classifiers

---

## 14.1 Large-scale Scene Recognition and Advanced Feature Encoding

### Visual Words

### Global Image Descriptors

### Global Texture Descriptors

### Bag of Words (may be covered in an earlier lecture)

### GIST Descriptors

### Gaussian Mixture Model

### The Fisher Vector

## 14.2 Detection with Sliding Windows: Dalal Triggs

### Object Model

### Dalal-Triggs Pedestrian Detector
---

## 15.1 Detection with Sliding Windows: Viola Jones

### Viola Jones Face Detector

### Boosting

### Attention Cascade

## 15.2 Descriptor Failure and Big Data

### Discriminative Part-Based Models

### PASCAL VOC 2009

### Paired Dictionary

### Amazon Mechanical Turk
---

## 16.1 Neural Networks and Convolutional Neural Networks

### Separability in Linear Spaces

### Perceptron

### Binary Classifier

### MultiClass Classifier

### Non-Linearities

### Rectified Linear Unit (ReLU)

### Hyperparameter

---

## 17.1 Training Neural Networks

### Convolutional Layer

### Stride

### Pooling Layer

### Dropout Layer

### Local Contrast Normalization

### Hidden Unit

### Feature Map

### Gradient Descent

### Loss

### Softmax

### Training

### Backward Propogation

### Stochastic Gradient Descent

### Data Augmetation

---

## 18.1 What do CNNs learn?

### Momentum

### Regularization

### Occlusion

### Adversarial Patches

### Saliency Maps

### Local Interpretable Model-agostic Explanations (LIME)

### Superpixels

### Neural Style

---

## 19.1 Architectures: ResNets, R-CNNs, FCNs, and UNets

### Batch Normalization

Rescales each training data batch by subtracting the mean and dividing by the standard deviation per batch so that the gradients are well-behaved, in that they do not explode or vanish.

### AlexNet

A convolutional neural network architecture developed in 2012 with 8 learnable layers that achieved a 15.3% error on the ImageNet dataset.

### Google LeNet

A convolutional neural network architecture developed in 2014 with 22 learnable layers that uses inception modules containing parallel layers. LeNet improved on the results of AlexNet's performance to achieve an error of 6.67% on the ImageNet dataset.

### ResNet

### Decoder Networks

### UNet

### Two-Stream Networks
---

## 20.1 Social Good and Dataset Bias


not adding to until src plans finalized

---
