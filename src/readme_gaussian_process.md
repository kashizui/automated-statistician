# On the Behavior of Gaussian Processes
There are a few interesting behaviors of that a gaussian process exhibits, depending largely on how one sets it up. This readme goes over the tunable parameters and the effect on the resulting GP regression depending on the choice of parameters.

## Tunable Parameters
The tunable parameters are:
```
1. noise
2. length_scale
```
### Noise
The noise parameter reflects how much noise your gaussian process thinks exists in the data it is observing. This is a very powerful tool, because it essentially indicates how much noise you're willing to tolerate within the system. If the noise is set to zero, the gaussian process regression will try its darndest to fit everything **perfectly**.

This behavior of the noise parameter comes with a massive gotcha: if you choose a noise that is **too low**. When this happens, all hell breaks loose. What is too low? Side-stepping the actual math, it has to do with the invertibility of the covariance matrix K(X, X). When two observations (x_1, x_2) are the same point (aka suppose it is a very crowded space!), then K(x_1, X) and K(x_2, X) will be exactly identical unless noise is allowed (see note).

One must be very careful when tuning this parameter in order to not cause singular matrix and run-time error. The singular matrix results from a combination of: how close the sample points are in the parameter space, the amount of noise introduced, and the length scales being used. 

Note: this rationale is not completely satisfactory. I'm needing to add crap tons of noise just to deal with the matrix invertibility and compact space problem. Whereas Rasmussen only needed to add a very very small amount of noise. Will need to double check this again later.

### Length scale

The length scale describes the extent to which the function evaluation at some point x can correlate with the function evaluation at the (unseen) points around x. Points within the length scale region of x tend to behave similarly to x. The greater the length scale, the greater the reach of influence.

With the length scale is very small, the prior dominates the behavior of the mean function at unseen points, with sharp spikes to the observed function evaluations at observed point locations. With the length scale is too big, it's equivalent (I think) to just averaging over all the function evaluations at seen points. A properly chosen length scale allows for the smooth varying of the function across its parameter space. 

It is very important to note that the the cost function is not convex with respect to the length scales. When doing optimization, it may be important to try random initializations within the region of interest, and then picking the best length scales. A safe bet is to pick a length scale within the range of the observed space.

### Tuning the Parameters
For carefully chosen noise (how do we decided beforehand?), length scale tuning is a very safe endeavor. Tuning both at the same time can be very scary. One possibility to address is this an EM-like operation, that fixes one and tunes the other. Hopefully such an approach makes the tuning feasible.