# Why histogram a reciprocal-space map

Each pixel at each step samples one point in (h, k, l). A rocking scan therefore gives a cloud of points, dense in
some places, sparse in others, and absent outside the swept volume.

`cristallina.rsm.RSM_interpolate` evaluates `griddata(points, values, grid, method="nearest")`. Every grid point takes
the value of the single nearest pixel:

- **No averaging.** Where many pixels fall into one voxel, all but one are discarded. The map is as noisy as a
  single pixel.
- **No coverage information.** Grid points far from any pixel still get a value, copied from whichever pixel is
  nearest. Unmeasured regions look measured.
- **No errors.**

[`histogram`][cristallina_scipp.rsm.histogram] sums counts per voxel and the monitor (I0 of each contributing pixel's
step) per voxel, then divides:

```text
I(voxel) = sum of counts C_p over pixels p in the voxel
           / sum of I0_p over the same pixels
```

This is the count rate per unit monitor, averaged over every contributing pixel, with a Poisson error from the summed
counts. Voxels no pixel reached are 0/0, so NaN, which shows the coverage honestly. If the map has holes, use coarser
bins rather than interpolating.

The [tutorial](../tutorials/rsm.md) compares the two methods on the same synthetic scan.
