#ifndef FIXED_POINT_H
#define FIXED_POINT_H

/*
 * Minimal fixed-point abstraction for this project.
 *
 * The current kernels (Conv1_*, Conv2_*, Fc1_*, Fc2_*) use float interfaces.
 * weights.h stores weights using the type 'fixed'.
 *
 * To keep everything consistent with GCC builds, we map:
 *     fixed -> float
 *
 * If you later switch to a true fixed-point type (e.g. ap_fixed for HLS),
 * you can change this typedef only.
 */
typedef float fixed;

#endif
