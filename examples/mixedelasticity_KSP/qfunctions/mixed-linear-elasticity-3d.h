// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for mixed-linear elasticity example using PETSc

#ifndef mixed_linear3d_h
#define mixed_linear3d_h

#include <ceed.h>
#include <math.h>

#include "utils.h"

#ifndef LINEAR_CTX
#define LINEAR_CTX
typedef struct LINEARContext_ *LINEARContext;
struct LINEARContext_ {
  CeedScalar E;
  CeedScalar nu;
};
#endif
// -----------------------------------------------------------------------------
// Strong form:
//  div(sigma)  +  f   = 0   in \Omega
//  div(u)      -  p/k = 0   in \Omega
//
//  where k is bulk modulus, and sigma_ij = p * delta_ij + 2 * mu * ed_ij
//  ed = e - 1/3 trace(e) * I is the deviatoric strain and e = 0.5*(grad(u) + (grad(u))^T )
// in indicial notation
//   mu * ui_jj + (1/3 mu + k) * uj_ji + fi = 0
//   ui_i   - p/k                           = 0
//
// Weak form: Find (u,p) \in VxQ (V=H1, Q=L^2) on \Omega
//  (grad(v), sigma)      = (v, f)
//  (q, div(u)) - (q,p/k) = 0
// We set the true solution in a way that vanishes on the boundary.
// This QFunction setup the rhs and true solution of the above equation
// Inputs:
//   coords     : coordinate of the physical element
//   wdetJ      : updated weight of quadrature
//
// Output:
//   true_soln  : pe and ue
//   rhs        : (v, f) = \int (v^T * f * wdetJ) dX
//                we save: rhs_u = f * wdetJ and rhs_p = 0
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMixedLinearRhs3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *coords = in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*true_soln)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*rhs_u)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1],
  (*rhs_p)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];

  // Context
  LINEARContext    context = (LINEARContext)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;
  const CeedScalar mu      = E / (2. * (1 + nu));
  const CeedScalar kappa   = E / (3. * (1 - 2 * nu));
  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar x = coords[i + 0 * Q], y = coords[i + 1 * Q], z = coords[i + 2 * Q];
    CeedScalar u1 = sin(PI_DOUBLE * x) * sin(PI_DOUBLE * y) * sin(PI_DOUBLE * z), u2 = 2 * u1, u3 = 3 * u1;
    CeedScalar u1_1 = PI_DOUBLE * cos(PI_DOUBLE * x) * sin(PI_DOUBLE * y) * sin(PI_DOUBLE * z),
               u2_2 = 2 * PI_DOUBLE * sin(PI_DOUBLE * x) * cos(PI_DOUBLE * y) * sin(PI_DOUBLE * z),
               u3_3 = 3 * PI_DOUBLE * sin(PI_DOUBLE * x) * sin(PI_DOUBLE * y) * cos(PI_DOUBLE * z);
    // Component 1
    true_soln[0][i] = u1;
    // Component 2
    true_soln[1][i] = u2;
    // Component 3
    true_soln[2][i] = u3;
    // Pressure, p = kappa * div(u)
    true_soln[3][i] = kappa * (u1_1 + u2_2 + u3_3);

    // mu*(u1_11 + u1_22 + u1_33) + (1/3 * mu + kappa)*(u1_11 + u2_21 + u3_31) + f1 = 0
    CeedScalar u1_11 = -PI_DOUBLE * PI_DOUBLE * u1, u1_22 = -PI_DOUBLE * PI_DOUBLE * u1, u1_33 = -PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u2_21 = 2 * PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * cos(PI_DOUBLE * y) * sin(PI_DOUBLE * z);
    CeedScalar u3_31 = 3 * PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * sin(PI_DOUBLE * y) * cos(PI_DOUBLE * z);
    CeedScalar f1    = -mu * (u1_11 + u1_22 + u1_33) - ((1. / 3.) * mu + kappa) * (u1_11 + u2_21 + u3_31);
    // Component 1
    rhs_u[0][i] = q_data[0][i] * f1;
    // mu*(u2_11 + u2_22 + u2_33) + (1/3 * mu + kappa)*(u1_12 + u2_22 + u3_32) + f2 = 0
    CeedScalar u2_11 = -2 * PI_DOUBLE * PI_DOUBLE * u1, u2_22 = -2 * PI_DOUBLE * PI_DOUBLE * u1, u2_33 = -2 * PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u1_12 = PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * cos(PI_DOUBLE * y) * sin(PI_DOUBLE * z);
    CeedScalar u3_32 = 3 * PI_DOUBLE * PI_DOUBLE * sin(PI_DOUBLE * x) * cos(PI_DOUBLE * y) * cos(PI_DOUBLE * z);
    CeedScalar f2    = -mu * (u2_11 + u2_22 + u2_33) - ((1. / 3.) * mu + kappa) * (u1_12 + u2_22 + u3_32);
    // Component 2
    rhs_u[1][i] = q_data[0][i] * f2;
    // mu*(u3_11 + u3_22 + u3_33) + (1/3 * mu + kappa)*(u1_13 + u2_23 + u3_33) + f3 = 0
    CeedScalar u3_11 = -3 * PI_DOUBLE * PI_DOUBLE * u1, u3_22 = -3 * PI_DOUBLE * PI_DOUBLE * u1, u3_33 = -3 * PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u1_13 = PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * sin(PI_DOUBLE * y) * cos(PI_DOUBLE * z);
    CeedScalar u2_23 = 2 * PI_DOUBLE * PI_DOUBLE * sin(PI_DOUBLE * x) * cos(PI_DOUBLE * y) * cos(PI_DOUBLE * z);
    CeedScalar f3    = -mu * (u3_11 + u3_22 + u3_33) - ((1. / 3.) * mu + kappa) * (u1_13 + u2_23 + u3_33);
    // Component 3
    rhs_u[2][i] = q_data[0][i] * f3;
    rhs_p[0][i] = 0.0;
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------
// This QFunction setup the lhs of the above equation
// Inputs:
//   dudX       : derivative of basis with respect to ref element coordinate; du/dX
//   q_data     : updated weight of quadrature and inverse of the Jacobian J; [wdetJ, dXdx]
//   basis_p    : basis for pressure field
//
// Output:
//   dvdX       : (grad(v), sigma) = \int (dv/dX)^T (dX/dx^T * sigma) * wdetJ dX
//                we save:    dvdX = (dX/dx^T * sigma) * wdetJ
//   q          : (q, div(u)) - (q,p/k) = \int q^T [div(u) - p/k] * wdetJ dX
//                we save:            q = [div(u) - p/k] * wdetJ
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMixedLinear3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1],
        (*p)[CEED_Q_VLA]           = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0], (*q)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

  // Context
  LINEARContext    context = (LINEARContext)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;
  const CeedScalar mu      = E / (2. * (1 + nu));
  const CeedScalar kappa   = E / (3. * (1 - 2 * nu));
  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar dudX[3][3] = {
        {ug[0][0][i], ug[0][1][i], ug[0][2][i]},
        {ug[1][0][i], ug[1][1][i], ug[1][2][i]},
        {ug[2][0][i], ug[2][1][i], ug[2][2][i]}
    };
    CeedScalar dXdx_voigt[9];
    for (CeedInt j = 0; j < 9; j++) {
      dXdx_voigt[j] = q_data[j + 1][i];
    }
    CeedScalar dXdx[3][3];
    VoigtUnpackNonSymmetric3(dXdx_voigt, dXdx);
    // Compute grad_u = dX/dx * du/dX
    CeedScalar grad_u[3][3];
    AlphaMatMatMult3(1.0, dXdx, dudX, grad_u);
    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)
    const CeedScalar e[3][3] = {
        {(grad_u[0][0] + grad_u[0][0]) / 2., (grad_u[0][1] + grad_u[1][0]) / 2., (grad_u[0][2] + grad_u[2][0]) / 2.},
        {(grad_u[1][0] + grad_u[0][1]) / 2., (grad_u[1][1] + grad_u[1][1]) / 2., (grad_u[1][2] + grad_u[2][1]) / 2.},
        {(grad_u[2][0] + grad_u[0][2]) / 2., (grad_u[2][1] + grad_u[1][2]) / 2., (grad_u[2][2] + grad_u[2][2]) / 2.}
    };
    CeedScalar e_kk = Trace3(e);
    // Compute Deviatoric Strain : ed (epsilon)
    // ed = e - 1/3 * trace(e) * I
    const CeedScalar ed[3][3] = {
        {e[0][0] - (1. / 3.) * e_kk, e[0][1],                    e[0][2]                   },
        {e[1][0],                    e[1][1] - (1. / 3.) * e_kk, e[1][2]                   },
        {e[2][0],                    e[2][1],                    e[2][2] - (1. / 3.) * e_kk}
    };
    // Compute Sigma = p*delta_ij + 2*mu*ed_ij
    const CeedScalar sigma[3][3] = {
        {p[0][i] + 2. * mu * ed[0][0], 2. * mu * ed[0][1],           2. * mu * ed[0][2]          },
        {2. * mu * ed[1][0],           p[0][i] + 2. * mu * ed[1][1], 2. * mu * ed[1][2]          },
        {2. * mu * ed[2][0],           2. * mu * ed[2][1],           p[0][i] + 2. * mu * ed[2][2]}
    };
    // save output:dX/dx^T * sigma * wdetJ
    AlphaMatTransposeMatMultAtQuadrature3(Q, i, q_data[0][i], dXdx, sigma, dvdX);
    // div(u) = trace(grad(u))
    CeedScalar div_u = Trace3(grad_u);
    // (q, div(u)) - (q,p/k) = q^T * (div(u) - p/k) * wdetJ
    q[0][i] = (div_u - p[0][i] / kappa) * q_data[0][i];
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------
#endif  // mixed_linear3d_h
