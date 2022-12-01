// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Utility functions for setting up POISSON_QUAD2D

#include "../include/register-problem.h"
#include "../include/setup-libceed.h"
#include "../qfunctions/poisson-error2d.h"
#include "../qfunctions/poisson-mass2d.h"
#include "../qfunctions/poisson-rhs2d.h"

// Hdiv_POISSON_MASS2D is registered in cl-option.c
PetscErrorCode Hdiv_POISSON_MASS2D(ProblemData problem_data, void *ctx) {
  PetscFunctionBeginUser;

  // ------------------------------------------------------
  //               SET UP POISSON_QUAD2D
  // ------------------------------------------------------
  problem_data->dim             = 2;
  problem_data->elem_node       = 4;
  problem_data->quadrature_mode = CEED_GAUSS;
  problem_data->setup_rhs       = SetupRhs2D;
  problem_data->setup_rhs_loc   = SetupRhs2D_loc;
  problem_data->residual        = SetupMass2D;
  problem_data->residual_loc    = SetupMass2D_loc;
  problem_data->setup_error     = SetupError2D;
  problem_data->setup_error_loc = SetupError2D_loc;

  PetscFunctionReturn(0);
}
