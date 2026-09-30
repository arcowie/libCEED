// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-ref.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>

#include "../cuda/ceed-cuda-common.h"

//------------------------------------------------------------------------------
// CUDA preferred MemType
//------------------------------------------------------------------------------
static int CeedGetPreferredMemType_Cuda_Petsc(CeedMemType *mem_type) {
  *mem_type = CEED_MEM_DEVICE;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//
// This backend behaves identically to `/gpu/cuda/ref`, except that it optionally reads additional information
// needed for JiT compilation from a file specified by the user at runtime, rather than requiring it to be fixed at
// compile time. The path to this file is passed as part of the resource string, e.g.:
//
//   /gpu/cuda/petsc:jit_source_file=/path/to/file.h
//
// The contents of the file are included verbatim ahead of every kernel compiled for this `Ceed`, so it can be used to
// supply application specific (e.g. PETSc specific) macros, types, or other definitions needed by user QFunctions at
// JiT time, without requiring a separate compile-time configuration of libCEED. If no `jit_source_file` option is
// given, this backend behaves exactly like `/gpu/cuda/ref`.
//------------------------------------------------------------------------------
static int CeedInit_Cuda_Petsc(const char *resource, Ceed ceed) {
  Ceed_Cuda *data;
  char      *resource_root;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/cuda/petsc"), ceed, CEED_ERROR_BACKEND, "Cuda backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Cuda(ceed, resource));
  CeedCallBackend(CeedSetJitSourceFile_Cuda(ceed, resource, "jit_source_file"));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType", CeedGetPreferredMemType_Cuda_Petsc));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", CeedVectorCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1", CeedBasisCreateH1_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHdiv", CeedBasisCreateHdiv_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHcurl", CeedBasisCreateHcurl_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate", CeedElemRestrictionCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreateAtPoints", CeedElemRestrictionCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate", CeedQFunctionContextCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreateAtPoints", CeedOperatorCreateAtPoints_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda_Petsc(void) { return CeedRegister("/gpu/cuda/petsc", CeedInit_Cuda_Petsc, 40); }

//------------------------------------------------------------------------------
