// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-common.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
// Device information backend init
//------------------------------------------------------------------------------
int CeedInit_Cuda(Ceed ceed, const char *resource) {
  Ceed_Cuda  *data;
  const char *device_spec = strstr(resource, ":device_id=");
  const int   device_id   = (device_spec) ? atoi(device_spec + 11) : -1;
  int         current_device_id;

  CeedCallCuda(ceed, cudaGetDevice(&current_device_id));
  if (device_id >= 0 && current_device_id != device_id) {
    CeedCallCuda(ceed, cudaSetDevice(device_id));
    current_device_id = device_id;
  }

  CeedCallBackend(CeedGetData(ceed, &data));
  data->device_id = current_device_id;
  CeedCallCuda(ceed, cudaGetDeviceProperties(&data->device_prop, current_device_id));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Parse and load a user-provided JiT source file from the resource string, if provided
//
// The resource may contain `:<option_name>=/path/to/file`, where the file is loaded and its contents are included in
// every subsequent JiT compilation for this `Ceed`. If the option is not present in the resource, this is a no-op.
//------------------------------------------------------------------------------
int CeedSetJitSourceFile_Cuda(Ceed ceed, const char *resource, const char *option_name) {
  Ceed_Cuda  *data;
  char        option[CEED_MAX_RESOURCE_LEN];
  const char *path_start;
  const char *path_end;
  size_t      path_len;

  snprintf(option, sizeof(option), ":%s=", option_name);
  path_start = strstr(resource, option);
  if (!path_start) return CEED_ERROR_SUCCESS;
  path_start += strlen(option);
  path_end = strchr(path_start, ':');
  path_len = path_end ? (size_t)(path_end - path_start) : strlen(path_start);
  CeedCheck(path_len > 0, ceed, CEED_ERROR_BACKEND, "Resource %s must specify a non-empty JiT source file path with '%s/path/to/file'", resource,
            option);

  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(path_len + 1, &data->jit_source_path));
  memcpy(data->jit_source_path, path_start, path_len);
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, data->jit_source_path, &data->jit_source));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend destroy
//------------------------------------------------------------------------------
int CeedDestroy_Cuda(Ceed ceed) {
  Ceed_Cuda *data;

  CeedCallBackend(CeedGetData(ceed, &data));
  if (data->cublas_handle) CeedCallCublas(ceed, cublasDestroy(data->cublas_handle));
  CeedCallBackend(CeedFree(&data->jit_source_path));
  CeedCallBackend(CeedFree(&data->jit_source));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Memory transfer utilities
//------------------------------------------------------------------------------
static inline int CeedSetDeviceGenericArray_Cuda(Ceed ceed, const void *source_array, CeedCopyMode copy_mode, size_t size_unit, CeedSize num_values,
                                                 void *target_array_owned, void *target_array_borrowed, void *target_array) {
  switch (copy_mode) {
    case CEED_COPY_VALUES:
      if (!*(void **)target_array_owned) CeedCallCuda(ceed, cudaMalloc(target_array_owned, size_unit * num_values));
      if (source_array) CeedCallCuda(ceed, cudaMemcpy(*(void **)target_array_owned, source_array, size_unit * num_values, cudaMemcpyDeviceToDevice));
      *(void **)target_array_borrowed = NULL;
      *(void **)target_array          = *(void **)target_array_owned;
      break;
    case CEED_OWN_POINTER:
      CeedCallCuda(ceed, cudaFree(*(void **)target_array_owned));
      *(void **)target_array_owned    = (void *)source_array;
      *(void **)target_array_borrowed = NULL;
      *(void **)target_array          = *(void **)target_array_owned;
      break;
    case CEED_USE_POINTER:
      CeedCallCuda(ceed, cudaFree(*(void **)target_array_owned));
      *(void **)target_array_owned    = NULL;
      *(void **)target_array_borrowed = (void *)source_array;
      *(void **)target_array          = *(void **)target_array_borrowed;
  }
  return CEED_ERROR_SUCCESS;
}

int CeedSetDeviceBoolArray_Cuda(Ceed ceed, const bool *source_array, CeedCopyMode copy_mode, CeedSize num_values, const bool **target_array_owned,
                                const bool **target_array_borrowed, const bool **target_array) {
  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(bool), num_values, target_array_owned, target_array_borrowed,
                                                 target_array));
  return CEED_ERROR_SUCCESS;
}

int CeedSetDeviceCeedInt8Array_Cuda(Ceed ceed, const CeedInt8 *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                    const CeedInt8 **target_array_owned, const CeedInt8 **target_array_borrowed, const CeedInt8 **target_array) {
  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(CeedInt8), num_values, target_array_owned,
                                                 target_array_borrowed, target_array));
  return CEED_ERROR_SUCCESS;
}

int CeedSetDeviceCeedIntArray_Cuda(Ceed ceed, const CeedInt *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                   const CeedInt **target_array_owned, const CeedInt **target_array_borrowed, const CeedInt **target_array) {
  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(CeedInt), num_values, target_array_owned,
                                                 target_array_borrowed, target_array));
  return CEED_ERROR_SUCCESS;
}

int CeedSetDeviceCeedScalarArray_Cuda(Ceed ceed, const CeedScalar *source_array, CeedCopyMode copy_mode, CeedSize num_values,
                                      const CeedScalar **target_array_owned, const CeedScalar **target_array_borrowed,
                                      const CeedScalar **target_array) {
  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(CeedScalar), num_values, target_array_owned,
                                                 target_array_borrowed, target_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
