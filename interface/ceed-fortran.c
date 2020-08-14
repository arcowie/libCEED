// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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

// Fortran interface
#include <ceed.h>
#include <ceed-impl.h>
#include <ceed-backend.h>
#include <ceed-fortran-name.h>
#include <stdlib.h>
#include <string.h>

#define FORTRAN_REQUEST_IMMEDIATE -1
#define FORTRAN_REQUEST_ORDERED -2
#define FORTRAN_NULL -3
#define FORTRAN_STRIDES_BACKEND -4
#define FORTRAN_VECTOR_ACTIVE -5
#define FORTRAN_VECTOR_NONE -6
#define FORTRAN_ELEMRESTRICTION_NONE -7
#define FORTRAN_BASIS_COLLOCATED -8
#define FORTRAN_QFUNCTION_NONE -9

static Ceed *Ceed_dict = NULL;
static int Ceed_count = 0;
static int Ceed_n = 0;
static int Ceed_count_max = 0;

// This test should actually be for the gfortran version, but we don't currently
// have a configure system to determine that (TODO).  At present, this will use
// the smaller integer when run with clang+gfortran=8, for example.  (That is
// sketchy, but will likely work for users that don't have huge character
// strings.)
#if __GNUC__ >= 8
typedef size_t fortran_charlen_t;
#else
typedef int fortran_charlen_t;
#endif

#define Splice(a, b) a ## b

// Fortran strings are generally unterminated and the length is passed as an
// extra argument after all the normal arguments.  Some compilers (I only know
// of Windows) place the length argument immediately after the string parameter
// (TODO).
//
// We can't just NULL-terminate the string in-place because that could overwrite
// other strings or attempt to write to read-only memory.  This macro allocates
// a string to hold the null-terminated version of the string that C expects.
#define FIX_STRING(stringname)                                          \
  char Splice(stringname, _c)[1024];                                    \
  if (Splice(stringname, _len) > 1023)                                  \
    CeedError(NULL, 1, "Fortran string length too long %zd", (size_t)Splice(stringname, _len)); \
  strncpy(Splice(stringname, _c), stringname, Splice(stringname, _len)); \
  Splice(stringname, _c)[Splice(stringname, _len)] = 0;                 \

// -----------------------------------------------------------------------------
// Ceed
// -----------------------------------------------------------------------------
#define fCeedInit FORTRAN_NAME(ceedinit,CEEDINIT)
void fCeedInit(const char *resource, int *ceed, int *err,
               fortran_charlen_t resource_len) {
  FIX_STRING(resource);
  if (Ceed_count == Ceed_count_max) {
    Ceed_count_max += Ceed_count_max/2 + 1;
    CeedRealloc(Ceed_count_max, &Ceed_dict);
  }

  Ceed *ceed_ = &Ceed_dict[Ceed_count];
  *err = CeedInit(resource_c, ceed_);

  if (*err == 0) {
    *ceed = Ceed_count++;
    Ceed_n++;
  }
}

#define fCeedIsDeterministic \
    FORTRAN_NAME(ceedisdeterministic,CEEDISDETERMINISTIC)
void fCeedIsDeterministic(int *ceed, int *isDeterministic, int *err) {
  *err = CeedIsDeterministic(Ceed_dict[*ceed], (bool *)isDeterministic);
}

#define fCeedGetPreferredMemType \
    FORTRAN_NAME(ceedgetpreferredmemtype,CEEDGETPREFERREDMEMTYPE)
void fCeedGetPreferredMemType(int *ceed, int *type, int *err) {
  *err = CeedGetPreferredMemType(Ceed_dict[*ceed], (CeedMemType *)type);
}

#define fCeedView FORTRAN_NAME(ceedview,CEEDVIEW)
void fCeedView(int *ceed, int *err) {
  *err = CeedView(Ceed_dict[*ceed], stdout);
}

#define fCeedDestroy FORTRAN_NAME(ceeddestroy,CEEDDESTROY)
void fCeedDestroy(int *ceed, int *err) {
  if (*ceed == FORTRAN_NULL) return;
  *err = CeedDestroy(&Ceed_dict[*ceed]);

  if (*err == 0) {
    *ceed = FORTRAN_NULL;
    Ceed_n--;
    if (Ceed_n == 0) {
      CeedFree(&Ceed_dict);
      Ceed_count = 0;
      Ceed_count_max = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedVector
// -----------------------------------------------------------------------------
static CeedVector *CeedVector_dict = NULL;
static int CeedVector_count = 0;
static int CeedVector_n = 0;
static int CeedVector_count_max = 0;

#define fCeedVectorCreate FORTRAN_NAME(ceedvectorcreate,CEEDVECTORCREATE)
void fCeedVectorCreate(int *ceed, int *length, int *vec, int *err) {
  if (CeedVector_count == CeedVector_count_max) {
    CeedVector_count_max += CeedVector_count_max/2 + 1;
    CeedRealloc(CeedVector_count_max, &CeedVector_dict);
  }

  CeedVector *vec_ = &CeedVector_dict[CeedVector_count];
  *err = CeedVectorCreate(Ceed_dict[*ceed], *length, vec_);

  if (*err == 0) {
    *vec = CeedVector_count++;
    CeedVector_n++;
  }
}

#define fCeedVectorSetArray FORTRAN_NAME(ceedvectorsetarray,CEEDVECTORSETARRAY)
void fCeedVectorSetArray(int *vec, int *memtype, int *copymode,
                         CeedScalar *array, int64_t *offset, int *err) {
  *err = CeedVectorSetArray(CeedVector_dict[*vec], (CeedMemType)*memtype,
                            (CeedCopyMode)*copymode,
                            (CeedScalar *)(array + *offset));
}

#define fCeedVectorTakeArray FORTRAN_NAME(ceedvectortakearray,CEEDVECTORTAKEARRAY)
void fCeedVectorTakeArray(int *vec, int *memtype, CeedScalar *array,
                          int64_t *offset, int *err) {
  CeedScalar *b;
  CeedVector vec_ = CeedVector_dict[*vec];
  *err = CeedVectorTakeArray(vec_, (CeedMemType)*memtype, &b);
  *offset = b - array;
}

#define fCeedVectorSyncArray FORTRAN_NAME(ceedvectorsyncarray,CEEDVECTORSYNCARRAY)
void fCeedVectorSyncArray(int *vec, int *memtype, int *err) {
  *err = CeedVectorSyncArray(CeedVector_dict[*vec], (CeedMemType)*memtype);
}

#define fCeedVectorSetValue FORTRAN_NAME(ceedvectorsetvalue,CEEDVECTORSETVALUE)
void fCeedVectorSetValue(int *vec, CeedScalar *value, int *err) {
  *err = CeedVectorSetValue(CeedVector_dict[*vec], *value);
}

#define fCeedVectorGetArray FORTRAN_NAME(ceedvectorgetarray,CEEDVECTORGETARRAY)
void fCeedVectorGetArray(int *vec, int *memtype, CeedScalar *array,
                         int64_t *offset, int *err) {
  CeedScalar *b;
  CeedVector vec_ = CeedVector_dict[*vec];
  *err = CeedVectorGetArray(vec_, (CeedMemType)*memtype, &b);
  *offset = b - array;
}

#define fCeedVectorGetArrayRead \
    FORTRAN_NAME(ceedvectorgetarrayread,CEEDVECTORGETARRAYREAD)
void fCeedVectorGetArrayRead(int *vec, int *memtype, CeedScalar *array,
                             int64_t *offset, int *err) {
  const CeedScalar *b;
  CeedVector vec_ = CeedVector_dict[*vec];
  *err = CeedVectorGetArrayRead(vec_, (CeedMemType)*memtype, &b);
  *offset = b - array;
}

#define fCeedVectorRestoreArray \
    FORTRAN_NAME(ceedvectorrestorearray,CEEDVECTORRESTOREARRAY)
void fCeedVectorRestoreArray(int *vec, CeedScalar *array,
                             int64_t *offset, int *err) {
  *err = CeedVectorRestoreArray(CeedVector_dict[*vec], &array);
  *offset = 0;
}

#define fCeedVectorRestoreArrayRead \
    FORTRAN_NAME(ceedvectorrestorearrayread,CEEDVECTORRESTOREARRAYREAD)
void fCeedVectorRestoreArrayRead(int *vec, const CeedScalar *array,
                                 int64_t *offset, int *err) {
  *err = CeedVectorRestoreArrayRead(CeedVector_dict[*vec], &array);
  *offset = 0;
}

#define fCeedVectorNorm \
    FORTRAN_NAME(ceedvectornorm,CEEDVECTORNORM)
void fCeedVectorNorm(int *vec, int *type, CeedScalar *norm, int *err) {
  *err = CeedVectorNorm(CeedVector_dict[*vec], (CeedNormType)*type, norm);
}

#define fCeedVectorReciprocal \
    FORTRAN_NAME(ceedvectorreciprocal,CEEDVECTORRECIPROCAL)
void fCeedVectorReciprocal(int *vec, int *err) {
  *err = CeedVectorReciprocal(CeedVector_dict[*vec]);
}

#define fCeedVectorView FORTRAN_NAME(ceedvectorview,CEEDVECTORVIEW)
void fCeedVectorView(int *vec, int *err) {
  *err = CeedVectorView(CeedVector_dict[*vec], "%12.8f", stdout);
}

#define fCeedVectorDestroy FORTRAN_NAME(ceedvectordestroy,CEEDVECTORDESTROY)
void fCeedVectorDestroy(int *vec, int *err) {
  if (*vec == FORTRAN_NULL) return;
  *err = CeedVectorDestroy(&CeedVector_dict[*vec]);

  if (*err == 0) {
    *vec = FORTRAN_NULL;
    CeedVector_n--;
    if (CeedVector_n == 0) {
      CeedFree(&CeedVector_dict);
      CeedVector_count = 0;
      CeedVector_count_max = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedElemRestriction
// -----------------------------------------------------------------------------
static CeedElemRestriction *CeedElemRestriction_dict = NULL;
static int CeedElemRestriction_count = 0;
static int CeedElemRestriction_n = 0;
static int CeedElemRestriction_count_max = 0;

#define fCeedElemRestrictionCreate \
    FORTRAN_NAME(ceedelemrestrictioncreate, CEEDELEMRESTRICTIONCREATE)
void fCeedElemRestrictionCreate(int *ceed, int *nelements, int *esize,
                                int *ncomp, int *compstride, int *lsize,
                                int *memtype, int *copymode, const int *offsets,
                                int *elemrestriction, int *err) {
  if (CeedElemRestriction_count == CeedElemRestriction_count_max) {
    CeedElemRestriction_count_max += CeedElemRestriction_count_max/2 + 1;
    CeedRealloc(CeedElemRestriction_count_max, &CeedElemRestriction_dict);
  }

  const int *offsets_ = offsets;

  CeedElemRestriction *elemrestriction_ =
    &CeedElemRestriction_dict[CeedElemRestriction_count];
  *err = CeedElemRestrictionCreate(Ceed_dict[*ceed], *nelements, *esize,
                                   *ncomp, *compstride, *lsize,
                                   (CeedMemType)*memtype,
                                   (CeedCopyMode)*copymode, offsets_,
                                   elemrestriction_);

  if (*err == 0) {
    *elemrestriction = CeedElemRestriction_count++;
    CeedElemRestriction_n++;
  }
}

#define fCeedElemRestrictionCreateStrided \
    FORTRAN_NAME(ceedelemrestrictioncreatestrided, CEEDELEMRESTRICTIONCREATESTRIDED)
void fCeedElemRestrictionCreateStrided(int *ceed, int *nelements, int *esize,
                                       int *ncomp, int *lsize, int *strides,
                                       int *elemrestriction, int *err) {
  if (CeedElemRestriction_count == CeedElemRestriction_count_max) {
    CeedElemRestriction_count_max += CeedElemRestriction_count_max/2 + 1;
    CeedRealloc(CeedElemRestriction_count_max, &CeedElemRestriction_dict);
  }

  CeedElemRestriction *elemrestriction_ =
    &CeedElemRestriction_dict[CeedElemRestriction_count];
  *err = CeedElemRestrictionCreateStrided(Ceed_dict[*ceed], *nelements, *esize,
                                          *ncomp, *lsize,
                                          *strides == FORTRAN_STRIDES_BACKEND ?
                                          CEED_STRIDES_BACKEND : strides,
                                          elemrestriction_);
  if (*err == 0) {
    *elemrestriction = CeedElemRestriction_count++;
    CeedElemRestriction_n++;
  }
}

#define fCeedElemRestrictionCreateBlocked \
    FORTRAN_NAME(ceedelemrestrictioncreateblocked,CEEDELEMRESTRICTIONCREATEBLOCKED)
void fCeedElemRestrictionCreateBlocked(int *ceed, int *nelements, int *esize,
                                       int *blocksize, int *ncomp,
                                       int *compstride, int *lsize,
                                       int *mtype, int *cmode,
                                       int *blkindices, int *elemrestriction,
                                       int *err) {

  if (CeedElemRestriction_count == CeedElemRestriction_count_max) {
    CeedElemRestriction_count_max += CeedElemRestriction_count_max/2 + 1;
    CeedRealloc(CeedElemRestriction_count_max, &CeedElemRestriction_dict);
  }

  CeedElemRestriction *elemrestriction_ =
    &CeedElemRestriction_dict[CeedElemRestriction_count];
  *err = CeedElemRestrictionCreateBlocked(Ceed_dict[*ceed],
                                          *nelements, *esize, *blocksize,
                                          *ncomp, *compstride, *lsize,
                                          (CeedMemType)*mtype,
                                          (CeedCopyMode)*cmode, blkindices,
                                          elemrestriction_);

  if (*err == 0) {
    *elemrestriction = CeedElemRestriction_count++;
    CeedElemRestriction_n++;
  }
}

#define fCeedElemRestrictionCreateBlockedStrided \
    FORTRAN_NAME(ceedelemrestrictioncreateblockedstrided, CEEDELEMRESTRICTIONCREATEBLOCKEDSTRIDED)
void fCeedElemRestrictionCreateBlockedStrided(int *ceed, int *nelements,
    int *esize, int *blksize, int *ncomp, int *lsize, int *strides,
    int *elemrestriction, int *err) {
  if (CeedElemRestriction_count == CeedElemRestriction_count_max) {
    CeedElemRestriction_count_max += CeedElemRestriction_count_max/2 + 1;
    CeedRealloc(CeedElemRestriction_count_max, &CeedElemRestriction_dict);
  }

  CeedElemRestriction *elemrestriction_ =
    &CeedElemRestriction_dict[CeedElemRestriction_count];
  *err = CeedElemRestrictionCreateBlockedStrided(Ceed_dict[*ceed], *nelements,
         *esize, *blksize, *ncomp, *lsize, strides, elemrestriction_);
  if (*err == 0) {
    *elemrestriction = CeedElemRestriction_count++;
    CeedElemRestriction_n++;
  }
}

static CeedRequest *CeedRequest_dict = NULL;
static int CeedRequest_count = 0;
static int CeedRequest_n = 0;
static int CeedRequest_count_max = 0;

#define fCeedElemRestrictionApply \
    FORTRAN_NAME(ceedelemrestrictionapply,CEEDELEMRESTRICTIONAPPLY)
void fCeedElemRestrictionApply(int *elemr, int *tmode, int *uvec, int *ruvec,
                               int *rqst, int *err) {
  int createRequest = 1;
  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*rqst == FORTRAN_REQUEST_IMMEDIATE || *rqst == FORTRAN_REQUEST_ORDERED)
    createRequest = 0;

  if (createRequest && CeedRequest_count == CeedRequest_count_max) {
    CeedRequest_count_max += CeedRequest_count_max/2 + 1;
    CeedRealloc(CeedRequest_count_max, &CeedRequest_dict);
  }

  CeedRequest *rqst_;
  if      (*rqst == FORTRAN_REQUEST_IMMEDIATE) rqst_ = CEED_REQUEST_IMMEDIATE;
  else if (*rqst == FORTRAN_REQUEST_ORDERED  ) rqst_ = CEED_REQUEST_ORDERED;
  else rqst_ = &CeedRequest_dict[CeedRequest_count];

  *err = CeedElemRestrictionApply(CeedElemRestriction_dict[*elemr],
                                  (CeedTransposeMode)*tmode,
                                  CeedVector_dict[*uvec],
                                  CeedVector_dict[*ruvec], rqst_);

  if (*err == 0 && createRequest) {
    *rqst = CeedRequest_count++;
    CeedRequest_n++;
  }
}

#define fCeedElemRestrictionApplyBlock \
    FORTRAN_NAME(ceedelemrestrictionapplyblock,CEEDELEMRESTRICTIONAPPLYBLOCK)
void fCeedElemRestrictionApplyBlock(int *elemr, int *block, int *tmode,
                                    int *uvec, int *ruvec, int *rqst, int *err) {
  int createRequest = 1;
  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*rqst == FORTRAN_REQUEST_IMMEDIATE || *rqst == FORTRAN_REQUEST_ORDERED)
    createRequest = 0;

  if (createRequest && CeedRequest_count == CeedRequest_count_max) {
    CeedRequest_count_max += CeedRequest_count_max/2 + 1;
    CeedRealloc(CeedRequest_count_max, &CeedRequest_dict);
  }

  CeedRequest *rqst_;
  if      (*rqst == FORTRAN_REQUEST_IMMEDIATE) rqst_ = CEED_REQUEST_IMMEDIATE;
  else if (*rqst == FORTRAN_REQUEST_ORDERED  ) rqst_ = CEED_REQUEST_ORDERED;
  else rqst_ = &CeedRequest_dict[CeedRequest_count];

  *err = CeedElemRestrictionApplyBlock(CeedElemRestriction_dict[*elemr], *block,
                                       (CeedTransposeMode)*tmode, CeedVector_dict[*uvec],
                                       CeedVector_dict[*ruvec], rqst_);

  if (*err == 0 && createRequest) {
    *rqst = CeedRequest_count++;
    CeedRequest_n++;
  }
}

#define fCeedElemRestrictionGetMultiplicity \
    FORTRAN_NAME(ceedelemrestrictiongetmultiplicity,CEEDELEMRESTRICTIONGETMULTIPLICITY)
void fCeedElemRestrictionGetMultiplicity(int *elemr, int *mult, int *err) {
  *err = CeedElemRestrictionGetMultiplicity(CeedElemRestriction_dict[*elemr],
         CeedVector_dict[*mult]);
}

#define fCeedElemRestrictionGetELayout \
    FORTRAN_NAME(ceedelemrestrictiongetelayout,CEEDELEMRESTRICTIONGETELAYOUT)
void fCeedElemRestrictionGetELayout(int *elemr, int *layout, int *err) {
  CeedInt layout_c[3];
  *err = CeedElemRestrictionGetELayout(CeedElemRestriction_dict[*elemr],
                                       &layout_c);
  for (int i=0; i<3; i++)
    layout[i] = layout_c[i];
}

#define fCeedElemRestrictionView \
    FORTRAN_NAME(ceedelemrestrictionview,CEEDELEMRESTRICTIONVIEW)
void fCeedElemRestrictionView(int *elemr, int *err) {
  *err = CeedElemRestrictionView(CeedElemRestriction_dict[*elemr], stdout);
}

#define fCeedRequestWait FORTRAN_NAME(ceedrequestwait, CEEDREQUESTWAIT)
void fCeedRequestWait(int *rqst, int *err) {
  // TODO Uncomment this once CeedRequestWait is implemented
  //*err = CeedRequestWait(&CeedRequest_dict[*rqst]);

  if (*err == 0) {
    CeedRequest_n--;
    if (CeedRequest_n == 0) {
      CeedFree(&CeedRequest_dict);
      CeedRequest_count = 0;
      CeedRequest_count_max = 0;
    }
  }
}

#define fCeedElemRestrictionDestroy \
    FORTRAN_NAME(ceedelemrestrictiondestroy,CEEDELEMRESTRICTIONDESTROY)
void fCeedElemRestrictionDestroy(int *elem, int *err) {
  if (*elem == FORTRAN_NULL) return;
  *err = CeedElemRestrictionDestroy(&CeedElemRestriction_dict[*elem]);

  if (*err == 0) {
    *elem = FORTRAN_NULL;
    CeedElemRestriction_n--;
    if (CeedElemRestriction_n == 0) {
      CeedFree(&CeedElemRestriction_dict);
      CeedElemRestriction_count = 0;
      CeedElemRestriction_count_max = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedBasis
// -----------------------------------------------------------------------------
static CeedBasis *CeedBasis_dict = NULL;
static int CeedBasis_count = 0;
static int CeedBasis_n = 0;
static int CeedBasis_count_max = 0;

#define fCeedBasisCreateTensorH1Lagrange \
    FORTRAN_NAME(ceedbasiscreatetensorh1lagrange, CEEDBASISCREATETENSORH1LAGRANGE)
void fCeedBasisCreateTensorH1Lagrange(int *ceed, int *dim,
                                      int *ncomp, int *P, int *Q, int *quadmode,
                                      int *basis, int *err) {
  if (CeedBasis_count == CeedBasis_count_max) {
    CeedBasis_count_max += CeedBasis_count_max/2 + 1;
    CeedRealloc(CeedBasis_count_max, &CeedBasis_dict);
  }

  *err = CeedBasisCreateTensorH1Lagrange(Ceed_dict[*ceed], *dim, *ncomp, *P, *Q,
                                         (CeedQuadMode)*quadmode,
                                         &CeedBasis_dict[CeedBasis_count]);

  if (*err == 0) {
    *basis = CeedBasis_count++;
    CeedBasis_n++;
  }
}

#define fCeedBasisCreateTensorH1 \
    FORTRAN_NAME(ceedbasiscreatetensorh1, CEEDBASISCREATETENSORH1)
void fCeedBasisCreateTensorH1(int *ceed, int *dim, int *ncomp, int *P1d,
                              int *Q1d, const CeedScalar *interp1d,
                              const CeedScalar *grad1d,
                              const CeedScalar *qref1d,
                              const CeedScalar *qweight1d, int *basis,
                              int *err) {
  if (CeedBasis_count == CeedBasis_count_max) {
    CeedBasis_count_max += CeedBasis_count_max/2 + 1;
    CeedRealloc(CeedBasis_count_max, &CeedBasis_dict);
  }

  *err = CeedBasisCreateTensorH1(Ceed_dict[*ceed], *dim, *ncomp, *P1d, *Q1d,
                                 interp1d, grad1d, qref1d, qweight1d,
                                 &CeedBasis_dict[CeedBasis_count]);

  if (*err == 0) {
    *basis = CeedBasis_count++;
    CeedBasis_n++;
  }
}

#define fCeedBasisCreateH1 \
    FORTRAN_NAME(ceedbasiscreateh1, CEEDBASISCREATEH1)
void fCeedBasisCreateH1(int *ceed, int *topo, int *ncomp, int *nnodes,
                        int *nqpts, const CeedScalar *interp,
                        const CeedScalar *grad, const CeedScalar *qref,
                        const CeedScalar *qweight, int *basis, int *err) {
  if (CeedBasis_count == CeedBasis_count_max) {
    CeedBasis_count_max += CeedBasis_count_max/2 + 1;
    CeedRealloc(CeedBasis_count_max, &CeedBasis_dict);
  }

  *err = CeedBasisCreateH1(Ceed_dict[*ceed], (CeedElemTopology)*topo, *ncomp,
                           *nnodes, *nqpts, interp, grad, qref, qweight,
                           &CeedBasis_dict[CeedBasis_count]);

  if (*err == 0) {
    *basis = CeedBasis_count++;
    CeedBasis_n++;
  }
}

#define fCeedBasisView FORTRAN_NAME(ceedbasisview, CEEDBASISVIEW)
void fCeedBasisView(int *basis, int *err) {
  *err = CeedBasisView(CeedBasis_dict[*basis], stdout);
}

#define fCeedQRFactorization \
    FORTRAN_NAME(ceedqrfactorization, CEEDQRFACTORIZATION)
void fCeedQRFactorization(int *ceed, CeedScalar *mat, CeedScalar *tau, int *m,
                          int *n, int *err) {
  *err = CeedQRFactorization(Ceed_dict[*ceed], mat, tau, *m, *n);
}

#define fCeedSymmetricSchurDecomposition \
    FORTRAN_NAME(ceedsymmetricschurdecomposition, CEEDSYMMETRICSCHURDECOMPOSITION)
void fCeedSymmetricSchurDecomposition(int *ceed, CeedScalar *mat,
                                      CeedScalar *lambda, int *n, int *err) {
  *err = CeedSymmetricSchurDecomposition(Ceed_dict[*ceed], mat, lambda, *n);
}

#define fCeedSimultaneousDiagonalization \
    FORTRAN_NAME(ceedsimultaneousdiagonalization, CEEDSIMULTANEOUSDIAGONALIZATION)
void fCeedSimultaneousDiagonalization(int *ceed, CeedScalar *matA,
                                      CeedScalar *matB, CeedScalar *x,
                                      CeedScalar *lambda, int *n, int *err) {
  *err = CeedSimultaneousDiagonalization(Ceed_dict[*ceed], matA, matB, x,
                                         lambda, *n);
}

#define fCeedBasisGetCollocatedGrad \
    FORTRAN_NAME(ceedbasisgetcollocatedgrad, CEEDBASISGETCOLLOCATEDGRAD)
void fCeedBasisGetCollocatedGrad(int *basis, CeedScalar *colograd1d,
                                 int *err) {
  *err = CeedBasisGetCollocatedGrad(CeedBasis_dict[*basis], colograd1d);
}

#define fCeedBasisApply FORTRAN_NAME(ceedbasisapply, CEEDBASISAPPLY)
void fCeedBasisApply(int *basis, int *nelem, int *tmode, int *emode,
                     int *u, int *v, int *err) {
  *err = CeedBasisApply(CeedBasis_dict[*basis], *nelem, (CeedTransposeMode)*tmode,
                        (CeedEvalMode)*emode,
                        *u == FORTRAN_VECTOR_NONE ? CEED_VECTOR_NONE : CeedVector_dict[*u],
                        CeedVector_dict[*v]);
}

#define fCeedBasisGetNumNodes \
    FORTRAN_NAME(ceedbasisgetnumnodes, CEEDBASISGETNUMNODES)
void fCeedBasisGetNumNodes(int *basis, int *P, int *err) {
  *err = CeedBasisGetNumNodes(CeedBasis_dict[*basis], P);
}

#define fCeedBasisGetNumQuadraturePoints \
    FORTRAN_NAME(ceedbasisgetnumquadraturepoints, CEEDBASISGETNUMQUADRATUREPOINTS)
void fCeedBasisGetNumQuadraturePoints(int *basis, int *Q, int *err) {
  *err = CeedBasisGetNumQuadraturePoints(CeedBasis_dict[*basis], Q);
}

#define fCeedBasisGetInterp1D \
    FORTRAN_NAME(ceedbasisgetinterp1d, CEEDBASISGETINTERP1D)
void fCeedBasisGetInterp1D(int *basis, CeedScalar *interp1d, int64_t *offset,
                           int *err) {
  const CeedScalar *interp1d_;
  CeedBasis basis_ = CeedBasis_dict[*basis];
  *err = CeedBasisGetInterp1D(basis_, &interp1d_);
  *offset = interp1d_ - interp1d;
}

#define fCeedBasisDestroy FORTRAN_NAME(ceedbasisdestroy,CEEDBASISDESTROY)
void fCeedBasisDestroy(int *basis, int *err) {
  if (*basis == FORTRAN_NULL) return;
  *err = CeedBasisDestroy(&CeedBasis_dict[*basis]);

  if (*err == 0) {
    *basis = FORTRAN_NULL;
    CeedBasis_n--;
    if (CeedBasis_n == 0) {
      CeedFree(&CeedBasis_dict);
      CeedBasis_count = 0;
      CeedBasis_count_max = 0;
    }
  }
}

#define fCeedGaussQuadrature FORTRAN_NAME(ceedgaussquadrature, CEEDGAUSSQUADRATURE)
void fCeedGaussQuadrature(int *Q, CeedScalar *qref1d, CeedScalar *qweight1d,
                          int *err) {
  *err = CeedGaussQuadrature(*Q, qref1d, qweight1d);
}

#define fCeedLobattoQuadrature \
    FORTRAN_NAME(ceedlobattoquadrature, CEEDLOBATTOQUADRATURE)
void fCeedLobattoQuadrature(int *Q, CeedScalar *qref1d, CeedScalar *qweight1d,
                            int *err) {
  *err = CeedLobattoQuadrature(*Q, qref1d, qweight1d);
}

// -----------------------------------------------------------------------------
// CeedQFunctionContext
// -----------------------------------------------------------------------------
static CeedQFunctionContext *CeedQFunctionContext_dict = NULL;
static int CeedQFunctionContext_count = 0;
static int CeedQFunctionContext_n = 0;
static int CeedQFunctionContext_count_max = 0;

#define fCeedQFunctionContextCreate \
    FORTRAN_NAME(ceedqfunctioncontextcreate,CEEDQFUNCTIONCONTEXTCREATE)
void fCeedQFunctionContextCreate(int *ceed, int *ctx, int *err) {
  if (CeedQFunctionContext_count == CeedQFunctionContext_count_max) {
    CeedQFunctionContext_count_max += CeedQFunctionContext_count_max/2 + 1;
    CeedRealloc(CeedQFunctionContext_count_max, &CeedQFunctionContext_dict);
  }

  CeedQFunctionContext *ctx_ =
    &CeedQFunctionContext_dict[CeedQFunctionContext_count];

  *err = CeedQFunctionContextCreate(Ceed_dict[*ceed], ctx_);
  if (*err) return;
  *ctx = CeedQFunctionContext_count++;
  CeedQFunctionContext_n++;
}

#define fCeedQFunctionContextSetData \
    FORTRAN_NAME(ceedqfunctioncontextsetdata,CEEDQFUNCTIONCONTEXTSETDATA)
void fCeedQFunctionContextSetData(int *ctx, int *memtype, int *copymode,
                                  CeedInt *n,
                                  CeedScalar *data, int64_t *offset, int *err) {
  size_t ctxsize = ((size_t) *n)*sizeof(CeedScalar);
  *err = CeedQFunctionContextSetData(CeedQFunctionContext_dict[*ctx],
                                     (CeedMemType)*memtype,
                                     (CeedCopyMode)*copymode, ctxsize,
                                     data + *offset);
}

#define fCeedQFunctionContextGetData \
    FORTRAN_NAME(ceedqfunctioncontextgetdata,CEEDQFUNCTIONCONTEXTGETDATA)
void fCeedQFunctionContextGetData(int *ctx, int *memtype, CeedScalar *data,
                                  int64_t *offset, int *err) {
  CeedScalar *b;
  CeedQFunctionContext ctx_ = CeedQFunctionContext_dict[*ctx];
  *err = CeedQFunctionContextGetData(ctx_, (CeedMemType)*memtype, &b);
  *offset = b - data;
}

#define fCeedQFunctionContextRestoreData \
    FORTRAN_NAME(ceedqfunctioncontextrestoredata,CEEDQFUNCTIONCONTEXTRESTOREDATA)
void fCeedQFunctionContextRestoreData(int *ctx, CeedScalar *data,
                                      int64_t *offset, int *err) {
  *err = CeedQFunctionContextRestoreData(CeedQFunctionContext_dict[*ctx],
                                         (void **)&data);
  *offset = 0;
}

#define fCeedQFunctionContextView \
    FORTRAN_NAME(ceedqfunctioncontextview,CEEDQFUNCTIONCONTEXTVIEW)
void fCeedQFunctionContextView(int *ctx, int *err) {
  *err = CeedQFunctionContextView(CeedQFunctionContext_dict[*ctx], stdout);
}

#define fCeedQFunctionContextDestroy \
    FORTRAN_NAME(ceedqfunctioncontextdestroy,CEEDQFUNCTIONCONTEXTDESTROY)
void fCeedQFunctionContextDestroy(int *ctx, int *err) {
  if (*ctx == FORTRAN_NULL) return;
  *err = CeedQFunctionContextDestroy(&CeedQFunctionContext_dict[*ctx]);

  if (*err == 0) {
    *ctx = FORTRAN_NULL;
    CeedQFunctionContext_n--;
    if (CeedQFunctionContext_n == 0) {
      CeedFree(&CeedQFunctionContext_dict);
      CeedQFunctionContext_count = 0;
      CeedQFunctionContext_count_max = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedQFunction
// -----------------------------------------------------------------------------
static CeedQFunction *CeedQFunction_dict = NULL;
static int CeedQFunction_count = 0;
static int CeedQFunction_n = 0;
static int CeedQFunction_count_max = 0;

static int CeedQFunctionFortranStub(void *ctx, int nq,
                                    const CeedScalar *const *u,
                                    CeedScalar *const *v) {
  CeedFortranContext fctx = ctx;
  CeedQFunctionContext innerctx = fctx->innerctx;
  int ierr;

  CeedScalar *ctx_ = NULL;
  // Note: Device backends are generating their own kernels from
  //         single source files, so only Host backends need to
  //         use this Fortran stub.
  if (innerctx) {
    ierr = CeedQFunctionContextGetData(innerctx, CEED_MEM_HOST, &ctx_);
    CeedChk(ierr);
  }

  fctx->f((void *)ctx_,&nq,u[0],u[1],u[2],u[3],u[4],u[5],u[6],
          u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15],
          v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],
          v[10],v[11],v[12],v[13],v[14],v[15],&ierr);

  if (innerctx) {
    ierr = CeedQFunctionContextRestoreData(innerctx, (void *)&ctx_);
    CeedChk(ierr);
  }

  return ierr;
}

#define fCeedQFunctionCreateInterior \
    FORTRAN_NAME(ceedqfunctioncreateinterior, CEEDQFUNCTIONCREATEINTERIOR)
void fCeedQFunctionCreateInterior(int *ceed, int *vlength,
                                  void (*f)(void *ctx, int *nq,
                                      const CeedScalar *u,const CeedScalar *u1,
                                      const CeedScalar *u2,const CeedScalar *u3,
                                      const CeedScalar *u4,const CeedScalar *u5,
                                      const CeedScalar *u6,const CeedScalar *u7,
                                      const CeedScalar *u8,const CeedScalar *u9,
                                      const CeedScalar *u10,const CeedScalar *u11,
                                      const CeedScalar *u12,const CeedScalar *u13,
                                      const CeedScalar *u14,const CeedScalar *u15,
                                      CeedScalar *v,CeedScalar *v1,CeedScalar *v2,
                                      CeedScalar *v3,CeedScalar *v4,
                                      CeedScalar *v5,CeedScalar *v6,
                                      CeedScalar *v7,CeedScalar *v8,
                                      CeedScalar *v9,CeedScalar *v10,
                                      CeedScalar *v11,CeedScalar *v12,
                                      CeedScalar *v13,CeedScalar *v14,
                                      CeedScalar *v15,int *err),
                                  const char *source, int *qf, int *err,
                                  fortran_charlen_t source_len) {
  FIX_STRING(source);
  if (CeedQFunction_count == CeedQFunction_count_max) {
    CeedQFunction_count_max += CeedQFunction_count_max/2 + 1;
    CeedRealloc(CeedQFunction_count_max, &CeedQFunction_dict);
  }

  CeedQFunction *qf_ = &CeedQFunction_dict[CeedQFunction_count];
  *err = CeedQFunctionCreateInterior(Ceed_dict[*ceed], *vlength,
                                     CeedQFunctionFortranStub, source_c, qf_);

  if (*err == 0) {
    *qf = CeedQFunction_count++;
    CeedQFunction_n++;
  }

  CeedFortranContext fctxdata;
  *err = CeedCalloc(1, &fctxdata);
  if (*err) return;
  fctxdata->f = f; fctxdata->innerctx = NULL;
  CeedQFunctionContext fctx;
  *err = CeedQFunctionContextCreate(Ceed_dict[*ceed], &fctx);
  if (*err) return;
  *err = CeedQFunctionContextSetData(fctx, CEED_MEM_HOST, CEED_OWN_POINTER,
                                     sizeof(*fctxdata), fctxdata);
  if (*err) return;
  *err = CeedQFunctionSetContext(*qf_, fctx);
  if (*err) return;
  CeedQFunctionContextDestroy(&fctx);
  if (*err) return;

  *err = CeedQFunctionSetFortranStatus(*qf_, true);
}

#define fCeedQFunctionCreateInteriorByName \
    FORTRAN_NAME(ceedqfunctioncreateinteriorbyname, CEEDQFUNCTIONCREATEINTERIORBYNAME)
void fCeedQFunctionCreateInteriorByName(int *ceed, const char *name, int *qf,
                                        int *err, fortran_charlen_t name_len) {
  FIX_STRING(name);
  if (CeedQFunction_count == CeedQFunction_count_max) {
    CeedQFunction_count_max += CeedQFunction_count_max/2 + 1;
    CeedRealloc(CeedQFunction_count_max, &CeedQFunction_dict);
  }

  CeedQFunction *qf_ = &CeedQFunction_dict[CeedQFunction_count];
  *err = CeedQFunctionCreateInteriorByName(Ceed_dict[*ceed], name_c, qf_);

  if (*err == 0) {
    *qf = CeedQFunction_count++;
    CeedQFunction_n++;
  }
}

#define fCeedQFunctionCreateIdentity \
    FORTRAN_NAME(ceedqfunctioncreateidentity, CEEDQFUNCTIONCREATEIDENTITY)
void fCeedQFunctionCreateIdentity(int *ceed, int *size, int *inmode,
                                  int *outmode, int *qf, int *err) {
  if (CeedQFunction_count == CeedQFunction_count_max) {
    CeedQFunction_count_max += CeedQFunction_count_max/2 + 1;
    CeedRealloc(CeedQFunction_count_max, &CeedQFunction_dict);
  }

  CeedQFunction *qf_ = &CeedQFunction_dict[CeedQFunction_count];
  *err = CeedQFunctionCreateIdentity(Ceed_dict[*ceed], *size,
                                     (CeedEvalMode)*inmode,
                                     (CeedEvalMode)*outmode, qf_);

  if (*err == 0) {
    *qf = CeedQFunction_count++;
    CeedQFunction_n++;
  }
}

#define fCeedQFunctionAddInput \
    FORTRAN_NAME(ceedqfunctionaddinput,CEEDQFUNCTIONADDINPUT)
void fCeedQFunctionAddInput(int *qf, const char *fieldname,
                            CeedInt *ncomp, CeedEvalMode *emode, int *err,
                            fortran_charlen_t fieldname_len) {
  FIX_STRING(fieldname);
  CeedQFunction qf_ = CeedQFunction_dict[*qf];

  *err = CeedQFunctionAddInput(qf_, fieldname_c, *ncomp, *emode);
}

#define fCeedQFunctionAddOutput \
    FORTRAN_NAME(ceedqfunctionaddoutput,CEEDQFUNCTIONADDOUTPUT)
void fCeedQFunctionAddOutput(int *qf, const char *fieldname,
                             CeedInt *ncomp, CeedEvalMode *emode, int *err,
                             fortran_charlen_t fieldname_len) {
  FIX_STRING(fieldname);
  CeedQFunction qf_ = CeedQFunction_dict[*qf];

  *err = CeedQFunctionAddOutput(qf_, fieldname_c, *ncomp, *emode);
}

#define fCeedQFunctionSetContext \
    FORTRAN_NAME(ceedqfunctionsetcontext,CEEDQFUNCTIONSETCONTEXT)
void fCeedQFunctionSetContext(int *qf, int *ctx, int *err) {
  CeedQFunction qf_ = CeedQFunction_dict[*qf];
  CeedQFunctionContext ctx_ = CeedQFunctionContext_dict[*ctx];

  CeedQFunctionContext fctx;
  *err = CeedQFunctionGetContext(qf_, &fctx);
  if (*err) return;
  CeedFortranContext fctxdata;
  *err = CeedQFunctionContextGetData(fctx, CEED_MEM_HOST, &fctxdata);
  if (*err) return;
  fctxdata->innerctx = ctx_;
  *err = CeedQFunctionContextRestoreData(fctx, (void **)&fctxdata);
}

#define fCeedQFunctionView \
    FORTRAN_NAME(ceedqfunctionview,CEEDQFUNCTIONVIEW)
void fCeedQFunctionView(int *qf, int *err) {
  CeedQFunction qf_ = CeedQFunction_dict[*qf];

  *err = CeedQFunctionView(qf_, stdout);
}

#define fCeedQFunctionApply \
    FORTRAN_NAME(ceedqfunctionapply,CEEDQFUNCTIONAPPLY)
//TODO Need Fixing, double pointer
void fCeedQFunctionApply(int *qf, int *Q,
                         int *u, int *u1, int *u2, int *u3,
                         int *u4, int *u5, int *u6, int *u7,
                         int *u8, int *u9, int *u10, int *u11,
                         int *u12, int *u13, int *u14, int *u15,
                         int *v, int *v1, int *v2, int *v3,
                         int *v4, int *v5, int *v6, int *v7,
                         int *v8, int *v9, int *v10, int *v11,
                         int *v12, int *v13, int *v14, int *v15, int *err) {
  CeedQFunction qf_ = CeedQFunction_dict[*qf];
  CeedVector *in;
  *err = CeedCalloc(16, &in);
  if (*err) return;
  in[0] = *u==FORTRAN_NULL?NULL:CeedVector_dict[*u];
  in[1] = *u1==FORTRAN_NULL?NULL:CeedVector_dict[*u1];
  in[2] = *u2==FORTRAN_NULL?NULL:CeedVector_dict[*u2];
  in[3] = *u3==FORTRAN_NULL?NULL:CeedVector_dict[*u3];
  in[4] = *u4==FORTRAN_NULL?NULL:CeedVector_dict[*u4];
  in[5] = *u5==FORTRAN_NULL?NULL:CeedVector_dict[*u5];
  in[6] = *u6==FORTRAN_NULL?NULL:CeedVector_dict[*u6];
  in[7] = *u7==FORTRAN_NULL?NULL:CeedVector_dict[*u7];
  in[8] = *u8==FORTRAN_NULL?NULL:CeedVector_dict[*u8];
  in[9] = *u9==FORTRAN_NULL?NULL:CeedVector_dict[*u9];
  in[10] = *u10==FORTRAN_NULL?NULL:CeedVector_dict[*u10];
  in[11] = *u11==FORTRAN_NULL?NULL:CeedVector_dict[*u11];
  in[12] = *u12==FORTRAN_NULL?NULL:CeedVector_dict[*u12];
  in[13] = *u13==FORTRAN_NULL?NULL:CeedVector_dict[*u13];
  in[14] = *u14==FORTRAN_NULL?NULL:CeedVector_dict[*u14];
  in[15] = *u15==FORTRAN_NULL?NULL:CeedVector_dict[*u15];
  CeedVector *out;
  *err = CeedCalloc(16, &out);
  if (*err) return;
  out[0] = *v==FORTRAN_NULL?NULL:CeedVector_dict[*v];
  out[1] = *v1==FORTRAN_NULL?NULL:CeedVector_dict[*v1];
  out[2] = *v2==FORTRAN_NULL?NULL:CeedVector_dict[*v2];
  out[3] = *v3==FORTRAN_NULL?NULL:CeedVector_dict[*v3];
  out[4] = *v4==FORTRAN_NULL?NULL:CeedVector_dict[*v4];
  out[5] = *v5==FORTRAN_NULL?NULL:CeedVector_dict[*v5];
  out[6] = *v6==FORTRAN_NULL?NULL:CeedVector_dict[*v6];
  out[7] = *v7==FORTRAN_NULL?NULL:CeedVector_dict[*v7];
  out[8] = *v8==FORTRAN_NULL?NULL:CeedVector_dict[*v8];
  out[9] = *v9==FORTRAN_NULL?NULL:CeedVector_dict[*v9];
  out[10] = *v10==FORTRAN_NULL?NULL:CeedVector_dict[*v10];
  out[11] = *v11==FORTRAN_NULL?NULL:CeedVector_dict[*v11];
  out[12] = *v12==FORTRAN_NULL?NULL:CeedVector_dict[*v12];
  out[13] = *v13==FORTRAN_NULL?NULL:CeedVector_dict[*v13];
  out[14] = *v14==FORTRAN_NULL?NULL:CeedVector_dict[*v14];
  out[15] = *v15==FORTRAN_NULL?NULL:CeedVector_dict[*v15];
  *err = CeedQFunctionApply(qf_, *Q, in, out);
  if (*err) return;

  *err = CeedFree(&in);
  if (*err) return;
  *err = CeedFree(&out);
}

#define fCeedQFunctionDestroy \
    FORTRAN_NAME(ceedqfunctiondestroy,CEEDQFUNCTIONDESTROY)
void fCeedQFunctionDestroy(int *qf, int *err) {
  if (*qf == FORTRAN_NULL) return;

  *err = CeedQFunctionDestroy(&CeedQFunction_dict[*qf]);
  if (*err == 0) {
    *qf = FORTRAN_NULL;
    CeedQFunction_n--;
    if (CeedQFunction_n == 0) {
      *err = CeedFree(&CeedQFunction_dict);
      CeedQFunction_count = 0;
      CeedQFunction_count_max = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// CeedOperator
// -----------------------------------------------------------------------------
static CeedOperator *CeedOperator_dict = NULL;
static int CeedOperator_count = 0;
static int CeedOperator_n = 0;
static int CeedOperator_count_max = 0;

#define fCeedOperatorCreate \
    FORTRAN_NAME(ceedoperatorcreate, CEEDOPERATORCREATE)
void fCeedOperatorCreate(int *ceed,
                         int *qf, int *dqf, int *dqfT, int *op, int *err) {
  if (CeedOperator_count == CeedOperator_count_max) {
    CeedOperator_count_max += CeedOperator_count_max/2 + 1;
    CeedRealloc(CeedOperator_count_max, &CeedOperator_dict);
  }

  CeedOperator *op_ = &CeedOperator_dict[CeedOperator_count];

  CeedQFunction dqf_  = CEED_QFUNCTION_NONE, dqfT_ = CEED_QFUNCTION_NONE;
  if (*dqf  != FORTRAN_QFUNCTION_NONE) dqf_  = CeedQFunction_dict[*dqf ];
  if (*dqfT != FORTRAN_QFUNCTION_NONE) dqfT_ = CeedQFunction_dict[*dqfT];

  *err = CeedOperatorCreate(Ceed_dict[*ceed], CeedQFunction_dict[*qf], dqf_,
                            dqfT_, op_);
  if (*err) return;
  *op = CeedOperator_count++;
  CeedOperator_n++;
}

#define fCeedCompositeOperatorCreate \
    FORTRAN_NAME(ceedcompositeoperatorcreate, CEEDCOMPOSITEOPERATORCREATE)
void fCeedCompositeOperatorCreate(int *ceed, int *op, int *err) {
  if (CeedOperator_count == CeedOperator_count_max) {
    CeedOperator_count_max += CeedOperator_count_max/2 + 1;
    CeedRealloc(CeedOperator_count_max, &CeedOperator_dict);
  }

  CeedOperator *op_ = &CeedOperator_dict[CeedOperator_count];

  *err = CeedCompositeOperatorCreate(Ceed_dict[*ceed], op_);
  if (*err) return;
  *op = CeedOperator_count++;
  CeedOperator_n++;
}

#define fCeedOperatorSetField \
    FORTRAN_NAME(ceedoperatorsetfield,CEEDOPERATORSETFIELD)
void fCeedOperatorSetField(int *op, const char *fieldname, int *r, int *b,
                           int *v, int *err, fortran_charlen_t fieldname_len) {
  FIX_STRING(fieldname);
  CeedElemRestriction r_;
  CeedBasis b_;
  CeedVector v_;

  CeedOperator op_ = CeedOperator_dict[*op];

  if (*r == FORTRAN_NULL) {
    r_ = NULL;
  } else if (*r == FORTRAN_ELEMRESTRICTION_NONE) {
    r_ = CEED_ELEMRESTRICTION_NONE;
  } else {
    r_ = CeedElemRestriction_dict[*r];
  }

  if (*b == FORTRAN_NULL) {
    b_ = NULL;
  } else if (*b == FORTRAN_BASIS_COLLOCATED) {
    b_ = CEED_BASIS_COLLOCATED;
  } else {
    b_ = CeedBasis_dict[*b];
  }
  if (*v == FORTRAN_NULL) {
    v_ = NULL;
  } else if (*v == FORTRAN_VECTOR_ACTIVE) {
    v_ = CEED_VECTOR_ACTIVE;
  } else if (*v == FORTRAN_VECTOR_NONE) {
    v_ = CEED_VECTOR_NONE;
  } else {
    v_ = CeedVector_dict[*v];
  }

  *err = CeedOperatorSetField(op_, fieldname_c, r_, b_, v_);
}

#define fCeedCompositeOperatorAddSub \
    FORTRAN_NAME(ceedcompositeoperatoraddsub, CEEDCOMPOSITEOPERATORADDSUB)
void fCeedCompositeOperatorAddSub(int *compositeop, int *subop, int *err) {
  CeedOperator compositeop_ = CeedOperator_dict[*compositeop];
  CeedOperator subop_ = CeedOperator_dict[*subop];

  *err = CeedCompositeOperatorAddSub(compositeop_, subop_);
  if (*err) return;
}

#define fCeedOperatorLinearAssembleQFunction \
    FORTRAN_NAME(ceedoperatorlinearassembleqfunction, CEEDOPERATORLINEARASSEMBLEQFUNCTION)
void fCeedOperatorLinearAssembleQFunction(int *op, int *assembledvec,
    int *assembledrstr, int *rqst, int *err) {
  // Vector
  if (CeedVector_count == CeedVector_count_max) {
    CeedVector_count_max += CeedVector_count_max/2 + 1;
    CeedRealloc(CeedVector_count_max, &CeedVector_dict);
  }
  CeedVector *assembledvec_ = &CeedVector_dict[CeedVector_count];

  // Restriction
  if (CeedElemRestriction_count == CeedElemRestriction_count_max) {
    CeedElemRestriction_count_max += CeedElemRestriction_count_max/2 + 1;
    CeedRealloc(CeedElemRestriction_count_max, &CeedElemRestriction_dict);
  }
  CeedElemRestriction *rstr_ =
    &CeedElemRestriction_dict[CeedElemRestriction_count];

  int createRequest = 1;
  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*rqst == -1 || *rqst == -2) {
    createRequest = 0;
  }

  if (createRequest && CeedRequest_count == CeedRequest_count_max) {
    CeedRequest_count_max += CeedRequest_count_max/2 + 1;
    CeedRealloc(CeedRequest_count_max, &CeedRequest_dict);
  }

  CeedRequest *rqst_;
  if (*rqst == -1) rqst_ = CEED_REQUEST_IMMEDIATE;
  else if (*rqst == -2) rqst_ = CEED_REQUEST_ORDERED;
  else rqst_ = &CeedRequest_dict[CeedRequest_count];

  *err = CeedOperatorLinearAssembleQFunction(CeedOperator_dict[*op],
         assembledvec_, rstr_, rqst_);
  if (*err) return;
  if (createRequest) {
    *rqst = CeedRequest_count++;
    CeedRequest_n++;
  }

  if (*err == 0) {
    *assembledrstr = CeedElemRestriction_count++;
    CeedElemRestriction_n++;
    *assembledvec = CeedVector_count++;
    CeedVector_n++;
  }
}

#define fCeedOperatorLinearAssembleDiagonal \
    FORTRAN_NAME(ceedoperatorlinearassemblediagonal, CEEDOPERATORLINEARASSEMBLEDIAGONAL)
void fCeedOperatorLinearAssembleDiagonal(int *op, int *assembledvec,
    int *rqst, int *err) {
  int createRequest = 1;
  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*rqst == -1 || *rqst == -2) {
    createRequest = 0;
  }

  if (createRequest && CeedRequest_count == CeedRequest_count_max) {
    CeedRequest_count_max += CeedRequest_count_max/2 + 1;
    CeedRealloc(CeedRequest_count_max, &CeedRequest_dict);
  }

  CeedRequest *rqst_;
  if (*rqst == -1) rqst_ = CEED_REQUEST_IMMEDIATE;
  else if (*rqst == -2) rqst_ = CEED_REQUEST_ORDERED;
  else rqst_ = &CeedRequest_dict[CeedRequest_count];

  *err = CeedOperatorLinearAssembleDiagonal(CeedOperator_dict[*op],
         CeedVector_dict[*assembledvec], rqst_);
  if (*err) return;
  if (createRequest) {
    *rqst = CeedRequest_count++;
    CeedRequest_n++;
  }
}

#define fCeedOperatorMultigridLevelCreate \
    FORTRAN_NAME(ceedoperatormultigridlevelcreate, CEEDOPERATORMULTIGRIDLEVELCREATE)
void fCeedOperatorMultigridLevelCreate(int *opFine, int *pMultFine,
                                       int *rstrCoarse, int *basisCoarse, int *opCoarse,
                                       int *opProlong, int *opRestrict, int *err) {
  // Operators
  CeedOperator opCoarse_, opProlong_, opRestrict_;

  // C interface call
  *err = CeedOperatorMultigridLevelCreate(
           CeedOperator_dict[*opFine], CeedVector_dict[*pMultFine],
           CeedElemRestriction_dict[*rstrCoarse],
           CeedBasis_dict[*basisCoarse],
           &opCoarse_, &opProlong_, &opRestrict_);

  if (*err) return;
  while (CeedOperator_count + 2 >= CeedOperator_count_max) {
    CeedOperator_count_max += CeedOperator_count_max/2 + 1;
  }
  CeedRealloc(CeedOperator_count_max, &CeedOperator_dict);
  CeedOperator_dict[CeedOperator_count] = opCoarse_;
  *opCoarse = CeedOperator_count++;
  CeedOperator_dict[CeedOperator_count] = opProlong_;
  *opProlong = CeedOperator_count++;
  CeedOperator_dict[CeedOperator_count] = opRestrict_;
  *opRestrict = CeedOperator_count++;
  CeedOperator_n += 3;
}

#define fCeedOperatorMultigridLevelCreateTensorH1 \
    FORTRAN_NAME(ceedoperatormultigridlevelcreatetensorh1, CEEDOPERATORMULTIGRIDLEVELCREATETENSORH1)
void fCeedOperatorMultigridLevelCreateTensorH1(int *opFine, int *pMultFine,
    int *rstrCoarse, int *basisCoarse, const CeedScalar *interpCtoF,
    int *opCoarse, int *opProlong, int *opRestrict, int *err) {
  // Operators
  CeedOperator opCoarse_, opProlong_, opRestrict_;

  // C interface call
  *err = CeedOperatorMultigridLevelCreateTensorH1(
           CeedOperator_dict[*opFine], CeedVector_dict[*pMultFine],
           CeedElemRestriction_dict[*rstrCoarse], CeedBasis_dict[*basisCoarse],
           interpCtoF, &opCoarse_, &opProlong_, &opRestrict_);

  if (*err) return;
  while (CeedOperator_count + 2 >= CeedOperator_count_max) {
    CeedOperator_count_max += CeedOperator_count_max/2 + 1;
  }
  CeedRealloc(CeedOperator_count_max, &CeedOperator_dict);
  CeedOperator_dict[CeedOperator_count] = opCoarse_;
  *opCoarse = CeedOperator_count++;
  CeedOperator_dict[CeedOperator_count] = opProlong_;
  *opProlong = CeedOperator_count++;
  CeedOperator_dict[CeedOperator_count] = opRestrict_;
  *opRestrict = CeedOperator_count++;
  CeedOperator_n += 3;
}

#define fCeedOperatorMultigridLevelCreateH1 \
    FORTRAN_NAME(ceedoperatormultigridlevelcreateh1, CEEDOPERATORMULTIGRIDLEVELCREATEH1)
void fCeedOperatorMultigridLevelCreateH1(int *opFine, int *pMultFine,
    int *rstrCoarse, int *basisCoarse, const CeedScalar *interpCtoF,
    int *opCoarse, int *opProlong, int *opRestrict, int *err) {
  // Operators
  CeedOperator opCoarse_, opProlong_, opRestrict_;

  // C interface call
  *err = CeedOperatorMultigridLevelCreateH1(
           CeedOperator_dict[*opFine], CeedVector_dict[*pMultFine],
           CeedElemRestriction_dict[*rstrCoarse], CeedBasis_dict[*basisCoarse],
           interpCtoF, &opCoarse_, &opProlong_, &opRestrict_);

  if (*err) return;
  while (CeedOperator_count + 2 >= CeedOperator_count_max) {
    CeedOperator_count_max += CeedOperator_count_max/2 + 1;
  }
  CeedRealloc(CeedOperator_count_max, &CeedOperator_dict);
  CeedOperator_dict[CeedOperator_count] = opCoarse_;
  *opCoarse = CeedOperator_count++;
  CeedOperator_dict[CeedOperator_count] = opProlong_;
  *opProlong = CeedOperator_count++;
  CeedOperator_dict[CeedOperator_count] = opRestrict_;
  *opRestrict = CeedOperator_count++;
  CeedOperator_n += 3;
}

#define fCeedOperatorView \
    FORTRAN_NAME(ceedoperatorview,CEEDOPERATORVIEW)
void fCeedOperatorView(int *op, int *err) {
  CeedOperator op_ = CeedOperator_dict[*op];

  *err = CeedOperatorView(op_, stdout);
}

#define fCeedOperatorCreateFDMElementInverse \
    FORTRAN_NAME(ceedoperatorcreatefdmelementinverse, CEEDOPERATORCREATEFDMELEMENTINVERSE)
void fCeedOperatorCreateFDMElementInverse(int *op, int *fdminv,
    int *rqst, int *err) {
  // Operator
  if (CeedOperator_count == CeedOperator_count_max) {
    CeedOperator_count_max += CeedOperator_count_max/2 + 1;
    CeedRealloc(CeedOperator_count_max, &CeedOperator_dict);
  }
  CeedOperator *fdminv_ =
    &CeedOperator_dict[CeedOperator_count];

  int createRequest = 1;
  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*rqst == -1 || *rqst == -2) {
    createRequest = 0;
  }

  if (createRequest && CeedRequest_count == CeedRequest_count_max) {
    CeedRequest_count_max += CeedRequest_count_max/2 + 1;
    CeedRealloc(CeedRequest_count_max, &CeedRequest_dict);
  }

  CeedRequest *rqst_;
  if (*rqst == -1) rqst_ = CEED_REQUEST_IMMEDIATE;
  else if (*rqst == -2) rqst_ = CEED_REQUEST_ORDERED;
  else rqst_ = &CeedRequest_dict[CeedRequest_count];

  *err = CeedOperatorCreateFDMElementInverse(CeedOperator_dict[*op],
         fdminv_, rqst_);
  if (*err) return;
  if (createRequest) {
    *rqst = CeedRequest_count++;
    CeedRequest_n++;
  }

  if (*err == 0) {
    *fdminv = CeedOperator_count++;
    CeedOperator_n++;
  }
}

#define fCeedOperatorApply FORTRAN_NAME(ceedoperatorapply, CEEDOPERATORAPPLY)
void fCeedOperatorApply(int *op, int *ustatevec,
                        int *resvec, int *rqst, int *err) {
  CeedVector ustatevec_ = (*ustatevec == FORTRAN_NULL) ?
                          NULL : (*ustatevec == FORTRAN_VECTOR_NONE ?
                                  CEED_VECTOR_NONE : CeedVector_dict[*ustatevec]);
  CeedVector resvec_ = (*resvec == FORTRAN_NULL) ?
                       NULL : (*resvec == FORTRAN_VECTOR_NONE ?
                               CEED_VECTOR_NONE : CeedVector_dict[*resvec]);

  int createRequest = 1;
  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*rqst == -1 || *rqst == -2) {
    createRequest = 0;
  }

  if (createRequest && CeedRequest_count == CeedRequest_count_max) {
    CeedRequest_count_max += CeedRequest_count_max/2 + 1;
    CeedRealloc(CeedRequest_count_max, &CeedRequest_dict);
  }

  CeedRequest *rqst_;
  if (*rqst == -1) rqst_ = CEED_REQUEST_IMMEDIATE;
  else if (*rqst == -2) rqst_ = CEED_REQUEST_ORDERED;
  else rqst_ = &CeedRequest_dict[CeedRequest_count];

  *err = CeedOperatorApply(CeedOperator_dict[*op],
                           ustatevec_, resvec_, rqst_);
  if (*err) return;
  if (createRequest) {
    *rqst = CeedRequest_count++;
    CeedRequest_n++;
  }
}

#define fCeedOperatorApplyAdd FORTRAN_NAME(ceedoperatorapplyadd, CEEDOPERATORAPPLYADD)
void fCeedOperatorApplyAdd(int *op, int *ustatevec,
                           int *resvec, int *rqst, int *err) {
  CeedVector ustatevec_ = *ustatevec == FORTRAN_NULL
                          ? NULL : CeedVector_dict[*ustatevec];
  CeedVector resvec_ = *resvec == FORTRAN_NULL
                       ? NULL : CeedVector_dict[*resvec];

  int createRequest = 1;
  // Check if input is CEED_REQUEST_ORDERED(-2) or CEED_REQUEST_IMMEDIATE(-1)
  if (*rqst == -1 || *rqst == -2) {
    createRequest = 0;
  }

  if (createRequest && CeedRequest_count == CeedRequest_count_max) {
    CeedRequest_count_max += CeedRequest_count_max/2 + 1;
    CeedRealloc(CeedRequest_count_max, &CeedRequest_dict);
  }

  CeedRequest *rqst_;
  if (*rqst == -1) rqst_ = CEED_REQUEST_IMMEDIATE;
  else if (*rqst == -2) rqst_ = CEED_REQUEST_ORDERED;
  else rqst_ = &CeedRequest_dict[CeedRequest_count];

  *err = CeedOperatorApplyAdd(CeedOperator_dict[*op],
                              ustatevec_, resvec_, rqst_);
  if (*err) return;
  if (createRequest) {
    *rqst = CeedRequest_count++;
    CeedRequest_n++;
  }
}

#define fCeedOperatorApplyJacobian \
    FORTRAN_NAME(ceedoperatorapplyjacobian, CEEDOPERATORAPPLYJACOBIAN)
void fCeedOperatorApplyJacobian(int *op, int *qdatavec, int *ustatevec,
                                int *dustatevec, int *dresvec, int *rqst,
                                int *err) {
// TODO Uncomment this when CeedOperatorApplyJacobian is implemented
//  *err = CeedOperatorApplyJacobian(CeedOperator_dict[*op], CeedVector_dict[*qdatavec],
//             CeedVector_dict[*ustatevec], CeedVector_dict[*dustatevec],
//             CeedVector_dict[*dresvec], &CeedRequest_dict[*rqst]);
}

#define fCeedOperatorDestroy \
    FORTRAN_NAME(ceedoperatordestroy, CEEDOPERATORDESTROY)
void fCeedOperatorDestroy(int *op, int *err) {
  if (*op == FORTRAN_NULL) return;
  *err = CeedOperatorDestroy(&CeedOperator_dict[*op]);
  if (*err == 0) {
    *op = FORTRAN_NULL;
    CeedOperator_n--;
    if (CeedOperator_n == 0) {
      *err = CeedFree(&CeedOperator_dict);
      CeedOperator_count = 0;
      CeedOperator_count_max = 0;
    }
  }
}

// -----------------------------------------------------------------------------
