/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <helper_cuda.h>
#include <vector>
#include "jacobi.h"


// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 8
#define SUB_GRP_SIZE 32

#if !defined(DPCT_COMPATIBILITY_TEMP) || DPCT_COMPATIBILITY_TEMP >= 600
#else
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

static void JacobiMethod(const float *A, const double *b,
                                    const float conv_threshold, double *x,
                                    double *x_new, double *sum,
                                    const sycl::nd_item<3> &item_ct1,
                                    double *x_shared, double *b_shared) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();
    // N_ROWS == n

  for (int i = item_ct1.get_local_id(2); i < N_ROWS;
       i += item_ct1.get_local_range(2)) {
    x_shared[i] = x[i];
  }

  if (item_ct1.get_local_id(2) < ROWS_PER_CTA) {
    int k = item_ct1.get_local_id(2);
    for (int i = k + (item_ct1.get_group(2) * ROWS_PER_CTA);
         (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
      b_shared[i % (ROWS_PER_CTA + 1)] = b[i];
    }
  }

  /*
  DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  for (int k = 0, i = item_ct1.get_group(2) * ROWS_PER_CTA;
       (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++) {
    double rowThreadSum = 0.0;
    for (int j = item_ct1.get_local_id(2); j < N_ROWS;
         j += item_ct1.get_local_range(2)) {
      rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
    }

    for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      rowThreadSum += sycl::shift_group_left(item_ct1.get_sub_group(),
                                             rowThreadSum, offset);
    }

    if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
    }
  }

  /*
  DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < ROWS_PER_CTA) {
    dpct::experimental::logical_group tile8 = dpct::experimental::logical_group(
        item_ct1, item_ct1.get_group(), ROWS_PER_CTA);
    double temp_sum = 0.0;

    int k = item_ct1.get_local_id(2);

    for (int i = k + (item_ct1.get_group(2) * ROWS_PER_CTA);
         (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
      double dx = b_shared[i % (ROWS_PER_CTA + 1)];
      dx /= A[i * N_ROWS + i];

      x_new[i] = (x_shared[i] + dx);
      temp_sum += sycl::fabs(dx);
    }

    for (int offset = tile8.get_local_linear_range() / 2; offset > 0;
         offset /= 2) {
      temp_sum += dpct::shift_sub_group_left(item_ct1.get_sub_group(), temp_sum,
                                             offset, 8);
    }

    if (tile8.get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          sum, temp_sum);
    }
  }
}

// Thread block size for finalError kernel should be multiple of 32
static void finalError(double *x, double *g_sum,
                       const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
  // Handle to thread block group
  auto cta = item_ct1.get_group();
  auto warpSum = (double *)dpct_local;
  double sum = 0.0;

  int globalThreadId = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2);

  for (int i = globalThreadId; i < N_ROWS;
       i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
    double d = x[i] - 1.0;
    sum += sycl::fabs(d);
  }

  sycl::sub_group tile32 = item_ct1.get_sub_group();

  for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
       offset > 0; offset /= 2) {
    sum += sycl::shift_group_left(item_ct1.get_sub_group(), sum, offset);
  }

  if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
    warpSum[item_ct1.get_local_id(2) /
            item_ct1.get_sub_group().get_local_range().get(0)] = sum;
  }

  /*
  DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  double blockSum = 0.0;
  if (item_ct1.get_local_id(2) <
      (item_ct1.get_local_range(2) /
       item_ct1.get_sub_group().get_local_range().get(0))) {
    blockSum = warpSum[item_ct1.get_local_id(2)];
  }

  if (item_ct1.get_local_id(2) < 32) {
    for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      blockSum +=
          sycl::shift_group_left(item_ct1.get_sub_group(), blockSum, offset);
    }
    if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          g_sum, blockSum);
    }
  }
}

double JacobiMethodGpuCudaGraphExecKernelSetParams(
    const float *A, const double *b, const float conv_threshold,
    const int max_iter, double *x, double *x_new, dpct::queue_ptr stream, sycl::queue q) {
    
    namespace sycl_ext = sycl::ext::oneapi::experimental;
   //  sycl::queue q = sycl::queue{sycl::gpu_selector_v};
    sycl_ext::command_graph graph(q.get_context(), q.get_device());
    
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);

  double sum = 0.0;
  int k = 0;
  double *params[] = {x, x_new};
  double *d_sum = NULL;
  DPCT_CHECK_ERROR(
      d_sum = sycl::malloc_device<double>(1, q));
    
  
  auto memset_node = graph.add([&](sycl::handler& h){
      
      h.fill(d_sum, 0, sizeof(double));
  });
    
     printf("check1\n");
  auto kernel_node = graph.add([&](sycl::handler &cgh) {
    sycl::local_accessor<double, 1> x_shared_acc_ct1(
                            sycl::range<1>(N_ROWS), cgh);

    sycl::local_accessor<double, 1> b_shared_acc_ct1(
                            sycl::range<1>(ROWS_PER_CTA + 1), cgh);
      
      printf("check2\n");

    cgh.parallel_for(
      sycl::nd_range<3>(nblocks * nthreads, nthreads),
                            [=](sycl::nd_item<3> item_ct1) [
                                [intel::reqd_sub_group_size(SUB_GRP_SIZE)]] {
                              JacobiMethod(A, b, conv_threshold, params[k % 2],
                                           params[(k + 1) % 2], d_sum, item_ct1,
                                           x_shared_acc_ct1.get_pointer(),
                                           b_shared_acc_ct1.get_pointer());
      });
  },  sycl_ext::property::node::depends_on(memset_node));
    
     printf("check3\n");
    
    auto memcpy_node = graph.add([&](sycl::handler &cgh) {
      cgh.memcpy(&sum, d_sum, sizeof(double));  
  }, sycl_ext::property::node::depends_on(kernel_node));
    
     printf("check4\n");
    
    auto exec_graph = graph.finalize();
    
     printf("check5\n");
    
    sycl::queue qexec = sycl::queue{sycl::gpu_selector_v, 
      {sycl::ext::intel::property::queue::no_immediate_command_list()}};
    dpct::has_capability_or_fail(qexec.get_device(), {sycl::aspect::fp64});
    
     printf("check6\n");
    
    for (k = 0; k < max_iter; k++) {
        
      qexec.submit([&](sycl::handler& cgh) {
      cgh.ext_oneapi_graph(exec_graph);
    }).wait(); 
    
     printf("check7, At iteration : %d\n", k);
     printf("Sum= %f\n",sum);
    //..................................................................

    if (sum <= conv_threshold) {
        
         printf("check8\n");
      q.memset(d_sum, 0, sizeof(double));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;

      size_t sharedMemSize =
          ((nthreads[2] / SUB_GRP_SIZE) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads), [=
          ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                finalError(x_new, d_sum, item_ct1,
                           dpct_local_acc_ct1.get_pointer());
              });
        });
      } else {
        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads), [=
          ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                finalError(x, d_sum, item_ct1,
                           dpct_local_acc_ct1.get_pointer());
              });
        });
      }
     
         printf("check9\n");
      q.memcpy(&sum, d_sum, sizeof(double)).wait();
      printf("Device iterations : %d\n", k + 1);
      printf("Device error : %.3e\n", sum);
      break;
    }
    }
     printf("check10\n");
  sycl::free(d_sum, q);
  return sum;
}

double JacobiMethodGpuCudaGraphExecUpdate(const float *A, const double *b,
                                          const float conv_threshold,
                                          const int max_iter, double *x,
                                          double *x_new,
                                          dpct::queue_ptr stream, sycl::queue q) {
  namespace sycl_ext = sycl::ext::oneapi::experimental;
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);

  double sum = 0.0;
  int k = 0;
  double *params[] = {x, x_new};
  double *d_sum = NULL;
  DPCT_CHECK_ERROR(
      d_sum = sycl::malloc_device<double>(1, q));
    
  sycl_ext::command_graph graph(q.get_context(), q.get_device());
  
  graph.begin_recording(q);
    
  sycl::event memset_node = q.fill(d_sum, 0, sizeof(double));
    
  sycl::event kernel_node = q.submit([&](sycl::handler &cgh) {
    cgh.depends_on({memset_node});
    sycl::local_accessor<double, 1> x_shared_acc_ct1(
                            sycl::range<1>(N_ROWS), cgh);

    sycl::local_accessor<double, 1> b_shared_acc_ct1(
                            sycl::range<1>(ROWS_PER_CTA + 1), cgh);

    cgh.parallel_for(
      sycl::nd_range<3>(nblocks * nthreads, nthreads),
                            [=](sycl::nd_item<3> item_ct1) [
                                [intel::reqd_sub_group_size(SUB_GRP_SIZE)]] {
                              JacobiMethod(A, b, conv_threshold, params[k % 2],
                                           params[(k + 1) % 2], d_sum, item_ct1,
                                           x_shared_acc_ct1.get_pointer(),
                                           b_shared_acc_ct1.get_pointer());
      });
  });
    
    q.submit([&](sycl::handler &cgh) {
      cgh.depends_on(kernel_node);
      cgh.memcpy(&sum, d_sum, sizeof(double));  
  });
 
  graph.end_recording();
  auto exec_graph = graph.finalize();

  
  for (k = 0; k < max_iter; k++) {

      q.submit([&](sycl::handler& cgh) {
      cgh.ext_oneapi_graph(exec_graph);
    }).wait(); 
      

    if (sum <= conv_threshold) {
      q.memset(d_sum, 0, sizeof(double));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;

      size_t sharedMemSize =
          ((nthreads[2] / SUB_GRP_SIZE) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads), [=
          ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                finalError(x_new, d_sum, item_ct1,
                           dpct_local_acc_ct1.get_pointer());
              });
        });
      } else {
        q.submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(nblocks * nthreads, nthreads), [=
          ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                finalError(x, d_sum, item_ct1,
                           dpct_local_acc_ct1.get_pointer());
              });
        });
      }
    
      q.memcpy(&sum, d_sum, sizeof(double)).wait();
      printf("Device iterations : %d\n", k + 1);
      printf("Device error : %.3e\n", sum);
      break;
    }
    }
  sycl::free(d_sum, q);
  return sum;
}

double JacobiMethodGpu(const float *A, const double *b,
                       const float conv_threshold, const int max_iter,
                       double *x, double *x_new, dpct::queue_ptr stream) {
  // CTA size
  sycl::range<3> nthreads(1, 1, 256);
  // grid size
  sycl::range<3> nblocks(1, 1, (N_ROWS / ROWS_PER_CTA) + 2);

  double sum = 0.0;
  double *d_sum;
  checkCudaErrors(DPCT_CHECK_ERROR(
      d_sum = sycl::malloc_device<double>(1, dpct::get_default_queue())));
  int k = 0;

  for (k = 0; k < max_iter; k++) {
    checkCudaErrors(DPCT_CHECK_ERROR(stream->memset(d_sum, 0, sizeof(double))));
    if ((k & 1) == 0) {
      /*
      DPCT1049:11: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
      stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:50: 'N_ROWS' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it
        is correct.
        */
        sycl::local_accessor<double, 1> x_shared_acc_ct1(
            sycl::range<1>(512 /*N_ROWS*/), cgh);
        /*
        DPCT1101:51: 'ROWS_PER_CTA + 1' expression was replaced with a
        value. Modify the code to use the original expression, provided in
        comments, if it is correct.
        */
        sycl::local_accessor<double, 1> b_shared_acc_ct1(
            sycl::range<1>(9 /*ROWS_PER_CTA + 1*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(nblocks * nthreads, nthreads),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              JacobiMethod(A, b, conv_threshold, x, x_new, d_sum, item_ct1,
                           x_shared_acc_ct1.get_pointer(),
                           b_shared_acc_ct1.get_pointer());
            });
      });
    } else {
      /*
      DPCT1049:12: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64});
      stream->submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:52: 'N_ROWS' expression was replaced with a value. Modify
        the code to use the original expression, provided in comments, if it
        is correct.
        */
        sycl::local_accessor<double, 1> x_shared_acc_ct1(
            sycl::range<1>(512 /*N_ROWS*/), cgh);
        /*
        DPCT1101:53: 'ROWS_PER_CTA + 1' expression was replaced with a
        value. Modify the code to use the original expression, provided in
        comments, if it is correct.
        */
        sycl::local_accessor<double, 1> b_shared_acc_ct1(
            sycl::range<1>(9 /*ROWS_PER_CTA + 1*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(nblocks * nthreads, nthreads),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              JacobiMethod(A, b, conv_threshold, x_new, x, d_sum, item_ct1,
                           x_shared_acc_ct1.get_pointer(),
                           b_shared_acc_ct1.get_pointer());
            });
      });
    }
    checkCudaErrors(
        DPCT_CHECK_ERROR(stream->memcpy(&sum, d_sum, sizeof(double))));
    checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));

    if (sum <= conv_threshold) {
      checkCudaErrors(
          DPCT_CHECK_ERROR(stream->memset(d_sum, 0, sizeof(double))));
      nblocks[2] = (N_ROWS / nthreads[2]) + 1;
      /*
      DPCT1083:14: The size of local memory in the migrated code may be
      different from the original code. Check that the allocated memory size in
      the migrated code is correct.
      */
      size_t sharedMemSize = ((nthreads[2] / 32) + 1) * sizeof(double);
      if ((k & 1) == 0) {
        /*
        DPCT1049:13: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});
        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                           [=](sycl::nd_item<3> item_ct1)
                               [[intel::reqd_sub_group_size(32)]] {
                                 finalError(x_new, d_sum, item_ct1,
                                            dpct_local_acc_ct1.get_pointer());
                               });
        });
      } else {
        /*
        DPCT1049:15: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp64});
        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
              sycl::range<1>(sharedMemSize), cgh);

          cgh.parallel_for(sycl::nd_range<3>(nblocks * nthreads, nthreads),
                           [=](sycl::nd_item<3> item_ct1)
                               [[intel::reqd_sub_group_size(32)]] {
                                 finalError(x, d_sum, item_ct1,
                                            dpct_local_acc_ct1.get_pointer());
                               });
        });
      }

      checkCudaErrors(
          DPCT_CHECK_ERROR(stream->memcpy(&sum, d_sum, sizeof(double))));
      checkCudaErrors(DPCT_CHECK_ERROR(stream->wait()));
      printf("GPU iterations : %d\n", k + 1);
      printf("GPU error : %.3e\n", sum);
      break;
    }
  }

  checkCudaErrors(
      DPCT_CHECK_ERROR(sycl::free(d_sum, dpct::get_default_queue())));
  return sum;
}
