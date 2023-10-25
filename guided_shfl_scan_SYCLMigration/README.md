# `SHFL_Scan` Sample
 
The `SHFL_Scan`, CUDA parallel prefix sum with shuffle intrinsics sample demonstrates the use of shuffle intrinsic __shfl_up_sync to perform a scan operation across a thread block. The sample also demonstrates the migration of these CUDA shuffle intrinsic APIs to group algorithm. The sample is implemented using SYCL* by migrating code from original CUDA source code and offloading computations to a CPU, GPU, or accelerator.

| Area                      | Description
|:---                       |:---
| What you will learn       | Migrate SHFL_Scan sample from CUDA to SYCL.
| Time to complete          | 15 minutes
| Category                  | Concepts and Functionality

> **Note**: This sample is based on the [SHFL_Scan](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/shfl_scan) sample in the NVIDIA/cuda-samples GitHub repository.

## Purpose

The Jacobi method is used to find approximate numerical solutions for systems of linear equations of the form $Ax = b$ in numerical linear algebra, which is diagonally dominant. 
The parallel implementation demonstrates the use of CUDA Graph through explicit API calls and Stream Capture. It also covers explanations of key SYCL concepts, such as

- Cooperative groups
- Shared Memory
- Reduction operation
- Streams 
- Atomics

 This sample illustrates the steps needed for manual migration of explicit CUDA Graph APIs such as `cudaGraphCreate()`, `cudaGraphAddMemcpyNode()`, `cudaGraphLaunch()` to SYCL equivalent APIs using [Taskflow](https://github.com/taskflow/taskflow) programming Model.

>  **Note**: The sample used the open-source SYCLomatic tool that assists developers in porting CUDA code to SYCL code. To finish the process, you must complete the rest of the coding manually and then tune to the desired level of performance for the target architecture. You can also use the Intel® DPC++ Compatibility Tool available to augment Base Toolkit.

This sample contains three versions in the following folders:

| Folder Name                             | Description
|:---                                     |:---
| `01_dpct_output`                        | Contains output of SYCLomatic tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_migrated`                      | Contains manually migrated SYCL code from CUDA code.
| `03_sycl_migrated_optimized`            | Contains manually migrated SYCL code from CUDA code with performance optimizations applied.

## Prerequisites

| Optimized for              | Description
|:---                        |:---
| OS                         | Ubuntu* 22.04
| Hardware                   | Intel® Gen9 <br> Gen11 <br> Xeon CPU <br> Data Center GPU Max <br> Nvidia Testla P100 <br> Nvidia A100 <br> Nvidia H100 
| Software                   | SYCLomatic (Tag - 20230720) <br> Intel® oneAPI Base Toolkit (Base Kit) version 2023.2.1 <br> oneAPI for NVIDIA GPUs plugin (version 2023.2.0) from Codeplay

For more information on how to install Syclomatic Tool, visit [Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html#gs.v354cy) <br>
Refer [oneAPI for NVIDIA GPUs plugin](https://developer.codeplay.com/products/oneapi/nvidia/) from Codeplay to execute sample on NVIDIA GPU.

## Key Implementation Details

This sample demonstrates the migration of the following prominent CUDA features:

- CUDA Graph APIs
- CUDA Stream Capture
- Atomic Operations
- Shared memory
- CUDA streams 
- Cooperative groups
- Warp-level Primitives

The Jacobi CUDA Graphs computations happen inside a two- kernel Jacobi Method and Final Error Kernels., Element reduction is performed to obtain the final error or sum value. 
In this sample, the vectors are loaded into shared memory for faster memory access and thread blocks are partitioned into tiles. Then reduction of input data is performed in each of the partitioned tiles using sub-group primitives. These intermediate results are then added to a final sum variable via an atomic add operation. 

The computation kernels can be scheduled using two alternative types of host function calls:

1.  Host function `JacobiMethodGpuCudaGraphExecKernelSetParams()`, which uses explicit CUDA Graph APIs 
2.  Host function `JacobiMethodGpu()`, which uses regular CUDA APIs to launch kernels.

>  **Note**: Refer to [Workflow for a CUDA* to SYCL* Migration](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for general information about the migration workflow.

### CUDA Source Code Evaluation

The Jacobi CUDA Graphs sample uses Jacobi iterative algorithm to determines the number of iterations needed to solve system of Linear Equations. All computations happen inside a for-loop.

There are two exit criteria from the loop:
  1.  Execution reaches the maximum number of iterations 
  2.  The final error falls below the desired tolerance.
 
Each iteration has two parts:
  - Jacobi Method computation
  - Final Error computation

In both `Jacobi Method` and `Final Error` Kernels reduction is performed to obtain the final error or sum value. 
The kernel uses cooperative groups, warp-level primitives, atomics and shared memory for the faster and frequent memory access to the block. These computation are loaded into kernel by host function which can be achieved through any one of the three methods. 
  1.  `JacobiMethodGpuCudaGraphExecKernelSetParams()`, which uses explicit CUDA Graph APIs
  2.  `JacobiMethodGpuCudaGraphExecUpdate()`, which uses CUDA stream capture APIs to launch
  3.  `JacobiMethodGpu()`, which uses regular CUDA APIs to launch kernels. 
  
  We migrate the first and third host function using SYCLomatic. We then migrate the remaining CUDA Graphs code section using [Taskflow](https://github.com/taskflow/taskflow) Programming Model. 
  We do not migrate `JacobiMethodGpuCudaGraphExecUpdate()`, because CUDA Stream Capture APIs are not yet supported in SYCL.

For information on how to use SYCLomatic, refer to the materials at *[Migrate from CUDA* to C++ with SYCL*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html)*.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Migrate the `Jacobi CUDA Graphs` Code

### Migrate the Code using SYCLomatic

For this sample, the SYCLomatic tool automatically migrates ~80% of the CUDA runtime APIs to SYCL. Follow these steps to generate the SYCL code using the compatibility tool.

1. Clone the required GitHub repository to your local environment.
   ```
   git clone https://github.com/NVIDIA/cuda-samples.git
   ```
2. Change to the JacobiCudaGraphs sample directory.
   ```
   cd cuda-samples/Samples/3_CUDA_Features/jacobiCudaGraphs/
   ```
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
   The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.

4. Pass the JSON file as input to the SYCLomatic tool. The result is written to a folder named dpct_output. The `--in-root` specifies path to the root of the source tree to be migrated. The `--gen-helper-function` option will make a copy of dpct header files/functions used in migrated code into the dpct_output folder as `include` folder. The `--use-experimental-features` option specifies experimental helper function used to logically group work-items.
   ```
   c2s -p compile_commands.json --in-root ../../.. --gen-helper-function --use-experimental-features=logical-group
   ```

### Manual Workarounds 


### Optimizations

Once you migrate the CUDA code to SYCL successfully and you have functional code, you can optimize the code by using profiling tools, which can help in identifying the hotspots such as operations/instructions taking longer time to execute, memory utilization, and the like. 

#### Reduction Operation Optimization 
    
    ```
    for (int offset = item_ct1.get_sub_group().get_local_linear_range() / 2;
         offset > 0; offset /= 2) {
      rowThreadSum += sycl::shift_group_left(item_ct1.get_sub_group(),
                                             rowThreadSum, offset);
    }
    ```
    
The sub-group function `shift_group_left` works by exchanging values between work-items in the sub-group via a shift. But needs to be looped to iterate among the sub-groups. 
    
    ```
    rowThreadSum = sycl::reduce_over_group(tile32, rowThreadSum, sycl::plus<double>());
    ```

The migrated code snippet with `sshift_group_left` API can be replaced with `reduce_over_group` to get better performance. The reduce_over_group implements the generalized sum of the array elements internally by combining values held directly by the work-items in a group. The work-group reduces a number of values equal to the size of the group and each work-item provides one value.

#### Atomic operation optimization
   
    ```
     if (item_ct1.get_sub_group().get_local_linear_id() == 0) {
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
    }
    ```
    
The `atomic_fetch_add` operation calls automatically add on SYCL atomic object. Here, the atomic_fetch_add is used to sum all the subgroup values into rowThreadSum variable. This can be optimized by replacing the atomic_fetch_add with atomic_ref from sycl namespace.
   
    ```
    if (tile32.get_local_linear_id() == 0) {
       sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                  sycl:: access::address_space::generic_space>
        at_h_sum{b_shared[i % (ROWS_PER_CTA + 1)]};
        at_h_sum -= rowThreadSum;
    }
    ```
    
The `sycl::atomic_ref`, references to value of the object to be added. The result is then assigned to the value of the referenced object.

These optimization changes are performed in JacobiMethod and FinalError Kernels which can be found in `03_sycl_migrated_optimized` folder.

## Build and Run the `Jacobi CUDA Graphs` Sample

>  **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\Program Files (x86)\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see *[Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html)*

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake .. or ( cmake -D INTEL_MAX_GPU=1 .. ) or ( cmake -D NVIDIA_GPU=1 .. )
   $ make
   ```

   **Note**: By default, no flag are enabled during build which supports Intel® UHD Graphics, Intel® Gen9, Gen11, Xeon CPU. <br>
    Enable **INTEL_MAX_GPU** flag during build which supports Intel® Data Center GPU Max 1550 or 1100 to get optimized performace. <br>
    Enable **NVIDIA_GPU** flag during build which supports NVIDIA GPUs.([oneAPI for NVIDIA GPUs](https://developer.codeplay.com/products/oneapi/nvidia/) plugin   from Codeplay is required to build for NVIDIA GPUs ) <br>

   By default, this command sequence will build the `02_sycl_migrated` and `03_sycl_migrated_optimized` versions of the program.
   
3. Run the program.
   
   Run `02_sycl_migrated` on GPU.
   ```
   $ make run0 
   $ make run1
   ```   
   Run `02_sycl_migrated` for CPU.
    ```
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run0
    $ make run1
    $ unset ONEAPI_DEVICE_SELECTOR
    ```
    
   Run `03_sycl_migrated_optimized` on GPU.
   ```
   $ make run_smo0 
   $ make run_smo1
   ```   
   Run `03_sycl_migrated_optimized` for CPU.
    ```
    $ export ONEAPI_DEVICE_SELECTOR=opencl:cpu
    $ make run_smo0
    $ make run_smo1
    $ unset ONEAPI_DEVICE_SELECTOR
    ```
   run0 and run_smo0 will build the sample with Cuda Graph host function i.e. `JacobiMethodGpuCudaGraphExecKernelSetParams()` and run1 and run_smo1 will build the sample with `JacobiMethodGpu()` host function respectively.
   
#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
$ make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.
  
## Example Output

The following example is for `03_sycl_migrated_optimized` for GPU on **Intel(R) UHD Graphics P630 [0x3e96]**.
```
CPU iterations : 2954
CPU error : 4.988e-03
CPU Processing time: 213.076996 (ms)
Device iterations : 2954
Device error : 4.988e-03
Device Processing time: 1344.270020 (ms)
&&&& jacobiCudaGraphs PASSED
Built target run_smo0
```

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).