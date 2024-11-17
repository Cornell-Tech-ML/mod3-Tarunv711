# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Tasks 3.1 and 3.2
(.venv) (base) tarunvenkatasamy@Taruns-MacBook-Pro-3 mod3-Tarunv711 % python project/parallel_check.py

MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (163)

================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (163)
---------------------------------------------------------------------------|loop #ID
    def _map(                                                              |
        out: Storage,                                                      |
        out_shape: Shape,                                                  |
        out_strides: Strides,                                              |
        in_storage: Storage,                                               |
        in_shape: Shape,                                                   |
        in_strides: Strides,                                               |
    ) -> None:                                                             |
        # TODO: Implement for Task 3.1.                                    |
        # Check if tensors are stride-aligned                              |
        # Check for aligned tensors                                        |
        is_aligned = (                                                     |
            len(out_strides) == len(in_strides)                            |
            and np.array_equal(out_strides, in_strides)                    |
            and np.array_equal(out_shape, in_shape)                        |
        )                                                                  |
                                                                           |
        if is_aligned:                                                     |
            # Fast path for aligned tensors                                |
            for idx in prange(len(out)):-----------------------------------| #1
                out[idx] = fn(in_storage[idx])                             |
            return                                                         |
                                                                           |
        # Calculate total elements in output                               |
        total_elements = np.prod(out_shape)--------------------------------| #0
                                                                           |
        # Main parallel processing loop                                    |
        for pos in prange(total_elements):---------------------------------| #2
            # Initialize coordinate arrays                                 |
            out_coords = np.empty(len(out_shape), np.int32)                |
            in_coords = np.empty(len(in_shape), np.int32)                  |
                                                                           |
            # Convert linear position to coordinates                       |
            to_index(pos, out_shape, out_coords)                           |
                                                                           |
            # Map coordinates to storage positions                         |
            out_pos = index_to_position(out_coords, out_strides)           |
            broadcast_index(out_coords, out_shape, in_shape, in_coords)    |
            in_pos = index_to_position(in_coords, in_strides)              |
                                                                           |
            # Apply the function                                           |
            out[out_pos] = fn(in_storage[in_pos])                          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #1, #0, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (192)
 is hoisted out of the parallel loop labelled #2 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: out_coords = np.empty(len(out_shape), np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (193)
 is hoisted out of the parallel loop labelled #2 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: in_coords = np.empty(len(in_shape), np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (232)

================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (232)
------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                 |
        out: Storage,                                                         |
        out_shape: Shape,                                                     |
        out_strides: Strides,                                                 |
        a_storage: Storage,                                                   |
        a_shape: Shape,                                                       |
        a_strides: Strides,                                                   |
        b_storage: Storage,                                                   |
        b_shape: Shape,                                                       |
        b_strides: Strides,                                                   |
    ) -> None:                                                                |
        # TODO: Implement for Task 3.1.                                       |
        # Special case - when tensors are stride-aligned, avoid indexing      |
        # Check if tensors are stride-aligned                                 |
        if (                                                                  |
            len(out_strides) == len(a_strides) == len(b_strides)              |
            and np.array_equal(out_strides, a_strides)                        |
            and np.array_equal(out_strides, b_strides)                        |
            and np.array_equal(out_shape, a_shape)                            |
            and np.array_equal(out_shape, b_shape)                            |
        ):                                                                    |
            # Optimized path for stride-aligned tensors                       |
            for idx in prange(len(out)):--------------------------------------| #3
                out[idx] = fn(a_storage[idx], b_storage[idx])                 |
            return                                                            |
                                                                              |
        # Handle tensors with non-aligned strides                             |
        total_elements = 1                                                    |
        for dim_size in out_shape:                                            |
            total_elements *= dim_size                                        |
                                                                              |
        for linear_idx in prange(total_elements):-----------------------------| #4
            # Initialize index buffers for each tensor                        |
            output_indices = np.empty(len(out_shape), dtype=np.int32)         |
            a_indices = np.empty(len(a_shape), dtype=np.int32)                |
            b_indices = np.empty(len(b_shape), dtype=np.int32)                |
                                                                              |
            # Convert linear index to multi-dimensional indices               |
            to_index(linear_idx, out_shape, output_indices)                   |
                                                                              |
            # Compute the position in the output tensor                       |
            output_pos = index_to_position(output_indices, out_strides)       |
                                                                              |
            # Map output indices to corresponding indices in 'a' tensor       |
            broadcast_index(output_indices, out_shape, a_shape, a_indices)    |
            a_pos = index_to_position(a_indices, a_strides)                   |
                                                                              |
            # Map output indices to corresponding indices in 'b' tensor       |
            broadcast_index(output_indices, out_shape, b_shape, b_indices)    |
            b_pos = index_to_position(b_indices, b_strides)                   |
                                                                              |
            # Apply the binary function and store the result                  |
            out[output_pos] = fn(a_storage[a_pos], b_storage[b_pos])          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #3, #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (265)
 is hoisted out of the parallel loop labelled #4 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: output_indices = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (266)
 is hoisted out of the parallel loop labelled #4 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: a_indices = np.empty(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (267)
 is hoisted out of the parallel loop labelled #4 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: b_indices = np.empty(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (310)

================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (310)
-------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                           |
        out: Storage,                                                                      |
        out_shape: Shape,                                                                  |
        out_strides: Strides,                                                              |
        a_storage: Storage,                                                                |
        a_shape: Shape,                                                                    |
        a_strides: Strides,                                                                |
        reduce_dim: int,                                                                   |
    ) -> None:                                                                             |
        # TODO: Implement for Task 3.1.                                                    |
        # Calculate output size                                                            |
        dim_count = len(out_shape)                                                         |
        total_elements = 1                                                                 |
        for dim in out_shape:                                                              |
            total_elements *= dim                                                          |
                                                                                           |
        # Main parallel loop over each element in the output tensor                        |
        for linear_idx in prange(total_elements):------------------------------------------| #6
            # Initialize index buffers for output and input tensors                        |
            output_indices = np.empty(dim_count, dtype=np.int32)                           |
            input_indices = np.empty(dim_count, dtype=np.int32)                            |
                                                                                           |
            # Convert linear index to multi-dimensional indices for output                 |
            to_index(linear_idx, out_shape, output_indices)                                |
                                                                                           |
            # Determine the position in the output tensor                                  |
            output_position = index_to_position(output_indices, out_strides)               |
                                                                                           |
            # Copy the output indices to input indices for reduction                       |
            input_indices[:] = output_indices[:]-------------------------------------------| #5
                                                                                           |
            # Initialize reduction with the first element along the reduction dimension    |
            input_indices[reduce_dim] = 0                                                  |
            initial_pos = index_to_position(input_indices, a_strides)                      |
            accumulated = a_storage[initial_pos]                                           |
                                                                                           |
            # Perform reduction by iterating over the specified dimension                  |
            for dim_val in range(1, a_shape[reduce_dim]):                                  |
                input_indices[reduce_dim] = dim_val                                        |
                current_pos = index_to_position(input_indices, a_strides)                  |
                # Apply the reduction function                                             |
                accumulated = fn(accumulated, a_storage[current_pos])                      |
                                                                                           |
            # Store the reduced result in the output tensor                                |
            out[output_position] = accumulated                                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--6 is a parallel loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--5 (serial)



Parallel region 0 (loop #6) had 0 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#6).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (329)
 is hoisted out of the parallel loop labelled #6 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: output_indices = np.empty(dim_count, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (330)
 is hoisted out of the parallel loop labelled #6 (it will be performed before
the loop is executed and reused inside the loop):
   Allocation:: input_indices = np.empty(dim_count, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (359)

================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/tarunvenkatasamy/Documents/MLE/mod3-Tarunv711/minitorch/fast_ops.py (359)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    # TODO: Implement for Task 3.2.                                                       |
    for i in prange(out_shape[0]):--------------------------------------------------------| #7
        for j in range(out_shape[1]):                                                     |
            for k in range(out_shape[2]):                                                 |
                # Initialize accumulator for dot product                                  |
                acc = 0.0                                                                 |
                # Compute dot product along shared dimension                              |
                for l in range(a_shape[-1]):                                              |
                    # Get positions in a and b storage                                    |
                    a_pos = i * a_batch_stride + j * a_strides[1] + l * a_strides[2]      |
                    b_pos = i * b_batch_stride + l * b_strides[1] + k * b_strides[2]      |
                    # Multiply and accumulate                                             |
                    acc += a_storage[a_pos] * b_storage[b_pos]                            |
                # Write result to output                                                  |
                out_pos = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]    |
                out[out_pos] = acc                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #7).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

# Task 3.5 - CPU Outputs
## python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
Epoch  0  loss  5.533565499894529 correct 43 avg time per epoch: 4.5771s
Epoch  10  loss  2.1655565255178697 correct 48 avg time per epoch: 1.2961s
Epoch  20  loss  2.4961986718064706 correct 49 avg time per epoch: 1.2897s
Epoch  30  loss  0.8784705797891394 correct 49 avg time per epoch: 1.2874s
Epoch  40  loss  0.2597141934873494 correct 50 avg time per epoch: 1.2696s
Epoch  50  loss  0.23213725826678505 correct 49 avg time per epoch: 1.2795s
Epoch  60  loss  0.7645277537435335 correct 49 avg time per epoch: 1.2617s
Epoch  70  loss  0.7274452811518857 correct 50 avg time per epoch: 1.2613s
Epoch  80  loss  0.9273933714776785 correct 49 avg time per epoch: 1.2882s
Epoch  90  loss  0.4519753601954643 correct 49 avg time per epoch: 1.2787s
Epoch  100  loss  0.043626572981468825 correct 49 avg time per epoch: 1.2615s
Epoch  110  loss  0.394213196851283 correct 50 avg time per epoch: 1.2535s
Epoch  120  loss  0.24044872165936196 correct 50 avg time per epoch: 1.2622s
Epoch  130  loss  0.21720678253348513 correct 50 avg time per epoch: 1.2635s
Epoch  140  loss  0.8344466623375164 correct 50 avg time per epoch: 1.2559s
Epoch  150  loss  0.15657952485964838 correct 50 avg time per epoch: 1.2362s
Epoch  160  loss  0.12149656040435658 correct 50 avg time per epoch: 1.2476s
Epoch  170  loss  0.2586945939886978 correct 50 avg time per epoch: 1.2362s
Epoch  180  loss  0.11607260486576464 correct 50 avg time per epoch: 1.2563s
Epoch  190  loss  0.04884709423628389 correct 49 avg time per epoch: 1.2537s
Epoch  200  loss  0.07384132874609149 correct 50 avg time per epoch: 1.2504s
Epoch  210  loss  0.7406287630403705 correct 50 avg time per epoch: 1.2587s
Epoch  220  loss  0.2378970828143555 correct 50 avg time per epoch: 1.2503s
Epoch  230  loss  0.05075708680535715 correct 50 avg time per epoch: 1.2709s
Epoch  240  loss  0.872140516012356 correct 50 avg time per epoch: 1.2556s
Epoch  250  loss  0.10750377464924893 correct 50 avg time per epoch: 1.2814s
Epoch  260  loss  0.1116891497359503 correct 50 avg time per epoch: 1.2596s
Epoch  270  loss  0.701844475842956 correct 50 avg time per epoch: 1.2469s
Epoch  280  loss  0.0218266992237432 correct 50 avg time per epoch: 1.2595s
Epoch  290  loss  0.027517122997636645 correct 50 avg time per epoch: 1.2671s
Epoch  300  loss  0.12300801198306432 correct 50 avg time per epoch: 1.2586s
Epoch  310  loss  0.029119635919842193 correct 50 avg time per epoch: 1.2709s
Epoch  320  loss  0.6370334354748823 correct 50 avg time per epoch: 1.2663s
Epoch  330  loss  0.6023393760073735 correct 50 avg time per epoch: 1.2663s
Epoch  340  loss  0.05701845241910724 correct 50 avg time per epoch: 1.2794s
Epoch  350  loss  0.6718765778871549 correct 50 avg time per epoch: 1.2624s
Epoch  360  loss  0.014051560505247333 correct 50 avg time per epoch: 1.2490s
Epoch  370  loss  0.500370008422024 correct 50 avg time per epoch: 1.2539s
Epoch  380  loss  0.16030951640954436 correct 50 avg time per epoch: 1.2570s
Epoch  390  loss  0.1494409866682568 correct 50 avg time per epoch: 1.2634s
Epoch  400  loss  0.09058571431214169 correct 50 avg time per epoch: 1.2594s
Epoch  410  loss  0.0032555963736699005 correct 49 avg time per epoch: 1.2503s
Epoch  420  loss  0.5993408997954539 correct 50 avg time per epoch: 1.2626s
Epoch  430  loss  0.00016965536322709467 correct 50 avg time per epoch: 1.2539s
Epoch  440  loss  0.4821977730920872 correct 50 avg time per epoch: 1.2581s
Epoch  450  loss  0.017225385631566472 correct 50 avg time per epoch: 1.2643s
Epoch  460  loss  0.029040186719832033 correct 50 avg time per epoch: 1.2573s
Epoch  470  loss  0.002206985622759047 correct 50 avg time per epoch: 1.2636s
Epoch  480  loss  0.5730447682097214 correct 50 avg time per epoch: 1.2664s
Epoch  490  loss  0.008754438472063844 correct 50 avg time per epoch: 1.2531s

## python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  6.617755445159329 correct 32 avg time per epoch: 4.0089s
Epoch  10  loss  4.49119766339581 correct 41 avg time per epoch: 1.2643s
Epoch  20  loss  3.0936294757449 correct 41 avg time per epoch: 1.2731s
Epoch  30  loss  3.6505540466457043 correct 47 avg time per epoch: 1.2638s
Epoch  40  loss  2.6344858198408287 correct 46 avg time per epoch: 1.2615s
Epoch  50  loss  3.1934115808664725 correct 47 avg time per epoch: 1.2699s
Epoch  60  loss  1.8748317990467087 correct 49 avg time per epoch: 1.3212s
Epoch  70  loss  1.8877349702527746 correct 49 avg time per epoch: 1.2669s
Epoch  80  loss  2.139839943058951 correct 49 avg time per epoch: 1.2625s
Epoch  90  loss  1.0915820883999099 correct 49 avg time per epoch: 1.2610s
Epoch  100  loss  1.4276092901127682 correct 49 avg time per epoch: 1.2663s
Epoch  110  loss  0.7257739463443104 correct 49 avg time per epoch: 1.2570s
Epoch  120  loss  0.4126375500470123 correct 50 avg time per epoch: 1.2551s
Epoch  130  loss  1.760107889109824 correct 49 avg time per epoch: 1.2595s
Epoch  140  loss  1.472048128151201 correct 50 avg time per epoch: 1.2434s
Epoch  150  loss  0.7279546706748402 correct 50 avg time per epoch: 1.2643s
Epoch  160  loss  0.9215779455174813 correct 49 avg time per epoch: 1.2445s
Epoch  170  loss  0.5418970969952266 correct 50 avg time per epoch: 1.2472s
Epoch  180  loss  0.41731602817344643 correct 50 avg time per epoch: 1.2563s
Epoch  190  loss  0.370643875042311 correct 49 avg time per epoch: 1.2473s
Epoch  200  loss  0.7154424227357099 correct 50 avg time per epoch: 1.2470s
Epoch  210  loss  0.9317011030041005 correct 50 avg time per epoch: 1.2538s
Epoch  220  loss  0.5372088507579457 correct 50 avg time per epoch: 1.2560s
Epoch  230  loss  0.49247906277964937 correct 50 avg time per epoch: 1.2794s
Epoch  240  loss  0.7054565626611377 correct 50 avg time per epoch: 1.2524s
Epoch  250  loss  0.22278010423271838 correct 50 avg time per epoch: 1.2723s
Epoch  260  loss  0.2403546703015368 correct 50 avg time per epoch: 1.2672s
Epoch  270  loss  0.6982808404766723 correct 50 avg time per epoch: 1.2694s
Epoch  280  loss  0.4374622476973729 correct 50 avg time per epoch: 1.2625s
Epoch  290  loss  1.1098143130257112 correct 49 avg time per epoch: 1.2659s
Epoch  300  loss  0.7286053488341413 correct 50 avg time per epoch: 1.2604s
Epoch  310  loss  0.39830744951645136 correct 50 avg time per epoch: 1.2680s
Epoch  320  loss  0.5389271536751022 correct 50 avg time per epoch: 1.2652s
Epoch  330  loss  0.3239024077820304 correct 50 avg time per epoch: 1.2626s
Epoch  340  loss  0.03732540728905802 correct 50 avg time per epoch: 1.2760s
Epoch  350  loss  0.515300935799844 correct 50 avg time per epoch: 1.2689s
Epoch  360  loss  0.8678538205153172 correct 50 avg time per epoch: 1.2833s
Epoch  370  loss  0.4420937346848898 correct 50 avg time per epoch: 1.2720s
Epoch  380  loss  0.18854954888343992 correct 50 avg time per epoch: 1.2557s
Epoch  390  loss  0.26484372158263325 correct 50 avg time per epoch: 1.2683s
Epoch  400  loss  0.20867353761125096 correct 50 avg time per epoch: 1.2599s
Epoch  410  loss  0.35697524644068945 correct 50 avg time per epoch: 1.2634s
Epoch  420  loss  0.3152521960252655 correct 50 avg time per epoch: 1.2590s
Epoch  430  loss  0.08749529310191931 correct 50 avg time per epoch: 1.2476s
Epoch  440  loss  0.010232274203263555 correct 50 avg time per epoch: 1.2702s
Epoch  450  loss  0.12611896093368585 correct 50 avg time per epoch: 1.2584s
Epoch  460  loss  0.4185852103601645 correct 50 avg time per epoch: 1.2593s
Epoch  470  loss  0.03817434934902303 correct 50 avg time per epoch: 1.2689s
Epoch  480  loss  0.06288700500220423 correct 50 avg time per epoch: 1.2744s
Epoch  490  loss  0.8294344774296897 correct 50 avg time per epoch: 1.2642s

## python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch  0  loss  10.246759653413235 correct 23 avg time per epoch: 4.8070s
Epoch  10  loss  5.174417176255028 correct 42 avg time per epoch: 1.2834s
Epoch  20  loss  3.2149214326850677 correct 43 avg time per epoch: 1.2719s
Epoch  30  loss  3.699254578170534 correct 43 avg time per epoch: 1.2855s
Epoch  40  loss  2.0032448608565776 correct 43 avg time per epoch: 1.2702s
Epoch  50  loss  4.052028366708647 correct 45 avg time per epoch: 1.2659s
Epoch  60  loss  1.5816528254855058 correct 44 avg time per epoch: 1.2524s
Epoch  70  loss  2.8585883043453295 correct 44 avg time per epoch: 1.2521s
Epoch  80  loss  1.5814092078649438 correct 46 avg time per epoch: 1.2733s
Epoch  90  loss  1.399501957563643 correct 46 avg time per epoch: 1.2579s
Epoch  100  loss  0.7721997821988897 correct 46 avg time per epoch: 1.2564s
Epoch  110  loss  0.9025854537334451 correct 46 avg time per epoch: 1.2551s
Epoch  120  loss  3.5583996577358614 correct 47 avg time per epoch: 1.2818s
Epoch  130  loss  3.90874563762993 correct 48 avg time per epoch: 1.2748s
Epoch  140  loss  1.2131712304698388 correct 48 avg time per epoch: 1.2653s
Epoch  150  loss  2.6541069378451008 correct 47 avg time per epoch: 1.2789s
Epoch  160  loss  1.1652547100676025 correct 48 avg time per epoch: 1.2569s
Epoch  170  loss  1.1873016785428951 correct 48 avg time per epoch: 1.2517s
Epoch  180  loss  0.704603144333939 correct 48 avg time per epoch: 1.2697s
Epoch  190  loss  1.2132969042448476 correct 50 avg time per epoch: 1.2703s
Epoch  200  loss  0.7582919728350266 correct 48 avg time per epoch: 1.2781s
Epoch  210  loss  0.7103816491433008 correct 48 avg time per epoch: 1.2632s
Epoch  220  loss  0.49940648415294087 correct 50 avg time per epoch: 1.2674s
Epoch  230  loss  1.4328350056123123 correct 50 avg time per epoch: 1.2694s
Epoch  240  loss  0.6599997434402669 correct 50 avg time per epoch: 1.2715s
Epoch  250  loss  0.27328079334703204 correct 50 avg time per epoch: 1.2838s
Epoch  260  loss  1.1392352412692874 correct 50 avg time per epoch: 1.2771s
Epoch  270  loss  0.24594444369740787 correct 50 avg time per epoch: 1.2683s
Epoch  280  loss  0.30431342305378944 correct 49 avg time per epoch: 1.2710s
Epoch  290  loss  0.3345754763817068 correct 49 avg time per epoch: 1.2576s
Epoch  300  loss  0.9741602745629578 correct 50 avg time per epoch: 1.2590s
Epoch  310  loss  1.766504753070036 correct 49 avg time per epoch: 1.2624s
Epoch  320  loss  0.9073386383370142 correct 50 avg time per epoch: 1.2655s
Epoch  330  loss  0.9988557390501127 correct 50 avg time per epoch: 1.2593s
Epoch  340  loss  0.13015125280190026 correct 50 avg time per epoch: 1.2585s
Epoch  350  loss  0.6493249047725368 correct 50 avg time per epoch: 1.2647s
Epoch  360  loss  0.33950773121729827 correct 49 avg time per epoch: 1.2867s
Epoch  370  loss  0.14542560778683075 correct 50 avg time per epoch: 1.2661s
Epoch  380  loss  0.9686891228554452 correct 50 avg time per epoch: 1.2600s
Epoch  390  loss  0.2459824919747293 correct 50 avg time per epoch: 1.2521s
Epoch  400  loss  0.13354879693889862 correct 50 avg time per epoch: 1.2463s
Epoch  410  loss  0.31661251523831196 correct 50 avg time per epoch: 1.2738s
Epoch  420  loss  0.17794500112661368 correct 50 avg time per epoch: 1.2564s
Epoch  430  loss  0.33818520419116505 correct 50 avg time per epoch: 1.2598s
Epoch  440  loss  0.9173929401001122 correct 50 avg time per epoch: 1.2683s
Epoch  450  loss  0.23962150584358982 correct 50 avg time per epoch: 1.2736s
Epoch  460  loss  0.9268432331444307 correct 50 avg time per epoch: 1.2620s
Epoch  470  loss  0.8180670027046285 correct 50 avg time per epoch: 1.2741s
Epoch  480  loss  0.07856701808024816 correct 50 avg time per epoch: 1.2813s
Epoch  490  loss  0.6635717783029357 correct 50 avg time per epoch: 1.2697s

# Task 3.5 - GPU Outputs
## python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
Epoch  0  loss  6.057312553568347 correct 45 avg time per epoch: 6.5772s
Epoch  10  loss  1.7776234839147493 correct 48 avg time per epoch: 0.0817s
Epoch  20  loss  1.2800492988226033 correct 48 avg time per epoch: 0.0819s
Epoch  30  loss  0.5617424552137325 correct 50 avg time per epoch: 0.0822s
Epoch  40  loss  0.9300645435230771 correct 50 avg time per epoch: 0.0807s
Epoch  50  loss  0.2968124095479543 correct 50 avg time per epoch: 0.0814s
Epoch  60  loss  1.8836307296175343 correct 50 avg time per epoch: 0.0872s
Epoch  70  loss  0.5277447137894133 correct 50 avg time per epoch: 0.0859s
Epoch  80  loss  0.7503305466728207 correct 50 avg time per epoch: 0.0839s
Epoch  90  loss  0.6432790368477371 correct 50 avg time per epoch: 0.0864s
Epoch  100  loss  0.6401483463068449 correct 50 avg time per epoch: 0.0821s
Epoch  110  loss  0.35678214522655843 correct 50 avg time per epoch: 0.0811s
Epoch  120  loss  0.7654748500228327 correct 50 avg time per epoch: 0.0845s
Epoch  130  loss  0.24099332681593702 correct 50 avg time per epoch: 0.0823s
Epoch  140  loss  0.3008283609557267 correct 50 avg time per epoch: 0.0811s
Epoch  150  loss  0.6450953751778623 correct 50 avg time per epoch: 0.0814s
Epoch  160  loss  0.34552503971570736 correct 50 avg time per epoch: 0.0825s
Epoch  170  loss  0.6664577790304881 correct 50 avg time per epoch: 0.0815s
Epoch  180  loss  0.28342355128619545 correct 50 avg time per epoch: 0.0815s
Epoch  190  loss  0.013060581425156956 correct 50 avg time per epoch: 0.0804s
Epoch  200  loss  0.20630382851850051 correct 50 avg time per epoch: 0.0887s
Epoch  210  loss  0.761928314371118 correct 50 avg time per epoch: 0.0833s
Epoch  220  loss  0.24867504504502574 correct 50 avg time per epoch: 0.0812s
Epoch  230  loss  0.09116149280066153 correct 50 avg time per epoch: 0.0836s
Epoch  240  loss  0.5819397414844847 correct 50 avg time per epoch: 0.0797s
Epoch  250  loss  0.10221333308116763 correct 50 avg time per epoch: 0.0813s
Epoch  260  loss  0.13425607005066378 correct 50 avg time per epoch: 0.0810s
Epoch  270  loss  0.7947387203603681 correct 50 avg time per epoch: 0.0820s
Epoch  280  loss  0.0035484054414386887 correct 50 avg time per epoch: 0.0824s
Epoch  290  loss  0.20375266069516113 correct 50 avg time per epoch: 0.0825s
Epoch  300  loss  0.2814799864955206 correct 50 avg time per epoch: 0.0799s
Epoch  310  loss  0.17095476371220647 correct 50 avg time per epoch: 0.0854s
Epoch  320  loss  0.31111945148251696 correct 50 avg time per epoch: 0.0812s
Epoch  330  loss  0.030421764242106717 correct 50 avg time per epoch: 0.0837s
Epoch  340  loss  0.43927513075077895 correct 50 avg time per epoch: 0.0846s
Epoch  350  loss  0.5630745611727873 correct 50 avg time per epoch: 0.0830s
Epoch  360  loss  0.13953939862489398 correct 50 avg time per epoch: 0.0833s
Epoch  370  loss  0.03624731691413122 correct 50 avg time per epoch: 0.0813s
Epoch  380  loss  0.1721503841778311 correct 50 avg time per epoch: 0.0825s
Epoch  390  loss  0.1918030242786825 correct 50 avg time per epoch: 0.0855s
Epoch  400  loss  0.08270168438482671 correct 50 avg time per epoch: 0.0811s
Epoch  410  loss  0.2894589564603278 correct 50 avg time per epoch: 0.0819s
Epoch  420  loss  0.3738018162957238 correct 50 avg time per epoch: 0.0829s
Epoch  430  loss  0.025062649208110386 correct 50 avg time per epoch: 0.0823s
Epoch  440  loss  0.35964192976015663 correct 50 avg time per epoch: 0.0820s
Epoch  450  loss  0.05459934817530681 correct 50 avg time per epoch: 0.0831s
Epoch  460  loss  0.08498086562209155 correct 50 avg time per epoch: 0.0809s
Epoch  470  loss  0.361999468076002 correct 50 avg time per epoch: 0.0855s
Epoch  480  loss  0.06094922548106635 correct 50 avg time per epoch: 0.0821s
Epoch  490  loss  0.1658993960999017 correct 50 avg time per epoch: 0.0820s

## python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Epoch  0  loss  5.410839629191172 correct 28 avg time per epoch: 6.3426s
Epoch  10  loss  6.240884936614521 correct 44 avg time per epoch: 0.0886s
Epoch  20  loss  4.684715456683926 correct 40 avg time per epoch: 0.0826s
Epoch  30  loss  4.639672132914053 correct 47 avg time per epoch: 0.0833s
Epoch  40  loss  3.0175456026366807 correct 49 avg time per epoch: 0.0904s
Epoch  50  loss  1.7863427298786119 correct 50 avg time per epoch: 0.0806s
Epoch  60  loss  1.7097204671101287 correct 50 avg time per epoch: 0.0811s
Epoch  70  loss  1.4908214981793984 correct 50 avg time per epoch: 0.0814s
Epoch  80  loss  1.0227393895116372 correct 50 avg time per epoch: 0.0842s
Epoch  90  loss  1.7688897145937312 correct 49 avg time per epoch: 0.0856s
Epoch  100  loss  1.2061357316136432 correct 50 avg time per epoch: 0.0908s
Epoch  110  loss  0.6714086289506107 correct 50 avg time per epoch: 0.0853s
Epoch  120  loss  0.3985145827346553 correct 50 avg time per epoch: 0.0943s
Epoch  130  loss  0.28462732195800833 correct 50 avg time per epoch: 0.1139s
Epoch  140  loss  0.5331998330019312 correct 50 avg time per epoch: 0.0934s
Epoch  150  loss  0.8571541989354877 correct 50 avg time per epoch: 0.0851s
Epoch  160  loss  0.2772147994608994 correct 50 avg time per epoch: 0.0824s
Epoch  170  loss  0.1245028877067547 correct 50 avg time per epoch: 0.1052s
Epoch  180  loss  0.568791375030449 correct 50 avg time per epoch: 0.0858s
Epoch  190  loss  0.5689024391704928 correct 50 avg time per epoch: 0.0920s
Epoch  200  loss  0.7751051309723404 correct 50 avg time per epoch: 0.0820s
Epoch  210  loss  0.2190204666243691 correct 50 avg time per epoch: 0.0834s
Epoch  220  loss  0.2827226861140963 correct 50 avg time per epoch: 0.0811s
Epoch  230  loss  0.5064501074408511 correct 50 avg time per epoch: 0.0808s
Epoch  240  loss  0.12691835989630088 correct 50 avg time per epoch: 0.0813s
Epoch  250  loss  0.1291546248573032 correct 50 avg time per epoch: 0.0813s
Epoch  260  loss  0.49812852931193435 correct 50 avg time per epoch: 0.0808s
Epoch  270  loss  0.490592102399403 correct 50 avg time per epoch: 0.0812s
Epoch  280  loss  0.3818009853013752 correct 50 avg time per epoch: 0.0819s
Epoch  290  loss  0.1647197676955553 correct 50 avg time per epoch: 0.0807s
Epoch  300  loss  0.6128427148877803 correct 50 avg time per epoch: 0.0813s
Epoch  310  loss  0.24075055274306553 correct 50 avg time per epoch: 0.0811s
Epoch  320  loss  0.5225620909705793 correct 50 avg time per epoch: 0.0817s
Epoch  330  loss  0.3890803774761218 correct 50 avg time per epoch: 0.0813s
Epoch  340  loss  0.46978720668719925 correct 50 avg time per epoch: 0.0834s
Epoch  350  loss  0.603051179273261 correct 50 avg time per epoch: 0.0839s
Epoch  360  loss  0.20441043460919484 correct 50 avg time per epoch: 0.0806s
Epoch  370  loss  0.03309434180005307 correct 50 avg time per epoch: 0.0808s
Epoch  380  loss  0.2879313255940808 correct 50 avg time per epoch: 0.0810s
Epoch  390  loss  0.05662477658818739 correct 50 avg time per epoch: 0.0841s
Epoch  400  loss  0.292336797585935 correct 50 avg time per epoch: 0.0810s
Epoch  410  loss  0.22295834379869123 correct 50 avg time per epoch: 0.0805s
Epoch  420  loss  0.15141797703548746 correct 50 avg time per epoch: 0.0802s
Epoch  430  loss  0.33067267176031245 correct 50 avg time per epoch: 0.0806s
Epoch  440  loss  0.2478165832114391 correct 50 avg time per epoch: 0.0801s
Epoch  450  loss  0.020976661888306238 correct 50 avg time per epoch: 0.0804s
Epoch  460  loss  0.2277984037316834 correct 50 avg time per epoch: 0.0815s
Epoch  470  loss  0.06307964360698143 correct 50 avg time per epoch: 0.0836s
Epoch  480  loss  0.15042305960249505 correct 50 avg time per epoch: 0.0831s
Epoch  490  loss  0.22460187506241097 correct 50 avg time per epoch: 0.0811s

## python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
Epoch  0  loss  9.536617332365866 correct 25 avg time per epoch: 6.2944s
Epoch  10  loss  3.8438000550486193 correct 38 avg time per epoch: 0.0851s
Epoch  20  loss  4.627829939785186 correct 44 avg time per epoch: 0.0920s
Epoch  30  loss  4.254780826749385 correct 46 avg time per epoch: 0.0945s
Epoch  40  loss  4.9283816214516385 correct 46 avg time per epoch: 0.0867s
Epoch  50  loss  3.4558697872557333 correct 46 avg time per epoch: 0.0876s
Epoch  60  loss  2.662358682297608 correct 45 avg time per epoch: 0.0991s
Epoch  70  loss  3.2619440602183527 correct 46 avg time per epoch: 0.0818s
Epoch  80  loss  2.3357048776642304 correct 46 avg time per epoch: 0.0803s
Epoch  90  loss  2.6930129386827977 correct 49 avg time per epoch: 0.0822s
Epoch  100  loss  2.7627074962155973 correct 50 avg time per epoch: 0.0815s
Epoch  110  loss  1.053332950945864 correct 48 avg time per epoch: 0.0880s
Epoch  120  loss  2.0926683100034253 correct 43 avg time per epoch: 0.0816s
Epoch  130  loss  1.6186383485160798 correct 45 avg time per epoch: 0.0818s
Epoch  140  loss  1.3377167908183287 correct 49 avg time per epoch: 0.0802s
Epoch  150  loss  1.5823783417503203 correct 48 avg time per epoch: 0.0816s
Epoch  160  loss  0.47019946868840373 correct 48 avg time per epoch: 0.0809s
Epoch  170  loss  1.451974748773158 correct 49 avg time per epoch: 0.0810s
Epoch  180  loss  0.602402825031838 correct 50 avg time per epoch: 0.0808s
Epoch  190  loss  1.2586743446887538 correct 49 avg time per epoch: 0.0805s
Epoch  200  loss  0.723821055260626 correct 50 avg time per epoch: 0.0800s
Epoch  210  loss  0.17174803566057661 correct 50 avg time per epoch: 0.0807s
Epoch  220  loss  0.839861676521363 correct 50 avg time per epoch: 0.0815s
Epoch  230  loss  0.8329503518586886 correct 50 avg time per epoch: 0.0810s
Epoch  240  loss  1.4704969832421344 correct 50 avg time per epoch: 0.0829s
Epoch  250  loss  0.5938778624396001 correct 50 avg time per epoch: 0.0831s
Epoch  260  loss  0.678764286432185 correct 50 avg time per epoch: 0.0810s
Epoch  270  loss  0.49167882426905857 correct 49 avg time per epoch: 0.0803s
Epoch  280  loss  1.1820241542468353 correct 50 avg time per epoch: 0.0808s
Epoch  290  loss  0.43157822186237393 correct 50 avg time per epoch: 0.0816s
Epoch  300  loss  1.1856670559619584 correct 50 avg time per epoch: 0.0809s
Epoch  310  loss  0.8170498261235172 correct 50 avg time per epoch: 0.0810s
Epoch  320  loss  1.0788862813701807 correct 50 avg time per epoch: 0.0812s
Epoch  330  loss  1.216144510302983 correct 49 avg time per epoch: 0.0937s
Epoch  340  loss  1.0543235310181487 correct 49 avg time per epoch: 0.0808s
Epoch  350  loss  1.1387101978956047 correct 49 avg time per epoch: 0.0842s
Epoch  360  loss  0.3953623811916506 correct 50 avg time per epoch: 0.0810s
Epoch  370  loss  0.6189514117215834 correct 50 avg time per epoch: 0.0820s
Epoch  380  loss  0.40669682151011066 correct 50 avg time per epoch: 0.0839s
Epoch  390  loss  0.792139736809746 correct 49 avg time per epoch: 0.0807s
Epoch  400  loss  0.6072307805781909 correct 50 avg time per epoch: 0.0805s
Epoch  410  loss  0.16827579855605054 correct 50 avg time per epoch: 0.0807s
Epoch  420  loss  0.20737374114364487 correct 50 avg time per epoch: 0.0810s
Epoch  430  loss  0.21061278185794063 correct 50 avg time per epoch: 0.0814s
Epoch  440  loss  0.5355082236550701 correct 50 avg time per epoch: 0.0810s
Epoch  450  loss  0.17516699277034709 correct 50 avg time per epoch: 0.0817s
Epoch  460  loss  0.2186968706833063 correct 50 avg time per epoch: 0.0800s
Epoch  470  loss  0.3918296146722191 correct 50 avg time per epoch: 0.0814s
Epoch  480  loss  0.3296734005781029 correct 50 avg time per epoch: 0.0805s
Epoch  490  loss  0.22282318415060448 correct 50 avg time per epoch: 0.0805s

# Task 3.5 - Larger hidden layer outputs - GPU
## python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 500 --DATASET simple --RATE 0.05
Epoch  0  loss  152.14234001578464 correct 47 avg time per epoch: 6.6061s
Epoch  10  loss  0.15573739740574583 correct 47 avg time per epoch: 0.3851s
Epoch  20  loss  0.5862945548879164 correct 50 avg time per epoch: 0.4060s
Epoch  30  loss  0.019849524539314416 correct 50 avg time per epoch: 0.3911s
Epoch  40  loss  0.2727153072915381 correct 50 avg time per epoch: 0.3861s
Epoch  50  loss  0.13087850542860913 correct 50 avg time per epoch: 0.3901s
Epoch  60  loss  0.015300411912938119 correct 50 avg time per epoch: 0.3835s
Epoch  70  loss  0.001776678391142428 correct 50 avg time per epoch: 0.4134s
Epoch  80  loss  0.03426133190763209 correct 50 avg time per epoch: 0.3907s
Epoch  90  loss  0.011477582203112876 correct 50 avg time per epoch: 0.3955s
Epoch  100  loss  0.00045097980686314824 correct 50 avg time per epoch: 0.3901s
Epoch  110  loss  0.0095034177169406 correct 50 avg time per epoch: 0.3911s
Epoch  120  loss  0.0010101752934985723 correct 50 avg time per epoch: 0.3857s
Epoch  130  loss  0.06428557965903281 correct 50 avg time per epoch: 0.3900s
Epoch  140  loss  0.015573654343975474 correct 50 avg time per epoch: 0.3888s
Epoch  150  loss  0.029206246792839823 correct 50 avg time per epoch: 0.3926s
Epoch  160  loss  0.14260591500354883 correct 50 avg time per epoch: 0.3924s
Epoch  170  loss  0.023894119408985014 correct 50 avg time per epoch: 0.3889s
Epoch  180  loss  0.024268072536303857 correct 50 avg time per epoch: 0.3980s
Epoch  190  loss  0.10506677228391033 correct 50 avg time per epoch: 0.3933s
Epoch  200  loss  0.08919960588315176 correct 50 avg time per epoch: 0.3890s
Epoch  210  loss  0.059181529872799885 correct 50 avg time per epoch: 0.3897s
Epoch  220  loss  0.022026656676914422 correct 50 avg time per epoch: 0.4104s
Epoch  230  loss  0.0393139042871153 correct 50 avg time per epoch: 0.3942s
Epoch  240  loss  0.09791259530832257 correct 50 avg time per epoch: 0.3946s
Epoch  250  loss  1.9194460460356746e-05 correct 50 avg time per epoch: 0.3911s
Epoch  260  loss  0.009741412918239248 correct 50 avg time per epoch: 0.3916s
Epoch  270  loss  0.029793748214661638 correct 50 avg time per epoch: 0.3948s
Epoch  280  loss  0.009061623108801245 correct 50 avg time per epoch: 0.3899s
Epoch  290  loss  0.014755030247931561 correct 50 avg time per epoch: 0.3920s
Epoch  300  loss  0.1205196692156317 correct 50 avg time per epoch: 0.3939s
Epoch  310  loss  0.06831239944414981 correct 50 avg time per epoch: 0.3935s
Epoch  320  loss  0.036481474373381305 correct 50 avg time per epoch: 0.3944s
Epoch  330  loss  3.0182946682649183e-07 correct 50 avg time per epoch: 0.3931s
Epoch  340  loss  0.0972299231391897 correct 50 avg time per epoch: 0.4248s
Epoch  350  loss  0.01989018181830129 correct 50 avg time per epoch: 0.4182s
Epoch  360  loss  0.025344173076427647 correct 50 avg time per epoch: 0.4106s
Epoch  370  loss  0.05864596298221252 correct 50 avg time per epoch: 0.4331s
Epoch  380  loss  0.028605458854776777 correct 50 avg time per epoch: 0.4145s
Epoch  390  loss  0.05142992341076641 correct 50 avg time per epoch: 0.4135s
Epoch  400  loss  0.018795965565060045 correct 50 avg time per epoch: 0.4148s
Epoch  410  loss  0.00045414492686740473 correct 50 avg time per epoch: 0.4117s
Epoch  420  loss  0.01840676409021123 correct 50 avg time per epoch: 0.4115s
Epoch  430  loss  0.06837353835501254 correct 50 avg time per epoch: 0.4197s
Epoch  440  loss  0.0471937887429897 correct 50 avg time per epoch: 0.4148s
Epoch  450  loss  0.012274590652424243 correct 50 avg time per epoch: 0.4146s
Epoch  460  loss  0.01278163511576873 correct 50 avg time per epoch: 0.4109s
Epoch  470  loss  0.06245979614528634 correct 50 avg time per epoch: 0.4118s
Epoch  480  loss  0.06305766921765124 correct 50 avg time per epoch: 0.4135s
Epoch  490  loss  0.0003788075554631194 correct 50 avg time per epoch: 0.4116s

# Task 3.5 - Larger hidden layer outputs - CPU
## python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET simple --RATE 0.05
Epoch  0  loss  49.73579860839248 correct 34 avg time per epoch: 4.6195s
Epoch  10  loss  2.181056898162607 correct 43 avg time per epoch: 1.9186s
Epoch  20  loss  0.7616881537302898 correct 48 avg time per epoch: 1.8340s
Epoch  30  loss  0.6834099481966746 correct 49 avg time per epoch: 1.9180s
Epoch  40  loss  0.007545068403548223 correct 49 avg time per epoch: 1.8183s
Epoch  50  loss  0.0004368768287100679 correct 50 avg time per epoch: 1.9160s
Epoch  60  loss  1.330972891608014 correct 50 avg time per epoch: 1.8552s
Epoch  70  loss  0.0290349883186655 correct 49 avg time per epoch: 1.9264s
Epoch  80  loss  0.0027986025440192726 correct 49 avg time per epoch: 1.8489s
Epoch  90  loss  0.39381362280351534 correct 50 avg time per epoch: 1.9271s
Epoch  100  loss  0.47312954453304196 correct 50 avg time per epoch: 1.8279s
Epoch  110  loss  0.2629435097955555 correct 50 avg time per epoch: 1.9237s
Epoch  120  loss  0.012949666623885045 correct 50 avg time per epoch: 1.8496s
Epoch  130  loss  0.3633359387533051 correct 50 avg time per epoch: 1.9154s
Epoch  140  loss  0.25202522332603416 correct 50 avg time per epoch: 1.8038s
Epoch  150  loss  0.1709183627120396 correct 50 avg time per epoch: 1.8876s
Epoch  160  loss  0.012166639550124784 correct 50 avg time per epoch: 1.8402s
Epoch  170  loss  0.044220443600506636 correct 50 avg time per epoch: 1.8644s
Epoch  180  loss  0.2237977377251114 correct 50 avg time per epoch: 1.8490s
Epoch  190  loss  0.001851053300776372 correct 50 avg time per epoch: 1.8684s
Epoch  200  loss  0.1724680999595229 correct 50 avg time per epoch: 1.8982s
Epoch  210  loss  0.03252100584384387 correct 50 avg time per epoch: 1.8237s
Epoch  220  loss  0.12736564763867475 correct 50 avg time per epoch: 1.8928s
Epoch  230  loss  0.012755672666467724 correct 50 avg time per epoch: 1.8243s
Epoch  240  loss  0.2589141826538479 correct 50 avg time per epoch: 1.8925s
Epoch  250  loss  0.0128948028608481 correct 50 avg time per epoch: 1.8210s
Epoch  260  loss  0.00040522502180796835 correct 50 avg time per epoch: 1.9139s
Epoch  270  loss  0.11223428878112031 correct 50 avg time per epoch: 1.8009s
Epoch  280  loss  0.010595126463759607 correct 50 avg time per epoch: 1.9070s
Epoch  290  loss  3.0583145051544882e-06 correct 50 avg time per epoch: 1.8334s
Epoch  300  loss  0.003025861415121577 correct 50 avg time per epoch: 1.9106s
Epoch  310  loss  0.0026639965901355226 correct 50 avg time per epoch: 1.8402s
Epoch  320  loss  0.04456434884540459 correct 50 avg time per epoch: 1.9204s
Epoch  330  loss  0.056842523645674124 correct 50 avg time per epoch: 1.8156s
Epoch  340  loss  0.03585532258216335 correct 50 avg time per epoch: 1.9062s
Epoch  350  loss  1.6329993394412234e-06 correct 50 avg time per epoch: 1.8264s
Epoch  360  loss  0.08872827035262491 correct 50 avg time per epoch: 1.9148s
Epoch  370  loss  0.09079838499908466 correct 50 avg time per epoch: 1.8245s
Epoch  380  loss  0.00722290833320929 correct 50 avg time per epoch: 1.9184s
Epoch  390  loss  0.08222123445567651 correct 50 avg time per epoch: 1.8500s
Epoch  400  loss  0.06769054017256902 correct 50 avg time per epoch: 1.9034s
Epoch  410  loss  0.06743768821420275 correct 50 avg time per epoch: 1.8266s
Epoch  420  loss  0.008545297375875043 correct 50 avg time per epoch: 1.9081s
Epoch  430  loss  0.020102551708142742 correct 50 avg time per epoch: 1.8299s
Epoch  440  loss  0.0780765232384261 correct 50 avg time per epoch: 1.9202s
Epoch  450  loss  0.05670802297233559 correct 50 avg time per epoch: 1.8296s
Epoch  460  loss  0.07637702047456306 correct 50 avg time per epoch: 1.9061s
Epoch  470  loss  0.0017777589967608641 correct 50 avg time per epoch: 1.8133s
Epoch  480  loss  0.008326395461706639 correct 50 avg time per epoch: 1.9220s
Epoch  490  loss  0.004759941092684553 correct 50 avg time per epoch: 1.8065s
