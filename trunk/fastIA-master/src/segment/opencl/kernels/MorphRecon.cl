/*
    MARKER_TYPE, MASK_TYPE <-- should be passed when program is build
*/

#define LOCAL_QUEUE_SIZE 5

#define DEBUG_PRINT

inline int propagate(__global MARKER_TYPE* marker, __global MASK_TYPE* mask,
                     int x, int y, int ncols, MARKER_TYPE pval)
{
    int returnValue = -1;
    int index = y*ncols + x;
    MARKER_TYPE markerXYval = marker[index];
    MASK_TYPE maskXYval = mask[index];

    if((markerXYval < pval) && (maskXYval != markerXYval))
    {
        MARKER_TYPE newValue = min(pval, (MARKER_TYPE)maskXYval);
        atomic_max(&(marker[index]), newValue);
        returnValue = index;
    }

    return returnValue;
}


__kernel void scan_forward_rows_kernel(__global MARKER_TYPE* marker,
                                       __global MASK_TYPE* mask,
                                       __global int* changed_global,
                                       __local MARKER_TYPE* marker_local,
                                       __local MASK_TYPE* mask_local,
                                       int width, int height)
{
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int global_id_y = get_global_id(1);

    int group_size_x = get_local_size(0);
    int group_size_y = get_local_size(1);

    int group_id_y = get_group_id(1);

    // load from global to local
    int row = global_id_y * width;

    int idx_local = local_id_y * group_size_x + local_id_x;

    int changed = 0;
    MARKER_TYPE previous_value = 0;

    int limit = ((width + group_size_x - 1) / group_size_x) * group_size_x;

    // all threads performs the same number of times
    for(int i = local_id_x; i < limit; i += group_size_x)
    {
        int idx_global = row + i;

        if(i < width && global_id_y < height)
        {
            marker_local[idx_local] = marker[idx_global];
            mask_local[idx_local] = mask[idx_global];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id_x == 0)
        {
            for(int col = 0; col < group_size_x; ++col)
            {
                int local_idx = local_id_y * group_size_x + col;

                MARKER_TYPE marker_val = marker_local[local_idx];
                MASK_TYPE mask_val = mask_local[local_idx];

                previous_value = min(max(marker_val, previous_value),
                                     (MARKER_TYPE)mask_val);

                marker_local[local_idx] = previous_value;

                changed |= marker_val ^ previous_value;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(i < width && global_id_y < height)
        {
            marker[idx_global] = marker_local[idx_local];
        }
 //       barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(changed)
        *changed_global = 1;
}

__kernel void scan_backward_rows_kernel(__global MARKER_TYPE* marker,
                                       __global MASK_TYPE* mask,
                                       __global int* changed_global,
                                       __local MARKER_TYPE* marker_local,
                                       __local MASK_TYPE* mask_local,
                                       int width, int height)
{
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int global_id_y = get_global_id(1);

    int group_size_x = get_local_size(0);
    int group_size_y = get_local_size(1);

    int group_id_y = get_group_id(1);

    // load from global to local
    int row = global_id_y * width;

    int idx_local = local_id_y * group_size_x + local_id_x;

    int changed = 0;
    MARKER_TYPE previous_value = 0;

    int limit = ((width + group_size_x - 1) / group_size_x) * group_size_x;

    // all threads performs the same number of times
    for(int i = limit - group_size_x + local_id_x; i >= 0; i -= group_size_x)
    {
        int idx_global = row + i;

        if(i < width && global_id_y < height)
        {
            marker_local[idx_local] = marker[idx_global];
            mask_local[idx_local] = mask[idx_global];
        }
        else
        {
            marker_local[idx_local] = 0;
            mask_local[idx_local] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id_x == 0)
        {
            for(int col = group_size_x - 1; col >= 0 ; --col)
            {
                int local_idx = local_id_y * group_size_x + col;

                MARKER_TYPE marker_val = marker_local[local_idx];
                MASK_TYPE mask_val = mask_local[local_idx];

                previous_value = min(max(marker_val, previous_value),
                                    (MARKER_TYPE)mask_val);

                marker_local[local_idx] = previous_value;

                changed |= marker_val ^ previous_value;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(i < width && global_id_y < height)
        {
            marker[idx_global] = marker_local[idx_local];
        }
 //       barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(changed)
        *changed_global = 1;
}

__kernel void scan_forward_columns_kernel(__global MARKER_TYPE* marker,
                                       __global MASK_TYPE* mask,
                                       __global int* changed_global,
                                       int width, int height)
{
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int global_id_x = get_global_id(0);

    int group_size_x = get_local_size(0);
    int group_size_y = get_local_size(1);

    int group_id_y = get_group_id(1);

    int changed = 0;
    MARKER_TYPE previous_value = 0;

    for(int i = 0; i < height; ++i)
    {
        if(global_id_x < width)
        {
            int idx = i * width + global_id_x;

            MARKER_TYPE marker_val = marker[idx];
            MASK_TYPE mask_val = mask[idx];

            previous_value = min(max(marker_val, previous_value),
                                (MARKER_TYPE)mask_val);

            marker[idx] = previous_value;

            changed |= marker_val ^ previous_value;
        }
    }

    if(changed)
        *changed_global = 1;
}

__kernel void scan_backward_columns_kernel(__global MARKER_TYPE* marker,
                                       __global MASK_TYPE* mask,
                                       __global int* changed_global,
                                       int width, int height)
{
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int global_id_x = get_global_id(0);

    int group_size_x = get_local_size(0);
    int group_size_y = get_local_size(1);

    int group_id_y = get_group_id(1);

    int changed = 0;
    MARKER_TYPE previous_value = 0;

    for(int i = height - 1; i >= 0 ; --i)
    {
        if(global_id_x < width)
        {
            int idx = i * width + global_id_x;

            MARKER_TYPE marker_val = marker[idx];
            MASK_TYPE mask_val = mask[idx];

            previous_value = min(max(marker_val, previous_value),
                                (MARKER_TYPE)mask_val);

            marker[idx] = previous_value;

            changed |= marker_val ^ previous_value;
        }
    }

    if(changed)
        *changed_global = 1;
}


__kernel void init_queue_kernel(__global MARKER_TYPE* marker,
                                __global MASK_TYPE* mask,
                                __global int* queue_data,
                                volatile __global int* queue_size,
                                int width, int height)
{
    int global_id_x = get_global_id(0);
    int global_id_y = get_global_id(1);

    bool is_candidate = 0;

    MARKER_TYPE marker_val;
    MASK_TYPE mask_val;

    int idx = global_id_y * width + global_id_x;

    if(global_id_x < width && global_id_y < height)
    {
        MARKER_TYPE central = marker[idx];

        if(global_id_x != 0)
        {
            marker_val = marker[idx - 1];
            mask_val = mask[idx - 1];

            if(marker_val < central && marker_val < mask_val)
            {
                is_candidate = 1;
            }
        }

        if(global_id_x != width - 1 && !is_candidate)
        {
            marker_val = marker[idx + 1];
            mask_val = mask[idx + 1];

            if(marker_val < central && marker_val < mask_val)
            {
                is_candidate = 1;
            }
        }

        if(global_id_y != 0 && !is_candidate)
        {
            marker_val = marker[idx - width];
            mask_val = mask[idx - width];

            if(marker_val < central && marker_val < mask_val)
            {
                is_candidate = 1;
            }
        }

        if(global_id_y != height - 1 && !is_candidate)
        {
            marker_val = marker[idx + width];
            mask_val = mask[idx + width];

            if(marker_val < central && marker_val < mask_val)
            {
                is_candidate = 1;
            }
        }
//        if(is_candidate)
//        {
//            printf("IS CANDIDATE!\n");
//            printf("x: %d\n", idx % width);
//            printf("y: %d\n", idx / width);;
//        }
    }

    if(is_candidate)
    {
        int queue_pos = atomic_inc(queue_size);
        queue_data[queue_pos] = idx;
    }
}



__kernel void morph_recon_kernel(__global int* total_inserts,
                                 __global MARKER_TYPE* marker,
                                 __global MASK_TYPE* mask,
                                 int ncols, int nrows,
                                 QUEUE_DATA,
                                 QUEUE_METADATA,
                                 // all this shared stuff for queues:
                                 __local int *local_queue,
                                 __local int *reduction_buffer,
                                 __local int* gotWork,
                                 // queue stuff:
                                 __local int* prefix_sum_input,
                                 __local int* prefix_sum_output)
{
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

  //  setCurrentQueue(QUEUE_WORKSPACE_ARG, group_id, group_id);

    int loopIt = 0;
    int workUnit = -1;
    int x, y;

    __local int* my_local_queue = local_queue + LOCAL_QUEUE_SIZE * local_id;
//int counter = 0;
    do{
        /* queue occupancy initialization */
        my_local_queue[0] = 0;

        // Try to get some work.
        workUnit = dequeueElement(queue_data, queue_metadata, &loopIt, gotWork);
        y = workUnit / ncols;
        x = workUnit % ncols; // modulo is very inefficient on gpu!

        assert(workUnit < ncols * nrows);

        MARKER_TYPE pval = 0;

        if(workUnit >= 0)
        {
            pval = marker[workUnit];
        }

        int retWork = -1;
        if(workUnit >= 0 && y > 0)
        {
            retWork = propagate(marker, mask, x, y-1, ncols, pval);

            if(retWork > 0)
            {
                my_local_queue[0]++;
                my_local_queue[my_local_queue[0]] = retWork;
            }
        }

        if(workUnit >= 0 && y < nrows-1)
        {
            retWork = propagate(marker, mask, x, y+1, ncols, pval);

            if(retWork > 0)
            {
                my_local_queue[0]++;
                my_local_queue[my_local_queue[0]] = retWork;
            }
        }

        if(workUnit >= 0 && x > 0)
        {
            retWork = propagate(marker, mask, x-1, y, ncols, pval);

            if(retWork > 0)
            {
                my_local_queue[0]++;
                my_local_queue[my_local_queue[0]] = retWork;
            }
        }

        if(workUnit >= 0 && x < ncols-1)
        {
            retWork = propagate(marker, mask, x+1, y, ncols, pval);

            if(retWork > 0)
            {
                my_local_queue[0]++;
                my_local_queue[my_local_queue[0]] = retWork;
            }
        }

        queueElement(queue_data, queue_metadata, my_local_queue,
                        prefix_sum_input, prefix_sum_output);

      //  printf("turn");
    }while(workUnit != -2);

    total_inserts[group_id] = TOTAL_INSERTS(group_id);
}
