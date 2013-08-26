#define LOCAL_QUEUE_SIZE 5

inline int propagate(__global int* seeds, __global uchar* image,
                     int x, int y, int ncols, uchar pval)
{
    int returnValue = -1;
    int index = y*ncols + x;
    uchar seedXYval = seeds[index];
    uchar imageXYval = image[index];

    if((seedXYval < pval) && (imageXYval != seedXYval))
    {
        int newValue = min(pval, imageXYval);
        atomic_max(&(seeds[index]), newValue);
        returnValue = index;
    }

    return returnValue;
}


__kernel void scan_forward_rows_kernel(__global int* marker,
                                       __global int* mask,
                                       __global int* changed_global,
                                       __local int* marker_local,
                                       __local int* mask_local,
                                       int width, int height)
{
    int local_id_x = get_local_id(0);
    int local_id_y = get_local_id(1);

    int group_size_x = get_local_size(0);
    int group_size_y = get_local_size(1);

    int group_id = get_group_id(0);

    // load from global to local

    int row = group_id * group_size_y + local_id_y;

    int idx_local = local_id_y * group_size_x + local_id_x;

    int step = group_size_x - 1;
    int changed = 0;

    for(int i = local_id_x; i < width; i += step)
    {
        int idx_global = row + i;

        marker_local[idx_local] = marker[idx_global];
        mask_local[idx_local] = mask[idx_global];

        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id_x == 0)
        {
            for(int col = 0; i < group_size_x - 1
                        && idx_global + col < width; ++i)
            {
                int marker_val = marker_local[col];
                int mask_val = mask_local[col];

                int marker_forward_val = idx_global + col + 1 < width
                                            ? marker_local[col+1] : 0;

                int marker_new = min(max(marker_val, marker_forward_val),
                                     mask_val);

                changed |= marker_val ^ marker_new;
            }
        }
#ifdef DEBUG_PRINT

        printf("back to global\n");
#endif

        barrier(CLK_LOCAL_MEM_FENCE);

        marker[idx_global] = marker_local[idx_local];
        //mask[idx_global] = mask_local[idx_local];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
#ifdef DEBUG_PRINT
    printf("changed %d\n", changed);
#endif
    if(changed)
        changed_global = 1;
}

__kernel void morph_recon_kernel(__global int* total_inserts,
                                 __global int* seeds,
                                 __global uchar* image,
                                 int ncols, int nrows,
                                 // all this shared stuff for queues:
                                 QUEUE_DATA,
                                 QUEUE_METADATA,
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
int counter = 0;
    do{
        /* queue occupancy initialization */
        my_local_queue[0] = 0;

        // Try to get some work.
        workUnit = dequeueElement(queue_data, queue_metadata, &loopIt, gotWork);
        y = workUnit / ncols;
        x = workUnit % ncols; // modulo is very inefficient on gpu!

        assert(workUnit < ncols * nrows);

        uchar pval = 0;

        if(workUnit >= 0)
        {
            pval = seeds[workUnit];
        }

        int retWork = -1;
        if(workUnit >= 0 && y > 0)
        {
            retWork = propagate(seeds, image, x, y-1, ncols, pval);

            if(retWork > 0)
            {
                my_local_queue[0]++;
                my_local_queue[my_local_queue[0]] = retWork;
            }
        }

        if(workUnit >= 0 && y < nrows-1)
        {
            retWork = propagate(seeds, image, x, y+1, ncols, pval);

            if(retWork > 0)
            {
                my_local_queue[0]++;
                my_local_queue[my_local_queue[0]] = retWork;
            }
        }

        if(workUnit >= 0 && x > 0)
        {
            retWork = propagate(seeds, image, x-1, y, ncols, pval);

            if(retWork > 0)
            {
                my_local_queue[0]++;
                my_local_queue[my_local_queue[0]] = retWork;
            }
        }

        if(workUnit >= 0 && x < ncols-1)
        {
            retWork = propagate(seeds, image, x+1, y, ncols, pval);

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

    total_inserts[group_id] = TOTAL_INSERTS[group_id];
}
