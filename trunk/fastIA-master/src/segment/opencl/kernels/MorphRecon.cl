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
        uchar newValue = min(pval, imageXYval);
        atomic_max(&(seeds[index]), newValue);
        returnValue = index;
    }

    return returnValue;
}

__kernel void morph_recon_kernel(__global int* total_inserts,
                                 __global int* seeds,
                                 __global uchar* image,
                                 int ncols, int nrows,
                                 // all this shared stuff for queues:
                                 QUEUE_WORKSPACE,
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

    setCurrentQueue(QUEUE_WORKSPACE_ARG, group_id, group_id);

    int loopIt = 0;
    int workUnit = -1;
    int x, y;

    __local int* my_local_queue = local_queue + LOCAL_QUEUE_SIZE * local_id;

    do{
        /* queue occupancy initialization */
        my_local_queue[0] = 0;

        // Try to get some work.
        workUnit = dequeueElement(QUEUE_WORKSPACE_ARG, &loopIt, gotWork);
        y = workUnit / ncols;
        x = workUnit % ncols; // modulo is very inefficient on gpu!

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

        queueElement(QUEUE_WORKSPACE_ARG, my_local_queue,
                        prefix_sum_input, prefix_sum_output);

    }while(workUnit != -2);

    total_inserts[group_id] = TOTAL_INSERTS[group_id];
}
