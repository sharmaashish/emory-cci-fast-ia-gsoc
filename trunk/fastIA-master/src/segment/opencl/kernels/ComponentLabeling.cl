int find(__local int* buf, int x)
{
    while (x != buf[x])
    {
        x = buf[x];
    }
    return x;
}

int findGlobal(__global int* buf, int x)
{
    while (x != buf[x])
    {
        x = buf[x];
    }
    return x;
}

void findAndUnion(__local int* buf, int g1, int g2)
{
    bool done;

    do
    {
        g1 = find(buf, g1);
        g2 = find(buf, g2);

        // it should hold that g1 == buf[g1] and g2 == buf[g2] now

        if (g1 < g2)
        {
            int old = atomic_min(&buf[g2], g1);
            done = (old == g2);
            g2 = old;
        }
        else if (g2 < g1)
        {
            int old = atomic_min(&buf[g1], g2);
            done = (old == g1);
            g1 = old;
        }
        else
        {
            done = true;
        }
    } while(!done);
}


void findAndUnionGlobal(__global int* buf, int g1, int g2)
{
    bool done;

    do
    {
        g1 = findGlobal(buf, g1);
        g2 = findGlobal(buf, g2);

        // it should hold that g1 == buf[g1] and g2 == buf[g2] now

        if (g1 < g2)
        {
            int old = atomic_min(&buf[g2], g1);
            done = (old == g2);
            g2 = old;
        }
        else if (g2 < g1)
        {
            int old = atomic_min(&buf[g1], g2);
            done = (old == g1);
            g1 = old;
        }
        else
        {
            done = true;
        }
    } while(!done);
}


__kernel void uf_local(__global int* label,
                       __global uchar* img,
                       __local int* s_buffer,
                       __local uchar* s_img,
                       int w, int h, int connectivity) {

    int x = get_global_id(0);
    int y = get_global_id(1);

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int group_id_x = get_group_id(0);
    int group_id_y = get_group_id(1);

//    int x = blockIdx.x*blockDim.x + threadIdx.x;
//    int y = blockIdx.y*blockDim.y + threadIdx.y;

    int global_index = x + y * w;
    int block_index = UF_BLOCK_SIZE_X * local_y + local_x;

//    __shared__ int s_buffer[UF_BLOCK_SIZE_X * UF_BLOCK_SIZE_Y];
//    __shared__ unsigned char s_img[UF_BLOCK_SIZE_X * UF_BLOCK_SIZE_Y];

    bool in_limits = x < w && y < h;

    s_buffer[block_index] = block_index;
    //s_img[block_index] = in_limits? tex2D(imgtex, x, y) : 0xFF;
    s_img[block_index] = in_limits? img[global_index] : 0xFF;

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE);


    uchar v = s_img[block_index];

    if (in_limits && local_x > 0 && s_img[block_index-1] == v) {
        findAndUnion(s_buffer, block_index, block_index - 1);
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE);

    if (in_limits && local_y >0 && s_img[block_index-UF_BLOCK_SIZE_X] == v) {
        findAndUnion(s_buffer, block_index, block_index - UF_BLOCK_SIZE_X);
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE);

    if (connectivity == 8)
    {
        if (in_limits && (local_x > 0 && local_y > 0)
            && s_img[block_index-UF_BLOCK_SIZE_X-1] == v)
        {
            findAndUnion(s_buffer, block_index,
                         block_index - UF_BLOCK_SIZE_X - 1);
        }

        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE);

        if (in_limits && (local_y > 0 && local_x < UF_BLOCK_SIZE_X-1)
            && s_img[block_index-UF_BLOCK_SIZE_X+1] == v)
        {
            findAndUnion(s_buffer, block_index,
                         block_index - UF_BLOCK_SIZE_X + 1);
        }

        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (in_limits)
    {
        int f = find(s_buffer, block_index);
        int fx = f % UF_BLOCK_SIZE_X;
        int fy = f / UF_BLOCK_SIZE_X;

        label[global_index] = (group_id_y * UF_BLOCK_SIZE_Y + fy) * w
                              + (group_id_x * UF_BLOCK_SIZE_X + fx);
    }
}


__kernel void uf_global(__global int* label,
                        __global uchar* img,
                        int w, int h, int connectivity)
{
//    int x = blockIdx.x*blockDim.x + threadIdx.x;
//    int y = blockIdx.y*blockDim.y + threadIdx.y;

    int x = get_global_id(0);
    int y = get_global_id(1);

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int global_index = x+y*w;

    bool in_limits = x < w && y < h;
   // unsigned char v = (in_limits? tex2D(imgtex, x, y) : 0xFF);

    uchar v = (in_limits ? img[global_index] : 0xFF);

    if (in_limits && y > 0 && local_y == 0
        && img[global_index - w] == v)
    {
        findAndUnionGlobal(label, global_index, global_index - w);
    }

    if (in_limits && x > 0 && local_x == 0
        && img[global_index - 1] == v)
    {
        findAndUnionGlobal(label, global_index, global_index - 1);
    }
// TONY:  this algorithm chunks the image, do local UF, then use only the first row or first column
//  to merge the chunks. (above 2 lines).  now we also need to do diagonals.
// upper left diagonal needs to be updated for the left and top lines

    if (connectivity == 8)
    {
        if (in_limits && y > 0 && x > 0
            && (local_y == 0 || local_x == 0)
            && img[global_index - w - 1] == v)
        {
            findAndUnionGlobal(label, global_index, global_index - w - 1);
        }

// upper right diagonal needs to be updated for the top and right lines.
        if (in_limits && x < w-1 && y > 0
            && (local_y == 0 || local_x == UF_BLOCK_SIZE_X-1)
            && img[global_index - w + 1] == v)
        {
            findAndUnionGlobal(label, global_index, global_index - w + 1);
        }
    }
}

__kernel void uf_final(__global int* label,
                       __global uchar* img,
                       int w, int h, int bgval)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int global_index = x + y * w;

    bool in_limits = x < w && y < h;

    if (in_limits)
    {
        label[global_index] =
                (img[global_index] == 0 ? bgval : findGlobal(label, global_index));
    }
}
