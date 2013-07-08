__kernel void invert(__global uchar *src, int src_pitch,
                     __global uchar *dst, int dst_pitch,
                     int width, int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < width && y < height)
    {
        int src_index = y * src_pitch + x;
        int dst_index = y * dst_pitch + x;

        dst[dst_index] = 255 - src[src_index];
    }
}

