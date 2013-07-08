
__kernel void threshold(__global uchar *src, int src_pitch,
                     __global uchar *dst, int dst_pitch,
                     int width, int height,
                     uchar lower, uchar upper,
                     uchar lower_inclusive, uchar upper_inclusive)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < width && y < height)
    {
        int src_index = y * src_pitch + x;
        int dst_index = y * dst_pitch + x;

        uchar value = src[src_index];
        bool pb = (value > lower) && (value < upper);

        if (lower_inclusive) pb = pb || (value == lower);
        if (upper_inclusive) pb = pb || (value == upper);

        dst[dst_index] = pb ? 255 : 0;
    }
}

