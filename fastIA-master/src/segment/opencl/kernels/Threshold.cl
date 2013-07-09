#ifndef TYPE_1
#define TYPE_1 uchar
#endif

__kernel void threshold(__global TYPE_1 *src, int src_pitch,
                     __global TYPE_1 *dst, int dst_pitch,
                     int width, int height,
                     TYPE_1 lower, TYPE_1 upper,
                     TYPE_1 lower_inclusive, TYPE_1 upper_inclusive)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < width && y < height)
    {
        int src_index = y * src_pitch + x;
        int dst_index = y * dst_pitch + x;

        TYPE_1 value = src[src_index];
        bool pb = (value > lower) && (value < upper);

        if (lower_inclusive) pb = pb || (value == lower);
        if (upper_inclusive) pb = pb || (value == upper);

        dst[dst_index] = pb ? 255 : 0;
    }
}

