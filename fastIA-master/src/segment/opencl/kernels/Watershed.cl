
// Step 1.
//__global__ void descent_kernel(float* labeled, const int w, const int h)
//{
//  int tx = threadIdx.x;  int ty = threadIdx.y;
//  int bx = blockIdx.x;   int by = blockIdx.y;
//  int bdx = blockDim.x;  int bdy = blockDim.y;
//  int i = bdx * bx + tx; int j = bdy * by + ty;

//  __shared__ float s_I[BLOCK_SIZE*BLOCK_SIZE];
//  int size = BLOCK_SIZE - 2;
//  int img_x = L2I(i,tx);
//  int img_y = L2I(j,ty);
//  int new_w = w + w * 2;
//  int new_h = h + h * 2;
//  int p = INDEX(img_y,img_x,w);

//  int ghost = (tx == 0 || ty == 0 ||
//  tx == bdx - 1 || ty == bdy - 1);

//  if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
//     (bx == (w / size - 1) && tx == bdx - 1) ||
//     (by == (h / size - 1) && ty == bdy - 1)) {
//       s_I[INDEX(ty,tx,BLOCK_SIZE)] = INF;
//  } else {
//     s_I[INDEX(ty,tx,BLOCK_SIZE)] = tex2D(img,img_x,img_y);
//  }

//  __syncthreads();

//  if (j < new_h && i < new_w && ghost == 0) {
//    float I_q_min = INF;
//    float I_p = tex2D(img,img_x,img_y);

//    int exists_q = 0;

//    for (int k = 0; k < 8; k++) {
//      int n_x = N_xs[k]+tx; int n_y = N_ys[k]+ty;
//      float I_q = s_I[INDEX(n_y,n_x,BLOCK_SIZE)];
//      if (I_q < I_q_min) I_q_min = I_q;
//    }

//    for (int k = 0; k < 8; k++) {
//      int x = N_xs[k]; int y = N_ys[k];
//      int n_x = x+tx; int n_y = y+ty;
//      int n_tx = L2I(i,n_x); int n_ty = L2I(j,n_y);
//      float I_q = s_I[INDEX(n_y,n_x,BLOCK_SIZE)];
//      int q = INDEX(n_ty,n_tx,w);
//      if (I_q < I_p && I_q == I_q_min) {
//        labeled[p] = -q;
//        exists_q = 1; break;
//      }
//    }
//    if (exists_q == 0) labeled[p] = PLATEAU;
//  }
//}

#define INF 9999999999
//#define INF 255
#define PLATEAU 0
#define BLOCK_SIZE 6

// Convert 2D index to 1D index.
#define INDEX(j,i,ld) ((j) * ld + (i))

// Convert local (shared memory) coord to global (image) coordinate.
#define L2I(ind,off) (((ind) / BLOCK_SIZE) * (BLOCK_SIZE - 2)-1+(off))

__kernel void descent_kernel(
                    __global uchar *src,
                    __global float* labeled,
                    __constant char* N_xs,
                    __constant char* N_ys,
                    __local float* s_I,
                     int w, int h)
{
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int bx = get_group_id(0);
    int by = get_group_id(1);

    int bdx = get_local_size(0);
    int bdy = get_local_size(1);

    int i = get_global_id(0);
    int j = get_global_id(1);

    int size = get_local_size(0) - 2;

    int img_x = L2I(i,tx);
    int img_y = L2I(j,ty);
    int new_w = w + w / 2;
    int new_h = h + h / 2;
    int p = INDEX(img_y,img_x,w);

    if(i == 0 && j == 0){
        printf("size: %d x %d\n", get_local_size(0), get_local_size(1));
        printf("global size: %d x %d\n", get_global_size(0), get_global_size(1));
    }

    // is set when pixel is on the edge of a tile
    int ghost = (tx == 0 || ty == 0 ||
          tx == bdx - 1 || ty == bdy - 1);

    if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
            (bx == (w / size - 1) && tx == bdx - 1) ||
            (by == (h / size - 1) && ty == bdy - 1)) {

        s_I[INDEX(ty,tx,BLOCK_SIZE)] = INF;

    } else {
        s_I[INDEX(ty,tx,BLOCK_SIZE)] = src[img_y * w + img_x];
        //0;//tex2D(img,img_x,img_y);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (j < new_h && i < new_w && ghost == 0) {
        float I_q_min = INF;
        float I_p = s_I[INDEX(ty,tx,BLOCK_SIZE)];
        //tex2D(img,img_x,img_y);

        int exists_q = 0;

        for (int k = 0; k < 8; k++) {
            int n_x = N_xs[k] + tx;
            int n_y = N_ys[k] + ty;

            float I_q = s_I[INDEX(n_y,n_x,BLOCK_SIZE)];

            if (I_q < I_q_min) I_q_min = I_q;
        }

        for (int k = 0; k < 8; k++) {

            int n_x = N_xs[k] + tx;
            int n_y = N_ys[k] + ty;
            int n_tx = L2I(i,n_x);
            int n_ty = L2I(j,n_y);
            float I_q = s_I[INDEX(n_y,n_x,BLOCK_SIZE)];

            int q = INDEX(n_ty,n_tx,w);

            if (I_q < I_p && I_q == I_q_min) {
                labeled[p] = q;
                exists_q = 1; break;
            }
        }
        if (exists_q == 0) labeled[p] = PLATEAU;
    }
}


// Step 2A.
__kernel void increment_kernel(
                        __global float* labeled,
                         int w, int h)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int p = INDEX(j,i,w);

    if (j < h && i < w && labeled[p] == PLATEAU) {
        labeled[p] = p + 1;
    }
}


// Step 2B.
__kernel void minima_kernel(
                    volatile __global int *C,
                    __global float* L,
                    __constant char* N_xs,
                    __constant char* N_ys,
                    __local float* s_L,
                     int w, int h)
{

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    int bx = get_group_id(0);
    int by = get_group_id(1);

    int bdx = get_local_size(0);
    int bdy = get_local_size(1);

    int i = get_global_id(0);
    int j = get_global_id(1);

    int size = BLOCK_SIZE - 2;
    int img_x = L2I(i,tx);
    int img_y = L2I(j,ty);

    int true_p = INDEX(img_y,img_x,w);
    int s_p = INDEX(ty,tx,BLOCK_SIZE);

    int new_w = w + w * 2;
    int new_h = h + h * 2;

    int ghost =  (tx == 0 || ty == 0 ||
    tx == bdx - 1 || ty == bdy - 1) ? 1 : 0;

    if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) ||
        (bx == (w / size - 1) && tx == bdx - 1) ||
        (by == (h / size - 1) && ty == bdy - 1)) {
        s_L[INDEX(ty,tx,BLOCK_SIZE)] = INF;
    } else {
        s_L[s_p] = L[INDEX(img_y,img_x,w)];
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE);

    int active = (j < new_h && i < new_w && s_L[s_p] > 0) ? 1 : 0;

    if (active == 1 && ghost == 0) {
        for (int k = 0; k < 8; k++) {
            int n_x = N_xs[k] + tx; int n_y = N_ys[k] + ty;
            int s_q = INDEX(n_y,n_x,BLOCK_SIZE);
            if (s_L[s_q] == INF) continue;
            if (s_L[s_q] > s_L[s_p])
                s_L[s_p] = s_L[s_q];
        }
        if (L[true_p] != s_L[s_p]) {
            L[true_p] = s_L[s_p];
            //atomicAdd(&C[0],1);

            atomic_inc(&C[0]);
        }
    }
}
