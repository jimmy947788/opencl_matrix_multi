__kernel void matrix5_dot_matrix5(
         __global int *A,
         __global int *B,
         __global int *C)
{
   
   int dim = get_work_dim();//dim=2
   int x = get_global_id(0);
   int y = get_global_id(1);
   int width = get_global_size(0);
   int height = get_global_size(1);
   int index = (y * width) + x;

   ///printf("dim=%d, x=%d, y=%d \n", dim, x, y);
   
   int A0 = A[(y * width) + 0];
   int A1 = A[(y * width) + 1];
   int A2 = A[(y * width) + 2];
   int A3 = A[(y * width) + 3];
   int A4 = A[(y * width) + 4];
   //printf("y=%d, A0=%d, A1=%d, A2=%d, A3=%d, A4=%d, index=%d \n", y, A0, A1, A2, A3, A4, index);

   int B0 = B[x + (0 * width)];
   int B1 = B[x + (1 * width)];
   int B2 = B[x + (2 * width)];
   int B3 = B[x + (3 * width)];
   int B4 = B[x + (4 * width)];
   //printf("x=%d, B0=%d, B1=%d, B2=%d, B3=%d, B4=%d, index=%d \n", x, B0, B1, B2, B3, B4, index);
   
   C[index] = (A0 * B0) + (A1 * B1) + (A2 * B2) + (A3 * B3) + (A4 * B4); 
   //printf("C[%d] = %d\n", index, C[index]);
   /*
   printf("C%d%d[%d] = A%d%d(%d) * B%d%d(%d) + A%d%d(%d) * B%d%d(%d) + A%d%d(%d) * B%d%d(%d) + A%d%d(%d) * B%d%d(%d) + A%d%d(%d) * B%d%d(%d)\n",
      x, y, index, 
      0, y, A0, 
      x, 0, B0, 
      1, y, A1, 
      x, 1, B1, 
      2, y, A2, 
      x, 2, B2, 
      3, y, A3, 
      x, 3, B3, 
      4, y, A4, 
      x, 4, B4)
   */
}
