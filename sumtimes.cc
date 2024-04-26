#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <mpi.h>

static void check(const hipError_t err, const char *const file, const int line)
{
  if (err == hipSuccess) return;
  fprintf(stderr,"GPU ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,hipGetErrorName(err),hipGetErrorString(err));
  fflush(stderr);
  exit(err);
}

static void check(const hipblasStatus_t err, const char *const file, const int line)
{
  if (err == HIPBLAS_STATUS_SUCCESS) return;
  fprintf(stderr,"BLAS ERROR AT LINE %d OF FILE '%s': %s\n",line,file,hipblasStatusToString(err));
  fflush(stderr);
  exit(err);
}

#define CHECK(X) check(X,__FILE__,__LINE__)

__global__ void init(const int rank, const int size, const int count, double *const sendbuf, double *const expected)
{
  const int i = threadIdx.x+blockIdx.x*blockDim.x;
  if (i < count) {
    sendbuf[i] = double(rank+1)*double(i+1);
    expected[i] = 0.5*double(size)*double(size+1)*double(i+1);
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc,&argv);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  int rank = MPI_PROC_NULL;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int nd = 0;
  CHECK(hipGetDeviceCount(&nd));
  assert(nd);
  int ndMax = 0;
  MPI_Allreduce(&nd,&ndMax,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  if (ndMax > 1) {
    // Assign GPUs to MPI tasks on a node in a round-robin fashion
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&local);
    int lrank = MPI_PROC_NULL;
    MPI_Comm_rank(local,&lrank);
    MPI_Comm_free(&local);
    const int target = lrank%nd;
    CHECK(hipSetDevice(target));
    int myd = -1;
    CHECK(hipGetDevice(&myd));
    for (int i = 0; i < size; i++) {
      if (rank == i) {
        printf("# MPI task %d with node rank %d using device %d (%d devices per node)\n",rank,lrank,myd,nd);
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  int countLo = 1;
  int countHi = 1024*1024*1024;
  int iters = 3;

  int i = 0;
  if (argc > 1) sscanf(argv[1],"%d",&i);
  if (i > 0) iters = i;
  i = 0;
  if (argc > 2) sscanf(argv[2],"%d",&i);
  if (i > 0) countLo = i;
  i = 0;
  if (argc > 3) sscanf(argv[3],"%d",&i);
  if (i > 0) countHi = i;

  hipblasHandle_t handle;
  CHECK(hipblasCreate(&handle));
  CHECK(hipblasSetPointerMode(handle,HIPBLAS_POINTER_MODE_HOST));

  const long bytesHi = countHi*sizeof(double);

  double *recvbuf = nullptr;
  CHECK(hipMalloc(&recvbuf,bytesHi));

  double *sendbuf = nullptr;
  CHECK(hipMalloc(&sendbuf,bytesHi));
  double *expected = nullptr;
  CHECK(hipMalloc(&expected,bytesHi));

  const int block = 256;
  const int grid = (countHi-1)/block+1;
  init<<<grid,block>>>(rank,size,countHi,sendbuf,expected);

  if (rank == 0) {
    printf("\n\n# Performance of MPI_Allreduce MPI_DOUBLE MPI_SUM with GPU buffers, %d tasks, %d iterations, counts %d to %d by halving\n",size,iters,countHi,countLo);
    printf("# count | GiB/rank (in+out) | seconds (min, avg, max) | GiB/s/rank (min, avg, max) | Total GF/s (min, avg, max)\n");
    fflush(stdout);
  }

  for (int count = countHi; count >= countLo; count /= 2) {

    const long bytes = long(count)*sizeof(double);
    assert(bytes <= bytesHi);
    const double gib = 2.0*double(bytes)/double(1024*1024*1024);
    const double gf = double(count)*double(size-1)/double(1024*1024*1024);

    double timeMin = 60.0*60.0*24.0*365.0;
    double timeSum = 0;
    double timeMax = 0;

    for (int i = 0; i <= iters; i++) {

      CHECK(hipMemset(recvbuf,0,bytes));
      CHECK(hipDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);

      const double before = MPI_Wtime();
      MPI_Allreduce(sendbuf,recvbuf,count,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      const double after = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD);

      const double alpha = -1.0;
      CHECK(hipblasDaxpy(handle,count,&alpha,expected,1,recvbuf,1));

      int result = 0;
      CHECK(hipblasIdamax(handle,count,recvbuf,1,&result));
      CHECK(hipDeviceSynchronize());
      result--; // Convert from Fortran to C indexing
      double diffMax = 0;
      double expectedMax = 0;
      CHECK(hipMemcpyAsync(&diffMax,&recvbuf[result],sizeof(double),hipMemcpyDeviceToHost,0));
      CHECK(hipMemcpy(&expectedMax,&expected[result],sizeof(double),hipMemcpyDeviceToHost));
      const double errMe = std::abs(diffMax/expectedMax);
      double err = 0;
      MPI_Reduce(&errMe,&err,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

      const double timeMe = after-before;
      double time = 0;
      MPI_Reduce(&timeMe,&time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      if (rank == 0) {
        if (i == 0) {
          printf("### 0 time %g err %g (warmup, ignored)\n",time,err);
        } else {
          timeMin = std::min(timeMin,time);
          timeSum += time;
          timeMax = std::max(timeMax,time);
          printf("### %d time %g err %g\n",i,time,err);
        }
        fflush(stdout);
      }
    }
    if (rank == 0) {
        const double timeAvg = timeSum/double(iters);
        printf("%d %g %g %g %g %g %g %g %g %g %g\n",count,gib,timeMin,timeAvg,timeMax,gib/timeMax,gib/timeAvg,gib/timeMin,gf/timeMax,gf/timeAvg,gf/timeMin);
        fflush(stdout);
    }
  }
  if (rank == 0) { printf("# Done\n"); fflush(stdout); }

  CHECK(hipFree(expected));
  expected = nullptr;
  CHECK(hipFree(sendbuf));
  sendbuf = nullptr;
  CHECK(hipFree(recvbuf));
  recvbuf = nullptr;

  MPI_Finalize();
  return 0;
}

