/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"
#include "assert.h"
namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(ncclWorkElem *args) {

    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    ncclRing *ring = &ncclShmem.channel.ring;
    int const *ringRanks = ring->devUserRanks;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? REDUCESCATTER_CHUNKSTEPS : 1));
    // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    const ssize_t minChunkSizeLL128 = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2);
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopSize = nChannels*chunkSize;
    ssize_t realloopSize = 0;
    const ssize_t size = args->coll.count;
    const ssize_t* flag = (ssize_t*)args->recvbuff;
    if(flag[0] == 0)
    {
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
      prims(tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->coll.redOpArg);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL)
        realChunkSize = size-gridOffset < loopSize ? args->coll.lastChunkSize : chunkSize;
      else if (Proto::Id == NCCL_PROTO_LL128)
        realChunkSize = min(divUp(size-gridOffset, nChannels*minChunkSizeLL128)*minChunkSizeLL128, chunkSize);
      realChunkSize = int(realChunkSize);

      ssize_t chunkOffset = gridOffset + bid*int(realChunkSize);

      /////////////// begin ReduceScatter steps ///////////////
      ssize_t offset;
      int nelem = min(realChunkSize, size-chunkOffset);
      int rankDest;

      // step 0: push data to next GPU
      rankDest = ringRanks[nranks-1];
      offset = chunkOffset + rankDest * size;
      prims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        rankDest = ringRanks[nranks-j];
        offset = chunkOffset + rankDest * size;
        prims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final result
      rankDest = ringRanks[0];
      offset = chunkOffset + rankDest * size;
      prims.recvReduceCopy(offset, chunkOffset, nelem, /*postOp=*/true);
    }
    }
    else {
    const int buff_offset = nranks + nranks + 1;
    ssize_t* sizeptr = (ssize_t*)args->sendbuff;
    ssize_t* offsetptr = sizeptr + nranks;
    float* fp = ((float*)args->sendbuff)+2*(buff_offset);
    offsetptr[0] = 0;
    for(int i = 1; i < nranks + 1; ++i)
    {
      offsetptr[i] = offsetptr[i-1] + sizeptr[i-1];
    }
    ssize_t min_size = sizeptr[0];
    ssize_t max_size = sizeptr[0];
    for(int i = 0; i < nranks; ++i)
    {
      if(sizeptr[i] < min_size)
        min_size = sizeptr[i];
      if(sizeptr[i] > max_size)
        max_size = sizeptr[i];
    }
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto>
    prims(tid, nthreads, &ring->prev, &ring->next, ((float*)args->sendbuff)+2*(buff_offset), args->recvbuff, args->coll.redOpArg);
    for (ssize_t gridOffset = 0; gridOffset < max_size; gridOffset += realloopSize) {
      
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(min_size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL)
      {
        realChunkSize = min(chunkSize, divUp(min_size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      else if (Proto::Id == NCCL_PROTO_LL128)
      {
        realChunkSize = min(chunkSize, divUp(min_size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      realChunkSize = int(realChunkSize);

      ssize_t chunkOffset = gridOffset + bid*int(realChunkSize);
      realloopSize = min(loopSize, min_size-gridOffset);

      /////////////// begin ReduceScatter steps ///////////////
      ssize_t offset;
      int nelem = min(realChunkSize, min_size-chunkOffset);
      int rankDest;

      // step 0: push data to next GPU
      rankDest = ringRanks[nranks-1];
      if(sizeptr[rankDest] <= max_size)
      {
      offset = chunkOffset + offsetptr[rankDest];
      prims.send(offset, nelem);
      }
      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        rankDest = ringRanks[nranks-j];
        if(sizeptr[rankDest] <= max_size)
        {
        offset = chunkOffset + offsetptr[rankDest];
        prims.recvReduceSend(offset, nelem);
        }
      }

      // step k-1: reduce this buffer and data, which will produce the final result
      rankDest = ringRanks[0];
      if(sizeptr[rankDest] <= max_size)
      {
      offset = chunkOffset + offsetptr[rankDest];
      prims.recvReduceCopy(offset, chunkOffset, nelem, /*postOp=*/true);
      }
      for(int i = 0; i < nranks; ++i)
      {
        if( sizeptr[i] - gridOffset <= realloopSize && sizeptr[i] <= max_size)
          {
            sizeptr[i] = max_size + 1;
          }
      }
      min_size = max_size + 1;
      for(int i = 0; i < nranks; ++i)
      {
        if(sizeptr[i] <= max_size && sizeptr[i] < min_size)
        {
          min_size = sizeptr[i];
        }
      }
    }
    }
 
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};
