#include<cuda_runtime.h>
#include<stdio.h>
#include"engine/cache.hpp"
#include"engine/walk.hpp"
typedef class gpu_block
{
public:
   bid_t blk,cache_index;
   vid_t start_vert,nverts;
   eid_t start_edge,nedges;
   eid_t *beg_pos;
   //std::vector<vid_t> degree;
   vid_t *csr;
   real_t *weights;
};
class gpu_cache
{
public:
   bid_t ncblock;
   gpu_block *cache_blocks;
   bid_t *walk_blocks;
};
class gpu_walks
{
public:
   hid_t hops;
   walk_t max;
   walker_t *walks;
};

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error:" << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE); // 或者你可以选择抛出异常或执行其他错误处理
    }
}

void initgpucache(gpu_block *gpu,graph_cache *cpu)
{
   for(int i=0;i<cpu->ncblock;i++)
   {
      checkCudaError(cudaMalloc((void**)&(gpu[i].beg_pos),sizeof(eid_t)*(cpu->cache_blocks[i].block->nverts+1)));
      checkCudaError(cudaMalloc((void**)&(gpu[i].csr),sizeof(vid_t)*(cpu->cache_blocks[i].block->nedges)));
      if(cpu->cache_blocks[i].weights!=NULL){
         checkCudaError(cudaMalloc((void**)&(gpu[i].weights),sizeof(real_t)*cpu->cache_blocks[i].block->nedges));
         checkCudaError(cudaMemcpy(gpu[i].weights,cpu->cache_blocks[i].weights,sizeof(real_t)*cpu->cache_blocks[i].block->nedges,cudaMemcpyHostToDevice));
      }
      checkCudaError(cudaMemcpy(gpu[i].beg_pos,cpu->cache_blocks[i].beg_pos,sizeof(eid_t)*(cpu->cache_blocks[i].block->nverts+1),cudaMemcpyHostToDevice));
      checkCudaError(cudaMemcpy(gpu[i].csr,cpu->cache_blocks[i].csr,sizeof(vid_t)*(cpu->cache_blocks[i].block->nedges),cudaMemcpyHostToDevice));
      gpu[i].blk=cpu->cache_blocks[i].block->blk;
      gpu[i].cache_index=cpu->cache_blocks[i].block->cache_index;
      gpu[i].nedges=cpu->cache_blocks[i].block->nedges;
      gpu[i].nverts=cpu->cache_blocks[i].block->nverts;
      gpu[i].start_edge=cpu->cache_blocks[i].block->start_edge;
      gpu[i].start_vert=cpu->cache_blocks[i].block->start_vert;
   }
}

void graphtogpu(graph_cache *cpu_cache,graph_block *cpu_block)
{
   gpu_cache *m;
   m=(gpu_cache*)malloc(sizeof(gpu_cache));
   m->ncblock=cpu_cache->ncblock;
   checkCudaError(cudaMalloc((void**)&(m->cache_blocks),sizeof(gpu_block)*cpu_cache->ncblock));
   checkCudaError(cudaMalloc((void**)&(m->walk_blocks),sizeof(bid_t)*cpu_cache->ncblock));
   gpu_cache *g_cache;
   checkCudaError(cudaMalloc((void**)&g_cache,sizeof(gpu_cache)));
   checkCudaError(cudaMemcpy(g_cache,m,sizeof(gpu_cache),cudaMemcpyHostToDevice));
   gpu_block *m_b=(gpu_block*)malloc(sizeof(gpu_block)*cpu_cache->ncblock);
   initgpucache(m_b,cpu_cache);
   checkCudaError(cudaMemcpy(g_cache->cache_blocks,m_b,sizeof(gpu_block)*cpu_cache->ncblock,cudaMemcpyHostToDevice));
   bid_t *m_w=(bid_t*)malloc(sizeof(bid_t)*cpu_cache->ncblock);
   for(int i=0;i<cpu_cache->ncblock;i++)
      m_w[i]=cpu_cache->walk_blocks[i];
   checkCudaError(cudaMemcpy(g_cache->walk_blocks,m_w,sizeof(bid_t)*cpu_cache->ncblock,cudaMemcpyHostToDevice)); 
}