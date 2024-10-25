#include<cuda_runtime.h>
#include <unordered_set>
#include<curand_kernel.h>
#include<curand.h>
#include"memory.hpp"
__global__ void update(gpu_walks gpuwalks,gpu_cache cache,real_t p, real_t q,curandState *state){
    const u_int32_t tid=1;//线程怎么分配？
    vid_t cur_vertex =gpuwalks.walks[tid].current , prev_vertex = gpuwalks.walks[tid].previous;
    hid_t hop = gpuwalks.walks[tid].hop;        
    bid_t cur_blk = cache.cache_blocks[gpuwalks.walks[tid].cur_index].blk;
    bid_t prev_blk = cache.cache_blocks[gpuwalks.walks[tid].prev_index].blk;
    bid_t cur_cache_index = gpuwalks.walks[tid].cur_index;
    bid_t prev_cache_index = gpuwalks.walks[tid].cur_index;
    wid_t run_step = 0;
    while (cur_cache_index != -1 && hop < gpuwalks.hops)
        {
            gpu_block *cur_block=&(cache.cache_blocks[cur_cache_index]);
            gpu_block *prev_block=&(cache.cache_blocks[prev_cache_index]);//可能有问题
            bid_t cur_index=gpuwalks.walks[tid].cur_index;
            bid_t prev_index=gpuwalks.walks[tid].prev_index;
            vid_t start_vertex =cache.cache_blocks[cur_index].start_vert;
            vid_t off = cur_vertex - start_vertex;
            vid_t prev_start_vertex = cache.cache_blocks[prev_index].start_vert;
            vid_t prev_off = prev_vertex - prev_start_vertex;

            eid_t adj_head = cur_block->beg_pos[off] - cur_block->start_edge, adj_tail = cur_block->beg_pos[off + 1] - cur_block->start_edge;
            eid_t prev_adj_head = prev_block->beg_pos[prev_off] - prev_block->start_edge, prev_adj_tail = prev_block->beg_pos[prev_off + 1] - prev_block->start_edge;
            vid_t next_vertex = 0;
            eid_t deg = adj_tail - adj_head;
            if (deg == 0){       
                hop=gpuwalks.hops-1;
            }
            else
            {
                real_t max_val = std::max(1.0 / p, std::max(1.0 / q, 1.0));
                real_t min_val = std::min(1.0 / p, std::min(1.0 / q, 1.0));
                bool accept = false;
                size_t rand_pos = 0;
                while (!accept)
                {
                    real_t rand_val =curand_uniform(&state[tid])*max_val;
                    rand_pos = seed->iRand(static_cast<uint32_t>(deg));
                    if (rand_val <= min_val)
                    {
                        accept = true;
                        break;
                    }
                    if (cur_block->csr[adj_head + rand_pos] == prev_vertex)
                    {
                        if (rand_val < 1.0 / p)
                            accept = true;
                    }
                    else if (std::binary_search(prev_block->csr + prev_adj_head, prev_block->csr + prev_adj_tail, cur_block->csr[adj_head + rand_pos]))
                    {
                        if (rand_val < 1.0)
                            accept = true;
                    }
                    else
                    {
                        if (rand_val < 1.0 / q)
                            accept = true;
                    }
                }
                assert(rand_pos<deg);
                next_vertex = cur_block->csr[adj_head + rand_pos];
            }
            prev_vertex = cur_vertex;
            cur_vertex = next_vertex;
            prev_blk = cur_blk;
            hop++;
            run_step++;
            prev_cache_index = cur_cache_index;
            if (!(cur_vertex >= cur_block->block->start_vert && cur_vertex < cur_block->block->start_vert + cur_block->block->nverts))//不在当前块
            {
                cur_blk = walk_manager->global_blocks->get_block(cur_vertex);
                cur_cache_index = (*(walk_manager->global_blocks))[cur_blk].cache_index;
                if (!continue_update)
                    break; 
            }
            if(cur_cache_index!=nblocks&&part){
                    walker_t newwalker=walker_makeup(WALKER_ID(walker),WALKER_SOURCE(walker),prev_vertex, cur_vertex, hop, cur_blk, prev_blk);
                    walk_manager->global_driver->load_block_info_part(newwalker,*cache,cur_cache_index,prev_cache_index);
                }
        }
}   