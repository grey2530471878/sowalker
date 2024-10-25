#ifndef _GRAPH_DRIVER_H_
#define _GRAPH_DRIVER_H_

#include "cache.hpp"
#include "util/io.hpp"
#include "api/graph_buffer.hpp"
#include "api/types.hpp"
#include "metrics/metrics.hpp"

/** graph_driver
 * This file contribute to define the operations of how to read from disk
 * or how to write graph data into disk
 */

class graph_driver {
private:
    int vertdesc, edgedesc, degdesc, whtdesc;  /* the beg_pos, csr, degree file descriptor */
    metrics &_m;
    bool _weighted;
public:
#ifdef IO_BAND
    eid_t total_bytes;
#endif
    graph_driver(graph_config *conf, metrics &m) : _m(m)
    {
        vertdesc = edgedesc = whtdesc = 0;
        this->setup(conf);
#ifdef IO_BAND
        total_bytes=0;
#endif
    }

    graph_driver(metrics &m) : _m(m) {
        vertdesc = edgedesc = whtdesc = 0;
    }
    void setup(graph_config *conf) {
        this->destory();
        std::string beg_pos_name = get_beg_pos_name(conf->base_name);
        std::string csr_name = get_csr_name(conf->base_name);
        logstream(LOG_DEBUG) << "load beg_pos_name : " << beg_pos_name << ", csr_name : " << csr_name << std::endl;
        vertdesc = open(beg_pos_name.c_str(), O_RDONLY);
        edgedesc = open(csr_name.c_str(), O_RDONLY);
        _weighted = conf->is_weighted;
        if (_weighted)
        {
            std::string weight_name = get_weights_name(conf->base_name);
            if(test_exists(weight_name)) whtdesc = open(weight_name.c_str(), O_RDONLY);
        }
    }

    void load_block_info(graph_cache &cache, graph_block *global_blocks, bid_t cache_index, bid_t block_index)
    {
        _m.start_time("load_block_info");
#ifdef PROF_STEPS
        std::cout << "run_steps_load_block_info" << std::endl;
#endif
        cache.cache_blocks[cache_index].block = &global_blocks->blocks[block_index];
        cache.cache_blocks[cache_index].block->status = ACTIVE;
        cache.cache_blocks[cache_index].block->cache_index = cache_index;

        cache.cache_blocks[cache_index].beg_pos = (eid_t *)realloc(cache.cache_blocks[cache_index].beg_pos, (global_blocks->blocks[block_index].nverts + 1) * sizeof(eid_t));
        load_block_vertex(vertdesc, cache.cache_blocks[cache_index].beg_pos, global_blocks->blocks[block_index]);
        load_block_edge(edgedesc, cache.cache_blocks[cache_index].csr, global_blocks->blocks[block_index]);
#ifdef IO_UTE
        std::cout<<"current load block is"<<block_index<<";nedges is:"<<global_blocks->blocks[block_index].nedges<<std::endl;
#endif
#ifdef IO_BAND
        total_bytes+=((global_blocks->blocks[block_index].nverts+1)*sizeof(eid_t)+(global_blocks->blocks[block_index].nedges)*sizeof(vid_t))/(1024*1024);
#endif
        if(_weighted) {
            cache.cache_blocks[cache_index].weights = (real_t *)realloc(cache.cache_blocks[cache_index].weights, global_blocks->blocks[block_index].nedges * sizeof(real_t));
            load_block_weight(whtdesc, cache.cache_blocks[cache_index].weights, global_blocks->blocks[block_index]);
        }

#ifdef PROF_METRIC
        cache.cache_blocks[cache_index].block->update_loaded_count();
#endif
        _m.stop_time("load_block_info");
    }


    void destory() {
        if(vertdesc > 0) close(vertdesc);
        if(edgedesc > 0) close(edgedesc);
        if(_weighted) {
            if(whtdesc > 0) close(whtdesc);
        }
    }
    void load_block_info_part(walker_t walk,graph_cache &cache,bid_t curc,bid_t prec)
    {
        _m.start_time("load_block_info_part");
        #ifdef PROF_STEPS
        std::cout << "run_steps_load_block_info_part" << std::endl;
        #endif
        vid_t curv=walk.current,prev=walk.previous;
        bid_t curb=walk.cur_index,preb=walk.prev_index;
        vid_t startv=cache.cache_blocks[curc].block->start_vert;
        eid_t starte=cache.cache_blocks[curc].block->start_edge;
        vid_t pstartv=cache.cache_blocks[prec].block->start_vert;
        eid_t pstarte=cache.cache_blocks[prec].block->start_edge;
        if(cache.cache_blocks[curc].beg_pos==NULL)
        {
            cache.cache_blocks[curc].beg_pos = (eid_t *)realloc(cache.cache_blocks[curc].beg_pos, (cache.cache_blocks[curc].block->nverts + 1) * sizeof(eid_t));
        }
        if(cache.cache_blocks[prec].beg_pos==NULL)
        {
            cache.cache_blocks[prec].beg_pos = (eid_t *)realloc(cache.cache_blocks[prec].beg_pos, (cache.cache_blocks[prec].block->nverts + 1) * sizeof(eid_t));
        }
        if(_weighted&&cache.cache_blocks[curc].weights==NULL)
        {
            cache.cache_blocks[curc].weights=(real_t *)realloc(cache.cache_blocks[curc].weights, cache.cache_blocks[curc].block->nedges * sizeof(real_t));
            load_block_weight(whtdesc, cache.cache_blocks[curc].weights, *(cache.cache_blocks[curc].block));
        }
        //logstream(LOG_INFO) << walk.id <<"  "<< curv <<"  "<<prev<< std::endl;
        if(cache.cache_blocks[curc].block->status==PART){
        load_block_vertex_part(vertdesc,&(cache.cache_blocks[curc].beg_pos[curv-startv]),curv*sizeof(eid_t),2);
        }
        if(cache.cache_blocks[prec].block->status==PART){
        load_block_vertex_part(vertdesc,&(cache.cache_blocks[prec].beg_pos[prev-pstartv]),prev*sizeof(eid_t),2);}
        if(cache.cache_blocks[curc].block->status==PART){
        load_block_edge_part(edgedesc,&(cache.cache_blocks[curc].csr[(cache.cache_blocks[curc].beg_pos[curv-startv]-starte)]),cache.cache_blocks[curc].beg_pos[curv-startv]*sizeof(vid_t),cache.cache_blocks[curc].beg_pos[curv-startv+1]-cache.cache_blocks[curc].beg_pos[curv-startv]);
        }
        if(cache.cache_blocks[prec].block->status==PART){
        load_block_edge_part(edgedesc,&(cache.cache_blocks[prec].csr[cache.cache_blocks[prec].beg_pos[prev-pstartv]-pstarte]),cache.cache_blocks[prec].beg_pos[prev-pstartv]*sizeof(vid_t),cache.cache_blocks[prec].beg_pos[prev-pstartv+1]-cache.cache_blocks[prec].beg_pos[prev-pstartv]);
        }//buf可能有问题
        #ifdef PROF_METRIC
        cache.cache_blocks[cache_index].block->update_loaded_count();
#endif
        _m.stop_time("load_block_info_part");
    }



    void load_block_vertex_part(int fd, eid_t *buf, off_t off,size_t count)
    {
        load_block_range(fd, buf, count, off);
    }
    void load_block_degree_part(int fd, eid_t *buf, off_t off,size_t count) {
        load_block_range(fd, buf, count, off);
    }
    void load_block_edge_part(int fd, vid_t *buf, off_t off,size_t count) {
        load_block_range(fd, buf, count, off);
    }






    void load_block_vertex(int fd, eid_t *buf, const block_t &block) {
        load_block_range(fd, buf, block.nverts + 1, block.start_vert * sizeof(eid_t));
    }
    void load_block_degree(int fd, vid_t *buf, const block_t &block) {
        load_block_range(fd, buf, block.nverts, block.start_vert * sizeof(vid_t));
    }

    void load_block_edge(int fd, vid_t *buf, const block_t &block) {
        load_block_range(fd, buf, block.nedges, block.start_edge * sizeof(vid_t));
    }

    void load_block_weight(int fd, real_t* buf, const block_t& block) {
        load_block_range(fd, buf, block.nedges, block.start_edge * sizeof(real_t));
    }

    void load_block_prob(int fd, real_t* buf, const block_t& block) {
        load_block_range(fd, buf, block.nedges, block.start_edge * sizeof(real_t));
    }

    void load_block_alias(int fd, vid_t* buf, const block_t& block) {
        load_block_range(fd, buf, block.nedges, block.start_edge * sizeof(vid_t));
    }

    template<typename walk_data_t>
    void load_walk(int fd, size_t cnt, size_t loaded_cnt, graph_buffer<walk_data_t> &walks) {
        off_t off = loaded_cnt * sizeof(walk_data_t);
        load_block_range(fd, walks.buffer_begin(), cnt, off);
        walks.set_size(cnt);
    }

    template<typename walk_data_t>
    void dump_walk(int fd, graph_buffer<walk_data_t> &walks) {
        dump_block_range(fd, walks.buffer_begin(), walks.size(), 0);
    }
};

#endif
