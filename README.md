# Verified a few tests
- I made many times the same mistakes. InputPipeline run and after hundreds of iterations, run out of memory. It is because of profiling and summary which take huge memory. So for good performance, do not use summary hook and small amount of profiling hook
- I varified the yolo on pascal voc works on corresponding tensorflow
- Applied inputpipeline and achieve performance improvement on des machine with single GPU pascal, this is so called thread-level parallelism
- Tried gRPC tensorflow by default communication way on 2 des machines, with one GPU on each. Performance is worse than second
- Tried Horovod allreduce on 2 des machines, but because of Network speed, still very slow. better than third, worse than second
- Tried gRPC on Tesla v100 workstation(4 gpu). gRPC by default works fine, because the speed are similiar on ecah machine, but synchronous
- Tried NCCL horovod all reduce on Tesla v100 workstation. Verified 2 things   
   - NCCL is better than gRPC on same workstation   
   - TODO: Scalability graph of both gRPC and NCCL   

# Install dataset and environ
In this horovod test, base test is fixed with warm-up and good local and global fps  
- cd pascalvoc/  
- bash download.sh  to download dataset(500 MB, sorry I only downlodaed one year data, but it is this big)
- (For environment, I only used tensorflow-gpu and opencv. If your environment does not work, try this). 
- cd env, bash build_env.sh


# Aim
- stable 99%-100% GPU usage for 2 tests.

#  Run 2 tests
### 0. start the miniconda environment:
conda info --envs
```bash
base                  *  /macierz/home/s170011/bin/miniconda3
py27i                    /macierz/home/s170011/bin/miniconda3/envs/py27i
py36tf                   /macierz/home/s170011/bin/miniconda3/envs/py36tf
py36tfi                  /macierz/home/s170011/bin/miniconda3/envs/py36tfi
```
cd /macierz/home/s170011/bin/miniconda3/bin
. activate py36tfi
# Don't know why now the py36tf report errors
conda list

### 1. Single machine training 
bash run_single.sh --watch_gpu=0 [--watch-gpu should be the same as the visible gpu]
### 2. Distributed training  
bash ps.sh  
bash worker0.sh --watch-gpu=0  
bash worker1.sh  --watch-gpu=1  
[--watch-gpu should be the same as the visible gpu]

# Change yolo_tensorflow/yolo/config.py
- Change parameters in yolo_tensorflow/yolo/config.py  
- For test1, just ignore PS_HOSTS and WORKER_HOSTS. Only change it for test2
- MUL_QUEUE_BATCH means how many times is the queue size is of batch_size. 2 means 10*2=20
- PROFILER_SAVE_STEP, SUMMARY_SAVE_STEP can not be too small, because I found especially summary take long time
- How does the yolo/config.py look like and where to change? change here  

#############################################################################  
PS_HOSTS  = '172.20.83.210:8897'  

WORKER_HOSTS = '172.20.83.210:8898,172.20.83.202:8898'  

BATCH_SIZE = 10  

NUM_ENQUEUE_THREADS = 2  

MUL_QUEUE_BATCH = 2  

PROFILER_SAVE_STEP = 60  

SUMMARY_SAVE_STEP = 120  

#############################################################################  
