import os
import argparse
import tensorflow as tf
import yolo.dis_config as cfg
from yolo.yolo_net import YOLONet
from utils.pascal_voc import Pascal_voc
slim = tf.contrib.slim
from utils.timer import Timer
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, default="")
    parser.add_argument("--task_index", type=int, default=0)
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--stop_globalstep', default=2000, type=int)
    parser.add_argument('--checkpoint_dir', default="checkpoint_dir",type=str)
    parser.add_argument('--watch_gpu',required=True ,type=int, help="watch gpu id filled Set it the same as visible gpu id")
    parser.add_argument('--warm_up_step',default = 20, type = int)

    profiler_save_steps = cfg.PROFILER_SAVE_STEP
    summary_save_steps = cfg.SUMMARY_SAVE_STEP
    FLAGS, unparsed = parser.parse_known_args()
    
    ps_hosts = cfg.PS_HOSTS.split(",")
    worker_hosts = cfg.WORKER_HOSTS.split(",")
    
    ps_size = len(ps_hosts)
    workers_size = len(worker_hosts)

    dispipe_dir="DisPipe_"+str(workers_size)+"workers"+str(ps_size)+"ps"+"_train_logs"
    if not os.path.exists(dispipe_dir):
        os.makedirs(dispipe_dir)

    inside_bsnQnM_dir = "Dis_Pipe_"+cfg.BS_NT_MUL_PREFIX
    logrootpath = os.path.join(dispipe_dir, inside_bsnQnM_dir)
    if not os.path.exists(logrootpath):
        os.makedirs(logrootpath)

   
    fpslog_name = "DisPipe_" +"task"+str(FLAGS.task_index) +cfg.BS_NT_MUL_PREFIX+ "_fpslog.txt"
    concated_path = logrootpath + "/" + fpslog_name

    checkpoint_dir = FLAGS.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    gpulog_name = "DisPipe" + "_task" + str(FLAGS.task_index)+"gpu"+str(FLAGS.watch_gpu)+cfg.BS_NT_MUL_PREFIX + "_gpulog.txt"

    ############
    ###########################gpulog#################################
    def start_gpulog(path, fname):
        # has to be called before start of training
        gpuinfo_path = path + "/" + fname
        with open(gpuinfo_path, 'w'):
            argument = 'timestamp,count,gpu_name,gpu_bus_id,memory.total,memory.used,utilization.gpu,utilization.memory'
        try:
            proc = subprocess.Popen(
                ['nvidia-smi --format=csv --query-gpu=%s %s %s %s' % (argument, ' -l', '-i '+ str(FLAGS.watch_gpu), '-f ' + gpuinfo_path)],shell=True)
        except KeyboardInterrupt:
            try:
                proc.kill()
            except OSError:
                pass
                proc.wait()
        return proc
    
    initial_learning_rate = cfg.LEARNING_RATE
    decay_steps = cfg.DECAY_STEPS
    decay_rate = cfg.DECAY_RATE
    staircase = cfg.STAIRCASE
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    print('Start training ...')
    
    ###############################pipeline###########################################
    tf.reset_default_graph()
    
    image_producer = Pascal_voc('train')
    
    (image, label) = image_producer.get_one_image_label_element()

    image_shape = (image_producer.image_size, image_producer.image_size, 3)  # possible value is a int number

    label_size = (image_producer.cell_size, image_producer.cell_size, 25)  # possible value is 0 or 1

    processed_queue = tf.FIFOQueue(capacity=int(image_producer.batch_size * cfg.MUL_QUEUE_BATCH),shapes = [image_shape, label_size],dtypes = [tf.float32, tf.float32],name = 'processed_queue')

    enqueue_processed_op = processed_queue.enqueue([image, label])

    num_enqueue_threads = min(image_producer.num_enqueue_threads, image_producer.gt_labels_length)

    queue_runner = tf.train.QueueRunner(processed_queue, [enqueue_processed_op] * num_enqueue_threads)
    tf.train.add_queue_runner(queue_runner)

    (images, labels) = processed_queue.dequeue_many(image_producer.batch_size)
    ##############################################################################

    #############################Parameters#######################################
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
        yolo = YOLONet(images, labels)
        # print('allocate variable and tensor successfuly')

        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, decay_steps,
            decay_rate, staircase, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(
            yolo.total_loss, optimizer, global_step=global_step)
    ################################################################################

    #############################loghook############################################
    profiler_hook = tf.train.ProfilerHook(save_steps=profiler_save_steps, output_dir=logrootpath, show_memory=True,show_dataflow=True)
    summary_op = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(save_steps=summary_save_steps, output_dir=logrootpath, summary_op=summary_op)

    if FLAGS.debug == True:
        tensors_to_log = [global_step, yolo.total_loss]
        def formatter(curvals):
            print("Global step %d, Loss %f!" % (
                curvals[global_step], curvals[yolo.total_loss]))
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100, formatter=formatter)
        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.stop_globalstep), logging_hook, profiler_hook, summary_hook]
      
    else:
        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.stop_globalstep), profiler_hook, summary_hook]
    
    # config.gpu_options.allocator_type = 'BFC'
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    proc = start_gpulog(logrootpath, gpulog_name)
    ################################################################################
    
    ###########################train####################################################
    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0), config=config,hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir,save_checkpoint_secs=3600) as sess:
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
        start_global_step_value = sess.run(global_step)
        timer = Timer()

        iters_per_toc = 20
        txtForm = "Training speed:global step %d, local avg %f fps, global %f fps, loss %f"

        #run and log
        n = 0
        while not sess.should_stop():
            n = n + 1
            if n==FLAGS.warm_up_step:
                start_global_step_value = sess.run(global_step)
                timer.tic(global_restart=True, start_global_step_value = start_global_step_value)
            if n % iters_per_toc ==0:
                timer.tic()

            yolo_loss, global_step_value, _ = sess.run([yolo.total_loss, global_step, train_op])
            
            if n % iters_per_toc == 0:
                local_avg_fps, global_avg_fps = timer.toc(iters_per_toc, global_step_value)
                txtData = global_step_value, local_avg_fps, global_avg_fps, yolo_loss
                print(txtForm % txtData)
                with open(concated_path, 'a+') as log:
                        log.write("%d,%.4f,%.4f,%.4f\n" % txtData)
 
        coord.request_stop()
        coord.join(threads)
    
    print('Done training.')
   # try:
   #     proc.terminate()
   # except OSError:
   #     pass
   #     print("Kill subprocess failed. Kill nvidia-smi mannually")
        
if __name__ == '__main__':
    main()
