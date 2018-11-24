import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.pascal_voc import Pascal_voc
slim = tf.contrib.slim
from utils.timer import Timer
import subprocess
from tensorflow.python import debug as tf_debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch_gpu', required=True, type=int, help="watch gpu id filled Set it the same as visible gpu id")
    parser.add_argument('--debug', default=True, type=bool)
    parser.add_argument('--stop_globalstep', default=1000, type=int)
    parser.add_argument('--checkpoint_dir', default="checkpoint_dir",type=str)
    parser.add_argument('--task_index',default=0, type=int)
    
    prof_save_step = cfg.PROFILER_SAVE_STEP #120
    sum_save_step = cfg.SUMMARY_SAVE_STEP #500
    FLAGS, unparsed = parser.parse_known_args()
    
    initial_learning_rate = cfg.LEARNING_RATE
    decay_steps = cfg.DECAY_STEPS
    decay_rate = cfg.DECAY_RATE
    staircase = cfg.STAIRCASE
    
    ########################dir#######################################
    singlepipe_dir = "Single_Pipe_train_logs"
    if not os.path.exists(singlepipe_dir):
        os.makedirs(singlepipe_dir)
    
    inside_bsnQnM_dir = "Single_Pipe"+cfg.BS_NT_MUL_PREFIX
    logrootpath = os.path.join(singlepipe_dir, inside_bsnQnM_dir)
    if not os.path.exists(logrootpath):
        os.makedirs(logrootpath)
    
    fpslog_name = "Single_Pipe"+cfg.BS_NT_MUL_PREFIX+ "fps_log.txt"
    concated_path = logrootpath + "/" + fpslog_name

    checkpoint_dir = FLAGS.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    gpulog_name = "Single_Pipe"+"gpu"+str(FLAGS.watch_gpu)+cfg.BS_NT_MUL_PREFIX+"_gpulog.txt"

    #########################pipeline###################################
    tf.reset_default_graph()
    
    image_producer = Pascal_voc('train')
    
    (current_index, image, label) = image_producer.get_one_image_label_element()
    current_index = tf.Print(current_index, data=[current_index],
                     message="CURRENT INDEX OF IMAGE IS :")

    image_shape = (image_producer.image_size, image_producer.image_size, 3)

    label_size = (image_producer.cell_size, image_producer.cell_size, 25)  # possible value is 0 or 1

    processed_queue = tf.FIFOQueue(capacity=int(image_producer.batch_size * 1.5),
    shapes = [image_shape, label_size],
    dtypes = [tf.float32, tf.float32],
    name = 'processed_queue')

    enqueue_processed_op = processed_queue.enqueue([image, label])

    num_enqueue_threads = min(image_producer.num_enqueue_threads, image_producer.gt_labels_length)

    queue_runner = tf.train.QueueRunner(processed_queue, [enqueue_processed_op] * num_enqueue_threads)
    tf.train.add_queue_runner(queue_runner)

    (images, labels) = processed_queue.dequeue_many(image_producer.batch_size)
    if FLAGS.debug == True:
        labels = tf.Print(labels, data=[processed_queue.size()],
                          message="Worker %d get_batch(), BatchSize %d, Queues left:" % (
                              FLAGS.task_index, cfg.BATCH_SIZE))

    #########################graph###################################
    
    with tf.device("/device:GPU:"+str(FLAGS.watch_gpu)):
        yolo = YOLONet(images, labels)
        
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, decay_steps,
            decay_rate, staircase, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(
            yolo.total_loss, optimizer, global_step=global_step)
        
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    # config.gpu_options.allocator_type = 'BFC'
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    

    #######################train#####################################
    
    
    sess = tf.InteractiveSession()

    if FLAGS.debug and FLAGS.tensorboard_debug_address:
        raise ValueError(
            "The --debug and --tensorboard_debug_address flags are mutually "
            "exclusive.")
    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=FLAGS.ui_type)
    elif FLAGS.tensorboard_debug_address:
        sess = tf_debug.TensorBoardDebugWrapperSession(
            sess, FLAGS.tensorboard_debug_address)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    start_global_step_value = sess.run(global_step)
    timer = Timer(start_global_step_value)

    iters_per_toc = 20
    txtForm = "Training speed: local avg %f fps, global %f fps, loss %f, global step: %d, predict to wait %s"
    local_max_iter = FLAGS.stop_globalstep - start_global_step_value
    
    timer.tic()
    yolo_loss, global_step_value, _ = sess.run([yolo.total_loss, global_step, train_op])
    n = 1
    while n<1000:
        n = n + 1
        if n > 0 and n % iters_per_toc == 0:
            if n > 0 and n % iters_per_toc == 0:
                local_avg_fps, global_avg_fps = timer.toc(iters_per_toc, global_step_value)
                timetowait = timer.remain(n, local_max_iter)
        
                txtData = local_avg_fps, global_avg_fps, yolo_loss, global_step_value, timetowait
                print(txtForm % txtData)
                
                with open(concated_path, 'a+') as log:
                    log.write("%.4f,%.4f,%.4f,%d,%s\n" % txtData)
                
                timer.tic()

        yolo_loss, global_step_value, _ = sess.run([yolo.total_loss, global_step, train_op])
    
    coord.request_stop()
    coord.join(threads)
        
    print('Done debugging training.')


if __name__ == '__main__':
    main()
