import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" # level 取"1":显示所有信息，"2":只显示 warning 和 Error, "3":只显示 Error

import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from hccl.manage.api import get_local_rank_id
from hccl.manage.api import get_rank_size
from hccl.manage.api import get_rank_id
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 14, '')
tf.app.flags.DEFINE_integer('num_readers', 16, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')

import model
import icdar

FLAGS = tf.app.flags.FLAGS


def tower_loss(images, score_maps, geo_maps, training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry = model.model(images, is_training=True)

    model_loss = model.loss(score_maps, f_score,
                            geo_maps, f_geometry,
                            training_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    npu_int = npu_ops.initialize_system()
    npu_shutdown = npu_ops.shutdown_system()

    config = tf.ConfigProto()
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #关闭remap开关
    init_sess = tf.Session(config=config)
    init_sess.run(npu_int)

    import os
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore and get_rank_id() == 0:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    lr_scaler = get_rank_size()
    global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate * lr_scaler, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    opt = NPUDistributedOptimizer(opt)

    # split
    input_images_split = tf.split(input_images, get_rank_size())
    input_score_maps_split = tf.split(input_score_maps, get_rank_size())
    input_geo_maps_split = tf.split(input_geo_maps, get_rank_size())
    input_training_masks_split = tf.split(input_training_masks, get_rank_size())

    tower_grads = []
    reuse_variables = None
    for i in range(get_rank_size()):
        #with tf.device('/gpu:%d' % i):
        with tf.name_scope('model_%d' % i) as scope:
            iis = input_images_split[i]
            isms = input_score_maps_split[i]
            igms = input_geo_maps_split[i]
            itms = input_training_masks_split[i]
            total_loss, model_loss = tower_loss(iis, isms, igms, itms, reuse_variables)
            batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
            reuse_variables = True

            grads = opt.compute_gradients(total_loss)
            tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    #summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    #saver = tf.train.Saver(tf.global_variables())
    #summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        #hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=FLAGS.max_steps // get_rank_size()),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'model_loss': model_loss, 'total_loss': total_loss, },
                                   every_n_iter=10),
    ]

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(get_local_rank_id())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    checkpoint_dir = FLAGS.checkpoint_path if get_rank_id() == 0 else None
    data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                    input_size=FLAGS.input_size,
                                    batch_size=FLAGS.batch_size_per_gpu * get_rank_size())
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        mon_sess.run(init)
        if FLAGS.pretrained_model_path is not None:
            variable_restore_op(mon_sess)

        while not mon_sess.should_stop():
            # Run a training step synchronously.
            data = next(data_generator)
            print('sess.run, rank: {}'.format(get_rank_id()))
            mon_sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                        input_score_maps: data[2],
                                                                        input_geo_maps: data[3],
                                                                        input_training_masks: data[4]})
    init_sess.run(npu_shutdown)
    init_sess.close()

if __name__ == '__main__':
    tf.app.run()
