import tensorflow as tf
import dataset
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
TRAIN_PATH='/home/qun/Fer2013_20180526/datasets/train'
# VALIDATION_PATH='/home/qun/Fer2013_20180526/datasets/val'
TEST_PATH='/home/qun/Fer2013_20180526/datasets/test'
IMAGE_SIZE=48
NUM_CHANNELS=1
BATCH_SIZE=294   #254    #294
TRAIN_STEP=100


NUM_CLASSES=2
lambd=0.001
NUM_FEATURES=16

def variable_weights(shape):
    init=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lambd)(init))
    return init


def variable_biase(shape):
    init=tf.constant(0.1,shape=shape)
    return tf.Variable(init)


def convolution(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


def pool_max_2X2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1,2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def forward(x,tst,iterr):
    keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('conv1_1'):
        W_conv1_1=variable_weights([3,3,1,NUM_FEATURES])
        b_conv1_1=variable_biase([NUM_FEATURES])
        t_conv1_1=convolution(x,W_conv1_1)+b_conv1_1
        h_conv1_1=tf.nn.relu(t_conv1_1)

    with tf.name_scope('conv1_2'):
        W_conv1_2=variable_weights([3,3,NUM_FEATURES,NUM_FEATURES])
        b_conv1_2=variable_biase([NUM_FEATURES])
        t_conv1_2=convolution(h_conv1_1,W_conv1_2)
        h_bn1_2, update_ema1_2 = batchnorm(t_conv1_2, tst, iterr, b_conv1_2, convolutional=True)
        h_conv1_2=tf.nn.relu(h_bn1_2)

    with tf.name_scope('pool1'):
        h_pool1=pool_max_2X2(h_conv1_2)

    with tf.name_scope('dropout1'):
        h_drop1=tf.nn.dropout(h_pool1,keep_prob)

    with tf.name_scope('conv2_1'):
        W_conv2_1=variable_weights([3,3,NUM_FEATURES,2*NUM_FEATURES])
        b_conv2_1=variable_biase([2*NUM_FEATURES])
        t_conv2_1 = convolution(h_drop1, W_conv2_1)
        h_bn2_1, update_ema2_1 = batchnorm(t_conv2_1, tst, iterr, b_conv2_1, convolutional=True)
        h_conv2_1 = tf.nn.relu(h_bn2_1)

    with tf.name_scope('conv2_2'):
        W_conv2_2=variable_weights([3,3,2*NUM_FEATURES,2*NUM_FEATURES])
        b_conv2_2=variable_biase([2*NUM_FEATURES])
        t_conv2_2 = convolution(h_conv2_1, W_conv2_2)
        h_bn2_2, update_ema2_2 = batchnorm(t_conv2_2, tst, iterr, b_conv2_2, convolutional=True)
        h_conv2_2 = tf.nn.relu(h_bn2_2)

    with tf.name_scope('pool2'):
        h_pool2=pool_max_2X2(h_conv2_2)

    with tf.name_scope('drop2'):
        h_drop2=tf.nn.dropout(h_pool2,keep_prob)

    with tf.name_scope('conv3_1'):
        W_conv3_1=variable_weights([3,3,2*NUM_FEATURES,2*2*NUM_FEATURES])
        b_conv3_1=variable_biase([2*2*NUM_FEATURES])
        t_conv3_1 = convolution(h_drop2, W_conv3_1)
        h_bn3_1, update_ema3_1 = batchnorm(t_conv3_1, tst, iterr, b_conv3_1, convolutional=True)
        h_conv3_1 = tf.nn.relu(h_bn3_1)

    with tf.name_scope('conv3_2'):
        W_conv3_2=variable_weights([3,3,2*2*NUM_FEATURES,2*2*NUM_FEATURES])
        b_conv3_2=variable_biase([2*2*NUM_FEATURES])
        t_conv3_2 = convolution(h_conv3_1, W_conv3_2)
        h_bn3_2, update_ema3_2 = batchnorm(t_conv3_2, tst, iterr, b_conv3_2, convolutional=True)
        h_conv3_2 = tf.nn.relu(h_bn3_2)

    with tf.name_scope('pool3'):
        h_pool3=pool_max_2X2(h_conv3_2)

    with tf.name_scope('drop3'):
        h_drop3=tf.nn.dropout(h_pool3,keep_prob)

    with tf.name_scope('conv4_1'):
        W_conv4_1=variable_weights([3,3,2*2*NUM_FEATURES,2*2*2*NUM_FEATURES])
        b_conv4_1=variable_biase([2*2*2*NUM_FEATURES])
        t_conv4_1 = convolution(h_drop3, W_conv4_1)
        h_bn4_1, update_ema4_1 = batchnorm(t_conv4_1, tst, iterr, b_conv4_1, convolutional=True)
        h_conv4_1 = tf.nn.relu(h_bn4_1)

    with tf.name_scope('conv4_2'):
        W_conv4_2=variable_weights([3,3,2*2*2*NUM_FEATURES,2*2*2*NUM_FEATURES])
        b_conv4_2=variable_biase([2*2*2*NUM_FEATURES])
        t_conv4_2 = convolution(h_conv4_1, W_conv4_2)
        h_bn4_2, update_ema4_2 = batchnorm(t_conv4_2, tst, iterr, b_conv4_2, convolutional=True)
        h_conv4_2 = tf.nn.relu(h_bn4_2)

    with tf.name_scope('pool4'):
        h_pool4=pool_max_2X2(h_conv4_2)

    with tf.name_scope('drop4'):
        h_drop4=tf.nn.dropout(h_pool4,keep_prob)

    with tf.name_scope('fc1'):
        W_fc1=variable_weights([3*3*2*2*2*NUM_FEATURES,512])
        b_fc1=variable_biase([512])
        h_drop4_flat=tf.reshape(h_drop4,[-1,3*3*2*2*2*NUM_FEATURES])
        h_fc1=tf.nn.relu(tf.matmul(h_drop4_flat,W_fc1)+b_fc1)

    with tf.name_scope('drop5'):
        h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

    with tf.name_scope('fc2'):
        W_fc2=variable_weights([512,256])
        b_fc2=variable_biase([256])
        h_fc2=tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

    with tf.name_scope('drop6'):
        h_fc2_drop=tf.nn.dropout(h_fc2,keep_prob)

    with tf.name_scope('fc3'):
        W_fc3=variable_weights([256,128])
        b_fc3=variable_biase([128])
        h_fc3=tf.matmul(h_fc2_drop,W_fc3)+b_fc3

    with tf.name_scope('drop7'):
        h_fc3_drop=tf.nn.dropout(h_fc3,keep_prob)

    with tf.name_scope('fc3'):
        W_fc3=variable_weights([128,2])
        b_fc3=variable_biase([2])
        y_conv=tf.matmul(h_fc3_drop,W_fc3)+b_fc3

    update_ema = tf.group(update_ema1_2, update_ema2_1, update_ema2_2, update_ema3_1,
                          update_ema3_2,update_ema4_1,update_ema4_2)

    return y_conv,keep_prob,update_ema


def draw_confusion_matrix(y_pred,y_true):

    cm=confusion_matrix(y_pred=y_pred,y_true=y_true)
    print("Confusion matrix")
    print(cm)
    target_names=['hugao','feihugao']
    print(classification_report(y_true, y_pred, target_names=target_names))

    # plt.matshow(confusion_matrics)
    #
    # # Make various adjustments to the plot.
    # plt.colorbar()
    # tick_marks = np.arange(NUM_CLASSES)
    # plt.xticks(tick_marks, range(NUM_CLASSES))
    # plt.yticks(tick_marks, range(NUM_CLASSES))
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    #
    # # Ensure the plot is shown correctly with multiple plots
    # # in a single Notebook cell.
    # plt.show()


def main(_):
    step = tf.placeholder(tf.int32)
    tst=tf.placeholder(tf.bool)
    classes = ['1','2']
    train_set=dataset.read_train_set(TRAIN_PATH,IMAGE_SIZE,classes)
    # validation_set=dataset.read_validation_set(VALIDATION_PATH,IMAGE_SIZE,classes)
    x=tf.placeholder(tf.float32,shape=[None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name='input_x')
    y_=tf.placeholder(tf.float32,shape=[None,2],name='truelabel_y_')
    y_conv,keep_prob,update_moving_averages=forward(x,tst,step)
    with tf.name_scope('loss'):
        cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
        tf.add_to_collection('losses',cross_entropy)
        losses=tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss',losses)

    with tf.name_scope('Adam_optimizer'):
        lr = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1 / math.e)
        train_step=tf.train.AdamOptimizer(lr).minimize(losses)

    with tf.name_scope('accuracy'):
        y_pred=tf.argmax(tf.nn.softmax(y_conv),1)
        y_true=tf.argmax(y_,1)
        cross_prediction=tf.equal(y_pred,y_true)
        accuracy=tf.reduce_mean(tf.cast(cross_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
        merged=tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer_train=tf.summary.FileWriter(r'log_train',sess.graph)
        # summary_writer_val=tf.summary.FileWriter(r'log_val',sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(TRAIN_STEP):
            train_images,train_labels,_,_=train_set.next_batch(BATCH_SIZE)
            if i%1==0:

                feed_dict_train0={x:train_images,y_:train_labels,keep_prob:1,step:i,tst:False}
                summary_train,train_accuracy=sess.run([merged,accuracy],feed_dict=feed_dict_train0)
                summary_writer_train.add_summary(summary_train, i)

                # val_images,val_labels,_,_=validation_set.next_batch(BATCH_SIZE)
                # feed_dict_val={x: val_images, y_: val_labels,keep_prob:1,step:i,tst:False}
                # summary_val,val_accuracy=sess.run([merged,accuracy], feed_dict=feed_dict_val)
                # summary_writer_val.add_summary(summary_val,i)
                print('on Step %d:Training accuracy' %i, train_accuracy*100)

                # if i%1000==0:
                #
                #     # y_pred_cls,y_true_cls=sess.run([y_pred,y_true],feed_dict=feed_dict_val)
                #     draw_confusion_matrix(y_pred_cls,y_true_cls)

            feed_dict_train={x:train_images,y_:train_labels,keep_prob:0.5,step:i,tst:False}
            sess.run(train_step,feed_dict=feed_dict_train)

        test_set=dataset.read_test_set(TEST_PATH,IMAGE_SIZE,classes)
        test_images,test_labels,_,_=test_set.next_batch(60)
        feed_dict_test={x:test_images,y_:test_labels,keep_prob:1,step:i,tst:False}
        y_pred_cls,y_true_cls,test_accuracy=sess.run([y_pred,y_true,accuracy],feed_dict=feed_dict_test)
        draw_confusion_matrix(y_pred_cls,y_true_cls)
        print('Test accuracy=%.1ff%%'%(test_accuracy*100))

    summary_writer_train.close()
    # summary_writer_val.close()


if __name__=="__main__":
    tf.app.run()



















