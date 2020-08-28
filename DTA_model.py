import tensorflow as tf
SMI_DIM = 65
PRO_DIM = 21
FILTERNUM = 32
SMI_FILTER_SIZE = [4,6,8]
PRO_FILTER_SIZE = [4,8,12]
EMBEDDING_DIM = 128
OUTPUT_NODE = 1
FC_SIZE = [1024, 1024, 512]


def variable_summaries(var, name):
    with tf.name_scope("summaries"):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean/" + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev/" + name, stddev)


def inference(smi_tensor, pro_tensor, regularizer=None,
              keep_prob=1, trainlabel=False):
    with tf.variable_scope('smi_embedding', reuse=tf.AUTO_REUSE):
        smi_wordembedding = tf.get_variable(
            "smi_word_embedding", [SMI_DIM, EMBEDDING_DIM])
        smi_embedding = tf.nn.embedding_lookup(smi_wordembedding, smi_tensor)

        pro_wordembedding = tf.get_variable(
            "pro_word_embedding", [PRO_DIM, EMBEDDING_DIM])
        pro_embedding = tf.nn.embedding_lookup(pro_wordembedding, pro_tensor)
    with tf.variable_scope('drug_conv'):
        conv1_weights = tf.get_variable(
            "weight1", [SMI_FILTER_SIZE[0], EMBEDDING_DIM, FILTERNUM],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_weights = tf.get_variable(
            "weight2", [SMI_FILTER_SIZE[1], FILTERNUM, FILTERNUM * 2],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_weights = tf.get_variable(
            "weight3", [SMI_FILTER_SIZE[2], FILTERNUM * 2, FILTERNUM * 3],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            "bias1", [FILTERNUM], initializer=tf.constant_initializer(0.1))
        conv2_biases = tf.get_variable(
            "bias2", [FILTERNUM * 2], initializer=tf.constant_initializer(0.1))
        conv3_biases = tf.get_variable(
            "bias3", [FILTERNUM * 3], initializer=tf.constant_initializer(0.1))
        variable_summaries(conv1_weights, 'W1')
        variable_summaries(conv2_weights, 'W2')
        variable_summaries(conv3_weights, 'W3')
        variable_summaries(conv1_biases, 'b1')
        variable_summaries(conv2_biases, 'b2')
        variable_summaries(conv3_biases, 'b3')
        smi_conv1 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv1d(
                    smi_embedding,
                    conv1_weights,
                    stride=1,
                    padding='VALID'),
                conv1_biases))
        smi_conv1 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv1d(
                    smi_conv1,
                    conv2_weights,
                    stride=1,
                    padding='VALID'),
                conv2_biases))
        smi_conv1 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv1d(
                    smi_conv1,
                    conv3_weights,
                    stride=1,
                    padding='VALID'),
                conv3_biases))
        # drug_feature = tf.squeeze(
        #     tf.nn.pool(
        #         smi_conv1,
        #         window_shape=[85],
        #         pooling_type="MAX",
        #         padding='VALID'),
        #     1)

    with tf.variable_scope('protein_conv'):
        conv1_weights = tf.get_variable(
            "weight", [PRO_FILTER_SIZE[0], EMBEDDING_DIM, FILTERNUM],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_weights = tf.get_variable(
            "weight1", [PRO_FILTER_SIZE[1], FILTERNUM, FILTERNUM * 2],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_weights = tf.get_variable(
            "weight2", [PRO_FILTER_SIZE[2], FILTERNUM * 2, FILTERNUM * 3],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [FILTERNUM], initializer=tf.constant_initializer(0.1))
        conv2_biases = tf.get_variable("bias1", [FILTERNUM * 2], initializer=tf.constant_initializer(0.1))
        conv3_biases = tf.get_variable("bias2", [FILTERNUM * 3], initializer=tf.constant_initializer(0.1))
        pro_conv1 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv1d(pro_embedding, conv1_weights, stride=1, padding='SAME'), conv1_biases))
        pro_conv1 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv1d(pro_conv1, conv2_weights, stride=1, padding='SAME'), conv2_biases))
        pro_conv1 = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv1d(pro_conv1, conv3_weights, stride=1, padding='SAME'), conv3_biases))
        # pro_pool = tf.nn.pool(pro_conv3, window_shape=[1179], pooling_type="MAX", padding='VALID')
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(conv1_weights))
            tf.add_to_collection('losses', regularizer(conv2_weights))
            tf.add_to_collection('losses', regularizer(conv3_weights))
        variable_summaries(conv1_weights, 'W1')
        variable_summaries(conv2_weights, 'W2')
        variable_summaries(conv3_weights, 'W3')
        variable_summaries(conv1_biases, 'b1')
        variable_summaries(conv2_biases, 'b2')
        variable_summaries(conv3_biases, 'b3')
    with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weight", [smi_conv1.get_shape()[2], pro_conv1.get_shape()[2]],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(weights))
        biases = tf.get_variable("bias",
                                 [pro_conv1.get_shape()[2]],
                                 initializer=tf.constant_initializer(0.1))
        atten1 = tf.nn.relu(tf.einsum('ajk,ki->aji', smi_conv1, weights))+biases
        atten2 = tf.nn.relu(tf.einsum('ajk,ki->aji', pro_conv1, weights))+biases
        variable_summaries(weights, "DT_att_weight")
        variable_summaries(biases, "DT_att_biases")
        alph = tf.nn.tanh(
            tf.einsum('aji,aik->ajk', atten1, tf.transpose(atten2, [0, 2, 1])))
        alphdrug = tf.nn.tanh(tf.reduce_sum(alph, 2))
        alphprotein = tf.nn.tanh(tf.reduce_sum(alph, 1))

        alphdrug = tf.tile(tf.expand_dims(alphdrug, 2), [1, 1, smi_conv1.get_shape()[2]])
        alphprotein = tf.tile(tf.expand_dims(alphprotein, 2), [1, 1, pro_conv1.get_shape()[2]])
        drug_feature = tf.multiply(alphdrug, smi_conv1)
        pretein_feature = tf.multiply(alphprotein, pro_conv1)
        drug_feature = tf.squeeze(
            tf.nn.pool(drug_feature, window_shape=[drug_feature.get_shape()[1]], pooling_type="MAX",
                       padding='VALID'), 1)
        pretein_feature = tf.squeeze(tf.nn.pool(pretein_feature, window_shape=[pretein_feature.get_shape()[1]], pooling_type="MAX", padding='VALID'),1)

    with tf.name_scope("concat_layer"):
        pair_feature = tf.concat([drug_feature, pretein_feature], 1)

    with tf.variable_scope('deep-fc-layer', reuse=tf.AUTO_REUSE):
        fc1_weights = tf.get_variable("weight1", [int(pair_feature.get_shape()[1]), FC_SIZE[0]],
                                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        fc1_biases = tf.get_variable("bias1",
                                     [FC_SIZE[0]],
                                     initializer=tf.constant_initializer(0.1),
                                     dtype=tf.float32)
        fc2_weights = tf.get_variable("weight2", [FC_SIZE[0], FC_SIZE[1]],
                                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        fc2_biases = tf.get_variable("bias2",
                                     [FC_SIZE[1]],
                                     initializer=tf.constant_initializer(0.1),
                                     dtype=tf.float32)
        fc3_weights = tf.get_variable("weight3", [FC_SIZE[1], FC_SIZE[2]],
                                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        fc3_biases = tf.get_variable("bias3",
                                     [FC_SIZE[2]],
                                     initializer=tf.constant_initializer(0.1),
                                     dtype=tf.float32)
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
            tf.add_to_collection('losses', regularizer(fc2_weights))
            tf.add_to_collection('losses', regularizer(fc3_weights))
        variable_summaries(fc1_weights, 'W1')
        variable_summaries(fc2_weights, 'W2')
        variable_summaries(fc3_weights, 'W3')

        variable_summaries(fc1_biases, 'b1')
        variable_summaries(fc2_biases, 'b2')
        variable_summaries(fc3_biases, 'b3')

        fc = tf.nn.leaky_relu(
            tf.matmul(
                tf.cast(
                    pair_feature,
                    tf.float32),
                fc1_weights) +
            fc1_biases)
        variable_summaries(fc, 'fc1')
        # fc = tf.layers.batch_normalization(fc, training=trainlabel)
        fc = tf.nn.dropout(fc, keep_prob)

        fc = tf.nn.leaky_relu(
            tf.matmul(fc, fc2_weights) + fc2_biases)
        variable_summaries(fc, 'fc2')
        # fc = tf.layers.batch_normalization(fc, training=trainlabel)
        fc = tf.nn.dropout(fc, keep_prob)

        fc = tf.nn.leaky_relu(
            tf.matmul(fc, fc3_weights) + fc3_biases)
        variable_summaries(fc, 'fc3')
        # fc = tf.layers.batch_normalization(fc, training=trainlabel)

    with tf.variable_scope('y-layer', reuse=tf.AUTO_REUSE):
        y_weights = tf.get_variable("weight", [FC_SIZE[2], 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
        y_biases = tf.get_variable(
            "bias",
            1,
            initializer=tf.constant_initializer(5),
            dtype=tf.float32)
        variable_summaries(y_weights, 'W')
        variable_summaries(y_biases, 'b')
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(y_weights))
        logit = tf.matmul(fc, y_weights) + y_biases
    return drug_feature, pretein_feature, logit
