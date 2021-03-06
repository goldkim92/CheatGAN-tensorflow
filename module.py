import tensorflow as tf
from ops import linear, conv2d, deconv2d, relu, lrelu,  batch_norm


def generator(z, options, reuse=False, name='gen'):
    # reuse or not
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
#        x = linear(z, 2*2*(8*options.nf), name='gen_linear') # batch*100
        x = tf.reshape(z, [-1,1,1,options.z_dim]) # 1*1*100
#        x = tf.reshape(x, [options.batch_size,2,2,8*options.nf]) # 2*2*1024
        
        if options.input_size == 64:
            x = relu(batch_norm(deconv2d(x, options.nf, ks=4, s=4, name='gen_deconv0'), 'gen_bn0')) # 4*4*1024
            x = relu(batch_norm(deconv2d(x, 4*options.nf, ks=5, s=2, name='gen_deconv1'), 'gen_bn1')) # 8*8*512
        else:
            x = relu(batch_norm(deconv2d(x, 4*options.nf, ks=4, s=4, name='gen_deconv1'), 'gen_bn1')) # 4*4*512
        x = relu(batch_norm(deconv2d(x, 2*options.nf, ks=5, s=2, name='gen_deconv2'), 'gen_bn2')) # 8*8*256 or 16*16*256
        x = relu(batch_norm(deconv2d(x, options.nf, ks=5, s=2, name='gen_deconv3'), 'gen_bn3')) # 16*16*128 or 32*32*128
        x = deconv2d(x, options.image_c, ks=5, s=2, name='gen_deconv4') # 32*32*1 or 64*64*1
        return tf.nn.tanh(x)


def discriminator(images, options, reuse=False, name='disc'):
    # reuse or not
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        x = lrelu(batch_norm(conv2d(images, options.nf, ks=5, s=2, name='disc_conv1'), 'disc_bn1')) # 16*16*128 or 32*32*128
        x = lrelu(batch_norm(conv2d(x, 2*options.nf, ks=5, s=2, name='disc_conv2'), 'disc_bn2')) # 8*8*256 or 16*16*256
        x = lrelu(batch_norm(conv2d(x, 4*options.nf, ks=5, s=2, name='disc_conv3'), 'disc_bn3')) # 4*4*512 or 8*8*512
        if options.input_size == 64:
            x = lrelu(batch_norm(conv2d(x, options.nf, ks=5, s=2, name='disc_conv0'), 'disc_bn0')) # 4*4*1024
        x = conv2d(x, 1, ks=4, s=1, name='disc_conv4') # 1*1*1
        x = tf.reshape(x, [-1, 1])
#        x = linear(tf.reshape(x, [options.batch_size,2*2*(8*options.nf)]), 1, name='disc_linear') # 100
        
        return tf.nn.sigmoid(x)
        

def wgan_gp_loss(real_img, fake_img, options):
    shape = tf.shape(real_img)
    epsilon = tf.random_uniform([shape[0], 1,1,1])
    hat_img = epsilon * real_img + (1.-epsilon) * fake_img
    d_out = discriminator(hat_img, options, reuse=True, name='disc')
    gradients = tf.gradients(d_out, xs=[hat_img])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    
    return gradient_penalty

def gan_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))

def cls_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))

def recon_loss(image1, image2):
    return tf.reduce_mean(tf.abs(image1 - image2))
