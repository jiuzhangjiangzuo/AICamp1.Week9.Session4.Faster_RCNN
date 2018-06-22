import io
from PIL import Image
import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('gt_input', '', 'Path to the gt input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('images_path', '', 'Path to images_folder')

FLAGS = flags.FLAGS


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_info, path):
    with tf.gfile.FastGFile(os.path.join(path, '{}'.format(image_info['filename'].rstrip())), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = image_info['filename'].encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for (xmin, xmax, ymin, ymax) in zip(image_info['xmin'], image_info['xmax'], image_info['ymin'], image_info['ymax']):
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(b'face')
        classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    path = os.path.join(os.getcwd(), FLAGS.images_path)
    
    with open(FLAGS.gt_input, 'r') as f:
        is_first_image = True

        image_info = {'filename' : '', 'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}

        for line in f:
            inputs = line.split(' ')

            if len(inputs) == 1:
                if inputs[0].rstrip().isdigit():
                    continue

                if not is_first_image:
                    tf_example = create_tf_example(image_info, path)
                    writer.write(tf_example.SerializeToString())

                    image_info = {'filename' : '', 'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}

                image_info['filename'] = inputs[0]
                is_first_image = False
            else:
                x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose, _= inputs
                image_info['xmin'].append(float(x1))
                image_info['xmax'].append(float(x1) + float(w))
                image_info['ymin'].append(float(y1))
                image_info['ymax'].append(float(y1) + float(h))
        

    writer.close()

    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
