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
    # 读出图片
    with tf.gfile.FastGFile(os.path.join(path, '{}'.format(image_info['filename'].rstrip())), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    # 得到图片的长宽
    width, height = image.size
    filename = image_info['filename'].encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    # 归一化一下xmin, xmax, ymin, ymax。使它们分布在0-1之间。
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
    # 指定TFRecord文件输出路径
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # 图片所在的绝对路径
    path = os.path.join(os.getcwd(), FLAGS.images_path)
    
    # 打开GT的文件
    with open(FLAGS.gt_input, 'r') as f:
        is_first_image = True

        # 保存图像的所有信息，包括图像的文件名、一系列的BBOX
        image_info = {'filename' : '', 'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}

        # 穷举文件中的所有行
        for line in f:
            inputs = line.split(' ')

            # 这一行是文件名或者是bbox的数量
            if len(inputs) == 1:
                # 如果是bbox的数量，就不做任何操作
                if inputs[0].rstrip().isdigit():
                    continue

                # 如果是文件名，且不表示第一张图片，把上一张图片的信息转换为tf.example.
                if not is_first_image:
                    tf_example = create_tf_example(image_info, path)
                    writer.write(tf_example.SerializeToString())

                    image_info = {'filename' : '', 'xmin': [], 'xmax': [], 'ymin': [], 'ymax': []}

                # 更新 image_info 里面的 filename.
                image_info['filename'] = inputs[0]
                is_first_image = False

            # 这一行是10个不同的数字
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
