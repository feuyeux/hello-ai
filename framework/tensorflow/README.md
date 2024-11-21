# tensorflow

## Laguanges

<https://www.tensorflow.org/api_docs>

```xml
<!-- https://mvnrepository.com/artifact/org.tensorflow/tensorflow -->
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>tensorflow</artifactId>
    <version>1.15.0</version>
</dependency>
```

### PIP 国内源

- 清华：<https://pypi.tuna.tsinghua.edu.cn/simple>
- 阿里云：<http://mirrors.aliyun.com/pypi/simple/>
- 中国科技大学 <https://pypi.mirrors.ustc.edu.cn/simple/>
- 华中理工大学：<http://pypi.hustunique.com/>
- 山东理工大学：<http://pypi.sdutlinux.org/>
- 豆瓣：<http://pypi.douban.com/simple/>

示例 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade tensorflow-gpu`

```sh
import tensorflow as tf
from tensorflow.keras import layers

print(tf.version)
print(tf.keras.__version__)


<module 'tensorflow._api.v2.version' from '/home/han/.local/lib/python3.6/site-packages/tensorflow/_api/v2/version/__init__.py'>

2.2.4-tf
```

<https://tensorflow.google.cn/beta/tutorials/quickstart/beginner>

```sh
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
```

<https://tensorflow.google.cn/beta/tutorials/text/text_generation>
