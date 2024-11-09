from sklearn.datasets import fetch_openml
import time
begin = int(round(time.time()*1000))
now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(begin/1000))
print(now)
mnist = fetch_openml('mnist_784', cache=True, version=1,
                     data_home='/Users/han/scikit_learn_data')
end = int(round(time.time()*1000))
now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end/1000))
print(now)
print((end-begin)/1000)
print(mnist['processing_date'])
