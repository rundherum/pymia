[1mdiff --git a/pymia/deeplearning/logging.py b/pymia/deeplearning/logging.py[m
[1mindex 329ff13..77a08bf 100644[m
[1m--- a/pymia/deeplearning/logging.py[m
[1m+++ b/pymia/deeplearning/logging.py[m
[36m@@ -31,6 +31,10 @@[m [mclass Logger:[m
     def log_visualization(self, epoch: int):[m
         pass[m
 [m
[32m+[m[32m    @abc.abstractmethod[m
[32m+[m[32m    def close(self):[m
[32m+[m[32m        pass[m
[32m+[m
 [m
 class TensorFlowLogger(Logger):[m
 [m
[36m@@ -62,6 +66,9 @@[m [mclass TensorFlowLogger(Logger):[m
         self.writer_valid = tf.summary.FileWriter(log_dir_valid)[m
 [m
     def __del__(self):[m
[32m+[m[32m        self.close()[m
[32m+[m
[32m+[m[32m    def close(self):[m
         self.writer_train.close()[m
         self.writer_valid.close()[m
 [m
[36m@@ -166,6 +173,9 @@[m [mclass TorchLogger(Logger):[m
         self.visualize_kernels = visualize_kernels[m
 [m
     def __del__(self):[m
[32m+[m[32m        self.close()[m
[32m+[m
[32m+[m[32m    def close(self):[m
         self.writer_train.close()[m
         self.writer_valid.close()[m
 [m
