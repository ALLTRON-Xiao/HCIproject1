main.py is for task2
please run task1.py or task3.py separately if want to run task1 or task3
build on python3.11, tensorflow2.20.0, opencv4.12

cnnsave1.txt contain info/result of saved model cnn_drop_adam1.keras 
cnnsave2.txt contain info/result of saved model cnn_drop_adam2.keras 
the only difference of this two model is cnn2 trained 10Epoch while cnn trained 5.

Video resource for task3:
    big-shake.mp4
    big-write.mp4
    small-f.mp4
    small-n.mp4
  big means bigger digits which more difference with MNIST than small
  shake is written done digits with some shake to interrupt
  write is video of writing digits
  small is more similar to MNIST dataset
      'f' means far,'n' means near
  

Result floder:
    task1result.txt contain results of task1;
    res.txt, mob.txt, vgg.txt is all frozen result of task2, 'res' represent Resnet-50, 'mob' represent MobileNetV3-Large, 'vgg' represent VGG16;
    frozenxx(_ll)_yy.txt is part frozen result of task2, 'xx' means which part unfrozen, 'll' means use lower learning rate, 'yy' means model name;