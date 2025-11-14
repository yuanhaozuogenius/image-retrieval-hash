<h2>environment</h2>
<ul>
<li>
Linux
</li>
<li>
pytorch >= 1.4 python>=3.6.5 and corresponding torchvision,numpy,tqdm,etc.
</li>
<li>
NVIDIA GPU 3090ti *8 &nbsp &nbsp CUDA V9.2
</li>
</ul>

---

<h2>dataset</h2>
<table>
<tr>
<td >dataset</td><td>class_num</td><td>label type</td><td>source</td>
</tr>
<tr>
<td>ImageNet</td><td>100</td><td>single</td><td><a href="https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?resourcekey=0-yXVCpvfmjTx-OBW6PsSMiA">source</a>#</td>
</tr>
<tr>
<td>COCO</td><td>80</td><td>multi</td><td><a href="https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?resourcekey=0-yXVCpvfmjTx-OBW6PsSMiA">source</a>#</td>
</tr>
<tr>
<td>NUS-WIDE</td><td>21</td><td>multi</td><td><a href="https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?resourcekey=0-yXVCpvfmjTx-OBW6PsSMiA">source</a>#</td>
</tr>
<tr>
<td>VOC2012</td><td>20</td><td>multi</td><td><a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html">source</a></td>
</tr>
<tr>
<td>CIFAR-10</td><td>10</td><td>single</td><td><a href="http://www.cs.toronto.edu/~kriz/cifar.html">source</a></td>
</tr>

</table>

* Note that '#' means it is not the official source, for fair comparision, we obtain the data
  from [HashNet](https://github.com/thuml/HashNet/tree/master/pytorch) ,which is the same
  as [CSQ](https://github.com/yuanli2333/Hadamard-Matrix-for-hashing)

---

<h2>train</h2>
<h3>coco/nuswide/voc2012</h3>
<code>python train.py --data_path xxxx --data_name coco --word2vec_file ../data/coco/coco_bert768_word2vec.pkl --epochs
90 --center_update --R 5000 --batch_size 64 --hash_bit 64</code>
<h3>ImageNet/cifar-10</h3>
<code>python train.py --data_path xxxx --data_name imagenet --word2vec_file
../data/imagenet/imagenet_bert768_word2vec.pkl --epochs 90 --fixed_weight --center_update --R 1000 --batch_size 64
--hash_bit 64</code>

&nbsp;
<h5>data_path settings</h5>
> <p style="font-size: small;">ImageNet: image_path: <code>xx/xxx/imagenet/image/xxxx.JPEG</code> so that the data_path : <code>xx/xxx/imagenet</code></p>
> <p style="font-size: small;">COCO: image_path: <code>xx/xxx/coco/data/train2014/xxxx.JPEG</code> so that the data_path : <code>xx/xxx/coco</code></p>

* you can modify the dataloader/data_list.py to adapt to your file path as well.

<h5>其他注意事项</h5>
1. win中执行时需修改调用logger.py中 Logger.getTimeStr()为：  
```def getTimeStr(time):
  return time.strftime("%Y-%m-%d_%H-%M-%S")
```
2. measure_utils.py中 np.float 全部替换为float  

3. 启动方法：src目录下终端执行 run_coco.sh  

4. 换数据集先调用：word_to_vector.py  
word2vec_file存储每个类别的 词向量（文本嵌入），用于初始化语义中心  
ex: coco_bert768_word2vec.pkl  
```{
"class_names": ["person", "car", "dog", ...],
"word_embeddings": torch.tensor([[...], [...], ...])  # shape [80, 768]
}
```

<h5>添加脚本运行权限</h5>
chmod +x run_cifar10.sh  
若设置为run config中启动 需要配置command option =
`--data_path D:/Datasets/coco2017 --data_name coco --word2vec_file ../data/coco/coco_bert768_word2vec.pkl --epochs 120 --fixed_weight --center_update --R 5000 --batch_size 64 --hash_bit 64 --start_test_epoch 10`  
working directory =`LSCSH_sourcecode-main\src`

Best map in paper : 0.882 net: MS-COCO 64bits   
