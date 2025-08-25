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

* Note that '#' means it is not the official source, for fair comparision, we obtain the data from [HashNet](https://github.com/thuml/HashNet/tree/master/pytorch) ,which is the same as [CSQ](https://github.com/yuanli2333/Hadamard-Matrix-for-hashing)

---

<h2>train</h2>
<h3>coco/nuswide/voc2012</h3>
<code>python train.py --data_path xxxx --data_name coco --word2vec_file ../data/coco/coco_bert768_word2vec.pkl --epochs 90 --center_update --R 5000 --batch_size 64 --hash_bit 64</code>
<h3>ImageNet/cifar-10</h3>
<code>python train.py --data_path xxxx --data_name imagenet --word2vec_file ../data/imagenet/imagenet_bert768_word2vec.pkl --epochs 90 --fixed_weight --center_update --R 1000 --batch_size 64 --hash_bit 64</code>

&nbsp;
<h5>data_path settings</h5>
> <p style="font-size: small;">ImageNet: image_path: <code>xx/xxx/imagenet/image/xxxx.JPEG</code> so that the data_path : <code>xx/xxx/imagenet</code></p>
> <p style="font-size: small;">COCO: image_path: <code>xx/xxx/coco/data/train2014/xxxx.JPEG</code> so that the data_path : <code>xx/xxx/coco</code></p>

* you can modify the dataloader/data_list.py to adapt to your file path as well.

  

<h5>其他注意事项</h5>

options.py中
parser.add_argument('--model_type', type=str, default='resnet50', help='The type of base model')


修改np.float低版本不支持的报错：measure_utils.py


取消 pretrained= True 警告 networks.py中将其替换为四行代码(共两处)
 


<h5>添加脚本运行权限</h5>
chmod +x run_cifar10.sh


<h5>Result Record</h5>
[20250722-05:21:49]=>[info]: epoch 89 Result：((np.float64(0.7517197921988321), np.float64(0.24132580429190598), np.float64(0.767627714500694)), (np.float64(0.7487518902491297), np.float64(0.12643406779661018), np.float64(0.745961)))
[20250722-05:21:49]=>[info]:    <==update center==>
[20250722-05:21:49]=>[info]: inter class loss 1592.5283203125
[20250722-05:21:49]=>[info]: inter class loss coarse 1534.0
[20250722-05:21:49]=>[info]: same hash bit num 0
[20250722-05:21:49]=>[info]: MAP epoch 89       MAP_best 0.7487518902491297     Is_best True    Best epoch 89
[20250722-05:21:49]=>[info]: save model ../result/cifar10-1/model\checkpoint.pth.tar

[20250722-05:21:50]=>[info]: start drawing ...
[20250722-05:21:50]=>[info]: Hash Pool Radius :1000
MAP1 :0.7488     Recall1 0.1264 Precision1 0.7460        MAP2 0.7676    Recall2 0.2413   Precision2 0.7517

Best map in paper : 0.882  net: MS-COCO 64bits   
