# pytorch_throughput

NOTE:
1. Network is selected as Resnet50. On each GPU, the batshsize is fixed at 32.
2. Instead of appling real images/data, we used synthetic data to measure the throughput under different experimental settings.

## Descriptions:
- one_gpu.py is used for testing the throughput of pytorch on a single GPU(K80, AWS p2.xlarge instance).
- multi_gpu_one_node.py is used for testing the case that there are several GPUs available on one server. 
(correspoding to the result in the first table)
- distributed_training.py is used for the case multi servers(each contains several GPUs) are available. This file is modified from 
https://github.com/pytorch/examples/tree/master/imagenet.

## Results:


<table class="tg">
  <tr>
    <th class="tg-baqh" colspan="5">Settings: p2.8xlarge, 32 images per gpu, Resnet50. Average on 500 steps</th>
  </tr>
  <tr>
    <td class="tg-baqh">Number of gpu</td>
    <td class="tg-baqh">fwd_throughput</td>
    <td class="tg-baqh">bwd_throughput</td>
    <td class="tg-baqh">overall throughput</td>
    <td class="tg-baqh">speedup</td>
  </tr>
  <tr>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">164</td>
    <td class="tg-baqh">55.7</td>
    <td class="tg-baqh">41.49</td>
    <td class="tg-baqh">1.00</td>
  </tr>
  <tr>
    <td class="tg-baqh">2</td>
    <td class="tg-baqh">195.89</td>
    <td class="tg-baqh">129.09</td>
    <td class="tg-baqh">77.65</td>
    <td class="tg-baqh">1.87</td>
  </tr>
  <tr>
    <td class="tg-baqh">4</td>
    <td class="tg-baqh">389.07</td>
    <td class="tg-baqh">256.34</td>
    <td class="tg-baqh">154.2</td>
    <td class="tg-baqh">3.72</td>
  </tr>
  <tr>
    <td class="tg-baqh">8</td>
    <td class="tg-baqh">760.66</td>
    <td class="tg-baqh">506.8</td>
    <td class="tg-baqh">303.5</td>
    <td class="tg-baqh">7.32</td>
  </tr>
</table>


<table class="tg">
  <tr>
    <th class="tg-c3ow" colspan="5">Settings: N * p2.8xlarge, 32 images per gpu, Resnet50. Average on 500 steps</th>
  </tr>
  <tr>
    <td class="tg-c3ow">Number of instance</td>
    <td class="tg-c3ow">fwd_throughput</td>
    <td class="tg-c3ow">bwd_throughput</td>
    <td class="tg-c3ow">overall throughput</td>
    <td class="tg-c3ow">speedup</td>
  </tr>
  <tr>
    <td class="tg-c3ow">2 (16 GPUs)</td>
    <td class="tg-c3ow">1262</td>
    <td class="tg-c3ow">1170</td>
    <td class="tg-c3ow">608</td>
    <td class="tg-c3ow">14.65</td>
  </tr>
  <tr>
    <td class="tg-c3ow">4 (32 GPUs)</td>
    <td class="tg-c3ow">2233</td>
    <td class="tg-c3ow">2271</td>
    <td class="tg-c3ow">1120</td>
    <td class="tg-c3ow">26.99</td>
  </tr>
  <tr>
    <td class="tg-c3ow">8 (64 GPUs)</td>
    <td class="tg-c3ow">4323</td>
    <td class="tg-c3ow">4561</td>
    <td class="tg-c3ow">2209</td>
    <td class="tg-c3ow">53.24</td>
  </tr>
  <tr>
    <td class="tg-c3ow">16 (128 GPUs)</td>
    <td class="tg-c3ow">8394</td>
    <td class="tg-c3ow">8640</td>
    <td class="tg-c3ow">4231</td>
    <td class="tg-c3ow">101.98</td>
  </tr>
</table>
