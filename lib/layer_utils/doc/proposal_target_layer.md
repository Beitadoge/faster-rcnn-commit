1.**proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes)** :

[参数]

```
[输入]
	[rpn_rois] : 大小为[anchor_size,5],,anchor_size大约为2k,,每行元素为[0,x1,y1,x2,y2],其中(x1,y1)为左上角的坐标.(x2,y2)为右下角的坐标
	[rpn_scores] : 大小为[anchor_size, ],,保存着每个框所得到的分数
	[gt_boxes] : 大小为[gt_size,5],,gt_size为一幅图中所有框的个数.每行元素为[x1,y1,x2,y2,gt_box_class]其中(x1,y1)为左上角的坐标.(x2,y2)为右下角的坐标,,gt_box_class为该框里面的物体的类别
	[_num_classes] : 需要分类的类别总数
	
```

[算法流程]

​	该算法首先得出每张图片需要提取的ROI数:**rois_per_image=256**,,在计算出每张图片提取的ROI中前景的数量:**fg_rois_per_image=64**,,然后调用_sample_rois函数得到 **labels**  **rois** **roi_scores** **bbox_targets**  **bbox_inside_weights**,最后在reshape一下



2.**_sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes)**:

```
[输入]
	[all_rois] : 大小为[anchor_size,5],,anchor_size大约为2k,,每行元素为[0,x1,y1,x2,y2],其中(x1,y1)为左上角的坐标.(x2,y2)为右下角的坐标
	[all_scores] : 大小为[anchor_size, ],,保存着每个框所得到的分数
	[gt_boxes] : 大小为[gt_size,5],,gt_size为一幅图中所有框的个数.每行元素为[x1,y1,x2,y2,gt_box_class]其中(x1,y1)为左上角的坐标.(x2,y2)为右下角的坐标,,gt_box_class为该框里面的物体的类别
	[fg_rois_per_image] : 每张图片中前景的个数
	[rois_per_image] : 每张图片中索要提取的proposal数
	[num_classes] : 需要分类的类别总数
	
[输出]
	[labels] : 大小为[256, ],保存着rois中所有positive的proposal的类别,negative的proposal的类别为0
	[rois] : 大小为[256,5],每行元素为[0,x1,y1,x2,y2]
	[roi_scores] : 大小为[256, ]保存着rois中每一个proposal的得分
	[bbox_targets] :[256 , num_class×4] 每行元素为［0,0,0,0,0,dx,dy,dw,dh,0,0,0,...］
	[bbox_inside_weights] : [256 , num_class×4] 每行元素为［0,0,0,0,0,1,1,1,1,0,0,0,...］
```

[算法流程]

​	该算法首先计算修整后的每个anchor与gt_boxe的IOU值,并存储在**overlap**数组中,**overlap**的大小为**[anchor_size,gt_size]**,其中overlap[i,j]表示第i个anchor与第j个gt_box的IOU值.

​	对于每一个anchor,把与它IOU值最大的gt_box的index保存在**gt_assignment**中,并把与它IOU最大的取值放在**max_overlaps**中,并把与它IOU值最大的gt_box的类别作为该anchor的类别,保存在**labels**中.这个类别不是指前景,后景,而是值具体的实际类别.

​	在max_overlaps中筛选出IOU值大于0.5的索引,并保存在**fg_inds**中,这里的索引号是指代anchors

​	在max_overlaps中筛选出IOU值大于或等于0,小于或等于0.5的索引,并保存在**bg_inds**中

​	之后在fg_inds中筛选出fg_rois_per_image个元素,并保存在fg_inds中,在bg_inds中筛选出256-fg_rois_per_image个元素,并保存在bg_inds中.

​	之后把**fg_inds**和**bg_inds**合并成**keep_inds**,,**keep_inds**是一个256维的向量,每个元素代表着**all_rois**的索引号.

​	然后更新**labels**,把筛选出来的这256个anchors的实际类别保存在**labels**中,并把背景anchors的类别设置为0

​	然后遍历**keep_inds**的元素,假设元素为k,把k对应的具体anchor--**all_rois[k]**的值[0,x1,y1,x2,y2]保存在**rois**中,则**rois**大小为[256,5],并把对应的分数保存在**roi_scores**中,其大小为[256, ]

​	然后调用**_compute_targets函数**,,	返回的是**bbox_target_data**,,大小为[256,5],,每一行的元素意义为[anchors_class,dx,dy,dw,dh],,其中dx,dy,dw,dh为筛选出来256个anchors距离gt_box之间的距离

​	然后调用**_get_bbox_regression_labels函数**,,返回的是 **bbox_targets**与**bbox_inside_weights**.其中bbox_targets的大小为[256 , num_class×4]的数组,**bbox_inside_weights**的大小为[256 , num_class×4]



3._compute_targets(ex_rois, gt_rois, labels):

```
[输入]
	[exrois] : [256,4]
	[gt_rois] : [256,4]
	[labels] : [256, ]

[输出]
	[bbox_target_data] : [256,5],,每行所代表的意义为[anchor_class,dx,dy,dw,dh]
```





4._get_bbox_regression_labels(bbox_target_data, num_classes):

```
[输入]
	bbox_target_data : 大小为[256,5],,每行所代表的意义为[anchor_class,dx,dy,dw,dh]
	num_classes : 总的类别数
	
[输出]
	[bbox_targets] :[256 , num_class×4] 每行元素为［0,0,0,0,0,dx,dy,dw,dh,0,0,0,...］
	[bbox_inside_weights] : [256 , num_class×4] 每行元素为［0,0,0,0,0,1,1,1,1,0,0,0,...］
```

[算法流程] 

​	首先初始化 **bbox_targets**为[256 , num_class×4]大小的零矩阵.**bbox_inside_weights**同样为[256 , num_class×4]大小的零矩阵.

​	在筛选出bbox_target_data[:,0]中大于0的索引号,并保存在inds中.

​	接着遍历inds中所有的索引,假如索引号为k,首先得到第**rois([256,5])**中的第k个anchor的类别为clss,,在更新 **bbox_targets**的值,具体的,**bbox_targets[ k , (4×cls : 4×cls+4)]**赋值为第k个anchor所对应的[dx,dy,dw,dh].然后更新**bbox_inside_weights[k , (4×cls : 4×cls+4)]**赋值为[1,1,1,1].

​	