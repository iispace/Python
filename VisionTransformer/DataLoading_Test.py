"""
Image File Structure

D:\Image\OrangeQuality\train\
|------------|-----------|--------|              
1.0\        2.0\       3.0\      4.0\
 |           |           |        |
 5.png      18.png     41.png    3.png
28.png      23.png     29.png    16.png
 
D:\Image\OrangeQuality\test\
|------------|-----------|--------|              
1.0\        2.0\       3.0\      4.0\
 |           |           |        |
15.png      8.png     2.png    11.png
8.png      31.png     19.png   26.png.
"""


IMAGE_ROOT = rf"D:\Image\OrangeQuality"
train_root = os.path.join(IMAGE_ROOT, "train")
test_root  = os.path.join(IMAGE_ROOT, "test")
print(f"train_root: {train_root}")
print(f"test_root : {test_root}")


# Load DataSet
OrangeQ_trainset = OrangeQualityDataset(train_root, img_size=12)
OrangeQ_testset = OrangeQualityDataset(test_root, img_size=12)

dataset_type = 'train'
#dataset_type = 'test'

if dataset_type == 'train':
    OrangeQ_dataset = OrangeQ_trainset
elif dataset_type == 'test':
    OrangeQ_dataset = OrangeQ_testset

 
# Prepare DataLoader
batch_size = 8
Org_dataloader = DataLoader(OrangeQ_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# Visualization Test by Batch iteration using the prepared DataLoader
for batch_seq, sample_batches in enumerate(Org_dataloader):
    # sample_batches: A List of Tuple(image, target, label, sample_id)
    if batch_seq == 0:
        for i in range(batch_size):
            print(f"seq: {i}")
            _img_RGB  = sample_batches[0][i].numpy() # torch.Size([batch_size, H, W, C] => numpy()
            targets   = sample_batches[1][i]         # 1D Tensor, torch.Size([])
            label     = sample_batches[2][i]         # str
            sample_id = sample_batches[3][i]         # 1D Tensor, torch.Size([])

            print(f"_img_RGB.shape   = {_img_RGB.shape}")    # torch.Size([batch_size, H, W, C])
            print(f"targets = {targets}") # torch.Size([10])
            print(f"label  = {label}")   # tuple , len = 10
            print(f"sample_id = {sample_id}") # torch.Size([10])

            sample_dic = {'image': _img_RGB, 'label': label, 'sample_id': sample_id}
            plt.figure(figsize=(3,3))
            plt.title(f"sample_seq: {i} ({batch_seq+1}th batch)", fontsize=10)
            plt.text(2.0, 0.2, f"sample_id: {sample_id}", color="yellow", fontsize=9)
            plt.axis('off')
            show_image_label(**sample_dic)
            plt.show()

            print()         
        break
           
