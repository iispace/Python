from tqdm import tqdm

plc_lst = list(train.keys())

LENGTH = len(plc_lst)

with tqdm(total=LENGTH) as progressbar:
    ## Here your code....
    progressbar.update(1)
