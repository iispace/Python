### Install pytorch_model_summary with conda:
<p>
Conda Env\>conda install -c conda-forge pytorch-model-summary
  
<p>
Github/pytorch_model_summary: https://github.com/amarczew/pytorch_model_summary/blob/master/pytorch_model_summary/model_summary.py
  
- summary(model, *inputs, batch_size=-1, show_input=False, show_hierarchical=False, print_summary=False, max_depth=1, show_parent_layers=False)<p>
  - model: pytorch model object  
  - input_shape(N x C x H x W): batch_size, channel, height, width
  - batch_size: if provided, it is printed in summary table
  - show_input: show input shape. Otherwise, output shape for each layer. (Default: False)
  - show_hierarchical: in addition of summary table, return hierarchical view of the model (Default: False)
  - print_summary: when true, is not required to use print function outside summary method (Default: False)
  - max_depth: it specifies how many times it can go inside user defined layers to show them (Default: 1)
  - show_parent_layer: it adds a column to show parent layers path until reaching current layer in depth. (Default: False)
  
<p>
from pytorch_model_summary import summary
