import torch
import pretrainedmodels
import pretrainedmodels.utils as utils
import pdb

model_name = 'resnet101' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()

load_img = utils.LoadImage()

# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model) 

path_img = 'data/car2.jpg'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor,
            requires_grad=False)

output_logits = model(input) # 1x1000
#print(output_logits)
#pdb.set_trace()
print(((torch.max(output_logits).item()-output_logits)<0.01).nonzero())
