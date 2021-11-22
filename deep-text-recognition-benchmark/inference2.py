
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import ResizeNormalize
from model import Model
from PIL import Image
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def load_model(model_path='best_accuracy.pth'):
    
    
    opt=argparse.Namespace()


    opt.FeatureExtraction='ResNet'
    opt.PAD=False
    opt.Prediction='Attn'
    opt.SequenceModeling='BiLSTM'
    opt.Transformation='TPS'
    opt.batch_max_length=60
    opt.batch_size=1
    opt.character='0123456789<abcdefghijklmnopqrstuvwxyz'
    opt.hidden_size=256
    opt.imgH=32
    opt.imgW=250
    opt.input_channel=1
    opt.num_fiducial=20
    opt.output_channel=512
    opt.rgb=True
    opt.saved_model='saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth'
    opt.sensitive=False
    opt.workers=4




    """ model configuration """

    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)

    model = torch.nn.DataParallel(model).to(device)

    #load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    model.eval()
    
    
    return model




def img_to_string(img,model):
  """
  im: str (path) or PIL image  

  """
  opt=argparse.Namespace()


  opt.PAD=False
  opt.batch_max_length=60
  opt.batch_size=1
  opt.character='0123456789<abcdefghijklmnopqrstuvwxyz'
  opt.hidden_size=256
  opt.imgH=32
  opt.imgW=250
  opt.input_channel=1
  opt.num_fiducial=20
  opt.output_channel=512
  opt.rgb=True
  opt.sensitive=False
  opt.workers=4





  converter = AttnLabelConverter(opt.character)  

  """ Prepare Image """
  try:
    if isinstance(img, str):
      img=Image.open(img).convert('RGB')
    else:
      img=img.convert('RGB')
  except IOError:
    print('Corrupted Image')     


  transform=ResizeNormalize((opt.imgW, opt.imgH))
  image_tensor=transform(img).unsqueeze(0)


  """ Prediction """

  with torch.no_grad():
    batch_size=opt.batch_size
    image=image_tensor.to(device)

    #max length prediction
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    preds = model(image, text_for_pred, is_train=False)

    # select max probabilty (greedy decoding) then decode index to character
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index, length_for_pred)


    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)



    pred_EOS = preds_str[0].find('[s]')
    pred = preds_str[0][:pred_EOS]  # prune after "end of sentence" token ([s])


    pred_max_prob = preds_max_prob[0][:pred_EOS]
    confidence_score = pred_max_prob.cumprod(dim=0)[-1]


  return pred, confidence_score


