
import torch
import torch.nn as nn
import pdb
import utils 
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from collections import OrderedDict

class MLP(nn.Module):
  def __init__(self, input_feats, out_feats):
    super().__init__()
    self.fc1 = nn.Linear(input_feats, 128)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.fc2 = nn.Linear(128, 64)
    self.bn2 = nn.BatchNorm1d(num_features=64)
    self.fc3 = nn.Linear(64, 32)
    self.bn3 = nn.BatchNorm1d(num_features=32)
    self.fc4 = nn.Linear(32, out_feats)
    self.bn4 = nn.BatchNorm1d(num_features=out_feats)
    self.leakyrelu = nn.LeakyReLU(0.1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()


  def forward(self, x):
    x = torch.flatten(x.float(), start_dim=1)
    x = self.bn1(self.relu(self.fc1(x.float())))
    x = self.bn2(self.relu(self.fc2(x)))
    x = self.bn3(self.relu(self.fc3(x)))
    x = self.sigmoid(self.fc4(x))

    return x


class GRRN(nn.Module):
    def __init__(self,state_dim, time_step = 4):
        super().__init__()
        self.time_step = time_step 
        self.state_dim = state_dim
        self.edge_types = 1
        self.edge_fcs = nn.ModuleList()

        for i in range(1):
            # incoming and outgoing edge embedding
            edge_fc = nn.Linear(self.state_dim, self.state_dim)
            self.edge_fcs.append(edge_fc)
            
        self.reset_gate = nn.Sequential(
            nn.Linear(self.state_dim*2, self.state_dim),
            nn.Sigmoid())
        self.update_gate = nn.Sequential(
            nn.Linear(self.state_dim*2, self.state_dim),
            nn.Sigmoid() )
        self.tansform = nn.Sequential(
            nn.Linear(self.state_dim*2, self.state_dim),
            nn.Tanh() )
        
        self.edge_attens = nn.ModuleList()

        for i in range(1):
            edge_attention =  nn.Sequential(
                nn.Linear(self.state_dim * 2, self.state_dim),
                nn.Linear(self.state_dim, 1),
                nn.Sigmoid(),
                )
            self.edge_attens.append(edge_attention)

        self._initialization()


    # inputs with feature dim [batch, node_num, hidden_state_dim]
    # A with feature dim [batch, node_num, node_num]
    # reture output with feature dim [batch, node_num, output_dim]
    def forward(self,inputs):
        
        node_num = inputs.size(1)

        prop_state = inputs 

        all_scores = []

        for t in range(1 + self.time_step):
            
            message_states = []
            
            for i in range(1):
                message_states.append(self.edge_fcs[i](prop_state))
             #(B X P X F)
            message_states_torch = torch.cat(message_states,dim=1).contiguous()
            message_states_torch = message_states_torch.view(-1,node_num,self.state_dim)

            relation_scores = []

            for i in range(1):
                relation_feature = message_states[i]
                feature_row_large = relation_feature.contiguous().view(-1,node_num,1,self.state_dim).repeat(1,1,node_num,1)
                feature_col_large = relation_feature.contiguous().view(-1,1,node_num,self.state_dim).repeat(1,node_num,1,1)
                feature_large = torch.cat((feature_row_large,feature_col_large),3)
                relation_score = self.edge_attens[i](feature_large)
                relation_scores.append(relation_score)
            
            graph_scores = torch.cat(relation_scores,dim=3).contiguous()
            all_scores.append(graph_scores)
            graph_scores = graph_scores.view(-1,node_num,node_num * self.edge_types)
            merged_message = torch.bmm(graph_scores, message_states_torch)


            a = torch.cat((merged_message,prop_state),2)

            r = self.reset_gate(a)
            z = self.update_gate(a)
            joined_input = torch.cat((merged_message, r * prop_state), 2)
            h_hat = self.tansform(joined_input)
            prop_state = (1 - z) * prop_state + z * h_hat


        return all_scores, merged_message

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)



class Graph_NoTemporal_Transformer(nn.Module):
    def __init__(self, input_feats, out_feats, label_levels, time_step=3,):
        super().__init__()
        self.group_num = out_feats
        self.hidden_dim = input_feats
        self.time_step = time_step
        self.graph_feat_size = 128

        encoder_layer = nn.TransformerEncoderLayer(d_model=184, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.encoder = nn.Sequential(nn.Linear(self.hidden_dim, 128), nn.Linear(128, 128))
        self.graph=GRRN(self.graph_feat_size, time_step).cuda()
        
        self.fc1 = nn.Linear(self.group_num**2, 6)
        self.bn1 = nn.BatchNorm1d(num_features=6)
        self.fc2 = nn.Linear(6, self.group_num)
        self.relu = nn.ReLU()
    

    def forward(self, feats):

        #(batch_size, sequence length, dim_model)
        #[batcn, node_num, 2048]

        feats = torch.flatten(feats, start_dim = 1, end_dim = 2)
        feats = self.transformer_encoder(feats)
        all_scores = self.graph(feats)
        out = all_scores[-1]
        out = self.bn1(self.relu(self.fc1(out)))
        out = self.fc2(out)

        return out



# class TransformerEncoder(nn.Module):
#     def __init__(self, input_feats, out_feats, label_levels, context_frames, nhead):
#         super().__init__()
#         self.group_num = out_feats
#         self.hidden_dim = input_feats
#         self.context_frames = context_frames
#         self.graph_feat_size = 128

#         encoder_layer = nn.TransformerEncoderLayer(d_model=input_feats, nhead=nhead, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

#         self.fc1 = nn.Linear(input_feats*context_frames*self.group_num, self.group_num)
#         self.bn1 = nn.BatchNorm1d(num_features=128)
#         self.fc2 = nn.Linear(128, self.group_num)
#         self.relu = nn.ReLU()

#         # self.Tanh = nn.Tanh()
    

#     def forward(self, feats):

#         #(batch_size, sequence length, dim_model)
#         #[batcn, node_num, 2048]

#         feats = torch.flatten(feats, start_dim = 1, end_dim = 2)
#         feats = self.transformer_encoder(feats)
#         out = self.fc1(feats)
#         # feats = self.bn1(self.relu(self.fc1(feats)))
#         # out = self.Tanh(self.fc2(feats))

        

#         return out



class SkeletalEncoder(nn.Module):
  '''
  input_shape:  (N, time, facial features) #changed to 96?
  output_shape: (N, 256, time)
  '''
  def __init__(self, input_channels=96, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))



    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x

class UNet1D(nn.Module):
  '''
  UNet model for 1D inputs
  (cite: ``https://arxiv.org/pdf/1505.04597.pdf``)
  Arguments
    input_channels (int): input channel size
    output_channels (int): output channel size (or the number of output features to be predicted)
    max_depth (int, optional): depth of the UNet (default: ``5``).
    kernel_size (int, optional): size of the kernel for each convolution (default: ``None``)
    stride (int, optional): stride of the convolution layers (default: ``None``)
  Shape
    Input: :math:`(N, C_{in}, L_{in})`
    Output: :math:`(N, C_{out}, L_{out})` where
      .. math::
        assert L_{in} >= 2^{max_depth - 1}
        L_{out} = L_{in}
        C_{out} = output_channels
  Inputs
    x (torch.Tensor): speech signal in form of a 3D Tensor
  Outputs
    x (torch.Tensor): input transformed to a lower frequency
      latent vector
  '''
  def __init__(self, input_channels, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.pre_downsampling_conv = nn.ModuleList([])
    self.conv1 = nn.ModuleList([])
    self.conv2 = nn.ModuleList([])
    self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
    self.max_depth = max_depth
    self.groups = groups

    ## pre-downsampling
    self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.pre_downsampling_conv.append(ConvNormRelu(output_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    for i in range(self.max_depth):
      self.conv1.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=True,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    for i in range(self.max_depth):
      self.conv2.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=False,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, return_bottleneck=False):
    input_size = x.shape[-1]
    assert input_size/(2**(self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(input_size, 2**(self.max_depth - 1))
    #assert np.log2(input_size) == int(np.log2(input_size)), 'Input size is {}. It must be a power of 2.'.format(input_size)
    assert num_powers_of_two(input_size) >= self.max_depth, 'Input size is {}. It must be a multiple of 2^(max_depth) = 2^{} = {}'.format(input_size, self.max_depth, 2**self.max_depth)

    x = nn.Sequential(*self.pre_downsampling_conv)(x)

    residuals = []
    residuals.append(x)
    for i, conv1 in enumerate(self.conv1):
      x = conv1(x)
      if i < self.max_depth - 1:
        residuals.append(x)

    bn = x
    for i, conv2 in enumerate(self.conv2):
      x = self.upconv(x) + residuals[self.max_depth - i - 1]
      x = conv2(x)

    if return_bottleneck:
      return x, bn
    else:
      return x


def num_powers_of_two(x):
  num_powers = 0
  while x>1:
    if x % 2 == 0:
      x /= 2
      num_powers += 1
    else:
      break
  return num_powers


class ConvNormRelu(nn.Module):
  def __init__(self, in_channels, out_channels,
               type='1d', leaky=False,
               downsample=False, kernel_size=None, stride=None,
               padding=None, p=0, groups=1):
    super(ConvNormRelu, self).__init__()
    if kernel_size is None and stride is None:
      if not downsample:
        kernel_size = 3
        stride = 1
      else:
        kernel_size = 4
        stride = 2

    if padding is None:
      if isinstance(kernel_size, int) and isinstance(stride, tuple):
        padding = tuple(int((kernel_size - st)/2) for st in stride)
      elif isinstance(kernel_size, tuple) and isinstance(stride, int):
        padding = tuple(int((ks - stride)/2) for ks in kernel_size)
      elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
        assert len(kernel_size) == len(stride), 'dims in kernel_size are {} and stride are {}. They must be the same'.format(len(kernel_size), len(stride))
        padding = tuple(int((ks - st)/2) for ks, st in zip(kernel_size, kernel_size))
      else:
        padding = int((kernel_size - stride)/2)


    in_channels = in_channels*groups
    out_channels = out_channels*groups
    if type == '1d':
      self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm1d(out_channels)
      self.dropout = nn.Dropout(p=p)
    elif type == '2d':
      self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm2d(out_channels)
      self.dropout = nn.Dropout2d(p=p)
    if leaky:
      self.relu = nn.LeakyReLU(negative_slope=0.2)
    else:
      self.relu = nn.ReLU()

  def forward(self, x, **kwargs):
    return self.relu(self.norm(self.dropout(self.conv(x))))

  

class GroupTransformerEncoder(nn.Module):
    def __init__(self, input_feats, out_feats, context_frames, label_levels, nhead = 5):
        super().__init__()
        self.hidden_dim = input_feats

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_feats, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.levels = label_levels

        self.fc1 = nn.Linear(input_feats*context_frames, self.levels)
        # self.Tanh = nn.Tanh()
    

    def forward(self, feats):

        #(batch_size, sequence length, dim_model)
        #[batcn, node_num, 2048]
        
        feats_shape = feats.shape
        feats = torch.flatten(feats, start_dim = 1, end_dim = 2)
        feats = self.transformer_encoder(feats)
        feats = feats.view(feats.shape[0],feats_shape[1], feats_shape[2], -1)
        feats = feats.flatten(start_dim = 2, end_dim = 3)
        out =self.fc1(feats)


        
        return out


class UNetTransformer(nn.Module):
  def __init__(self, input_feats, out_feats, label_levels, context_frames, path_to_pretrained, hidden_dim = 256, time_step=3):
    super().__init__()

    self.individualRegressor = IndividualEncoder(input_feats, 1, label_levels =label_levels)
    self.path_to_pretrained = path_to_pretrained
    
    new_state_dict = OrderedDict()
    if path_to_pretrained:
      state_dict = torch.load(path_to_pretrained)
      for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
      self.individualRegressor.load_state_dict(new_state_dict)

    self.individualEncoder = torch.nn.Sequential(*(list(self.individualRegressor.children())[:-5]))
    self.groupEncoder = GroupTransformerEncoder(hidden_dim, out_feats, context_frames, label_levels, nhead = 4)

  def forward(self, feats):
    feats_shape = feats.shape

    
    if self.path_to_pretrained:
      with torch.no_grad():
        feats = self.individualEncoder(feats.view(-1, feats_shape[2],feats_shape[3]))
    else:
      feats = self.individualEncoder(feats.view(-1, feats_shape[2],feats_shape[3]))
    
    
    feats = feats.view(feats_shape[0], feats_shape[1], feats.shape[1],feats.shape[2])

    
    feats = feats.permute(0,1,3,2)
    feats = self.groupEncoder(feats)
      

    return feats

class Graph_NoTemporal(nn.Module):
    def __init__(self, input_feats, out_feats, label_levels, time_step=3):
        super().__init__()
        self.group_num = out_feats
        self.hidden_dim = input_feats
        self.time_step = time_step
        self.graph_feat_size = 128

        self.encoder = nn.Sequential(nn.Linear(self.hidden_dim, 128), nn.Linear(128, 128))
        self.graph=GRRN(self.graph_feat_size, time_step)

        
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(num_features=self.group_num)
        self.fc2 = nn.Linear(64, label_levels)
        self.relu = nn.ReLU()
        

    def forward(self, feats):

        #[batcn, node_num, 2048]
        
        feats = torch.flatten(feats, start_dim = 2)
        
        feats = self.encoder(feats)
        all_scores, merged_message = self.graph(feats)
        out = merged_message
        out = self.bn1(self.relu(self.fc1(out)))
        out = self.relu(self.fc2(out))

        return out

class UNetGraph(nn.Module):
  def __init__(self, input_feats, out_feats, label_levels, context_frames, path_to_pretrained, hidden_dim = 256, time_step=3):
    super().__init__()
    self.individualRegressor = IndividualEncoder(input_feats, 1, label_levels =label_levels)
    self.path_to_pretrained = path_to_pretrained
    
    new_state_dict = OrderedDict()
    if path_to_pretrained:
      state_dict = torch.load(path_to_pretrained)
      for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
      self.individualRegressor.load_state_dict(new_state_dict)

    # state_dict = torch.load(path_to_pretrained)
    # self.individualRegressor.load_state_dict(state_dict)
    self.individualEncoder = torch.nn.Sequential(*(list(self.individualRegressor.children())[:-5]))
    self.groupEncoder = Graph_NoTemporal(hidden_dim*context_frames, out_feats, label_levels)

  def forward(self, feats):
    feats_shape = feats.shape

    if self.path_to_pretrained:
      with torch.no_grad():
        
        feats = self.individualEncoder(feats.view(-1, feats_shape[2],feats_shape[3]))
    else:
      feats = self.individualEncoder(feats.view(-1, feats_shape[2],feats_shape[3]))
    feats = feats.view(feats_shape[0], feats_shape[1], feats.shape[1],feats.shape[2])
    feats = feats.permute(0,1,3,2).flatten(start_dim = 2, end_dim = 3)
    
    feats = self.groupEncoder(feats)
      
    return feats


class IndividualEncoder(nn.Module):
    def __init__(self, input_feats, out_feats, label_levels = 9, hidden_dim = 256, time_step=3):
      super().__init__()
      self.group_num = out_feats
      self.input_feats = input_feats
      self.time_step = time_step

      self.spatial_encoder = SkeletalEncoder(input_channels=input_feats)
      self.temporal_encoder = UNet1D(hidden_dim, hidden_dim)

      self.levels = label_levels
      self.fc1 = nn.Linear(hidden_dim*64, 128)
      self.bn1 = nn.BatchNorm1d(num_features=128)
      self.fc2 = nn.Linear(128, self.levels)
      self.relu = nn.ReLU()

      

      self.decoder = nn.Sequential(self.fc1, self.bn1, self.fc2)
        # self.Tanh = nn.Tanh()
     

    def forward(self, x):

      #(batch_size, sequence length, dim_model)
      #[batcn, node_num, 2048]
      x = self.spatial_encoder(x)
      x = self.temporal_encoder(x)
      x = x.flatten(start_dim = 1)
      x = self.decoder(x)

      return x


class FrozenGPT(nn.Module):
    def __init__(self, input_feats, out_feats, label_levels = 9, hidden_dim = 256, time_step=3):
      super().__init__()
      self.group_num = out_feats
      self.input_feats = input_feats
      self.time_step = time_step
      self.levels = label_levels


      #prefix encoder
      self.fc1 = nn.Linear(256, 512)
      self.fc2 = nn.Linear(512, 768)
      self.spatial_encoder = SkeletalEncoder(input_channels=input_feats)
      self.temporal_encoder = UNet1D(hidden_dim, hidden_dim)
      self.encoder = nn.Sequential(self.fc1, nn.Tanh(), self.fc2)


      #output classifier
      self.classifier = nn.Sequential(nn.Linear((256+64)*768, label_levels))

      self.relu = nn.ReLU()
      self.persona_questions = utils.persona_questions

      #split this 
      self.LLM = GPT2Model.from_pretrained('gpt2').to("cuda:1")
      self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
      # self.wte = torch.nn.Sequential(*(list(self.LLM.children())[:1]))
      # self.gptmodel = torch.nn.Sequential(*(list(self.LLM.children())[3:]))

      self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.Tanh = nn.Tanh()
     

    def forward(self, feats, personas):

      batch_size = feats.shape[0]

      persona_str = np.tile(self.persona_questions, (personas.shape[0],1))
      np.putmask(persona_str, (personas>3).cpu(), "")
      persona_str = [" ".join(x) for x in persona_str]

      encoded_input = self.tokenizer(persona_str, padding= 'max_length', max_length = 256, return_tensors='pt')#['input_ids'].cuda()
      inputs_embeds = self.LLM.wte(encoded_input['input_ids'].cuda())
      attention_mask = encoded_input['attention_mask'].cuda()
      # out = self.LLM.forward(inputs_embeds = inputs_embeds, attention_mask = attention_mask)
    
      #viz encoder
      feats = self.spatial_encoder(feats)
      feats = self.temporal_encoder(feats)
      feats = feats.permute(0,2,1)
      feats = self.encoder(feats)
      
      
      
      attention_mask_prefix = torch.ones(feats.shape[0], feats.shape[1]).cuda()
      #concat 

      inputs_embeds = torch.cat((feats, inputs_embeds), dim = 1)
      attention_mask = torch.cat((attention_mask_prefix, attention_mask), dim = 1)

      out = self.LLM.forward(inputs_embeds = inputs_embeds, attention_mask = attention_mask)
      out = out.last_hidden_state.flatten(start_dim = 1)
      out = self.classifier(out)

      return out


class FrozenGPTCNNLSTM(nn.Module):
    def __init__(self, input_feats, out_feats, label_levels = 9, hidden_dim = 256, time_step=3):
      super().__init__()
      self.group_num = out_feats
      self.input_feats = input_feats
      self.time_step = time_step
      self.levels = label_levels


      self.CNN = torch.nn.Conv1d(64, 64, kernel_size =8, stride = 4)
      self.LSTM = nn.LSTM(32,  16, batch_first = True)
      self.dropout = nn.Dropout(0.5)
      self.fc1 = nn.Linear(1024, 768)
      

      #output classifier
      self.classifier = nn.Sequential(nn.Linear((256+1)*768, label_levels))

      self.relu = nn.ReLU()
      self.persona_questions = utils.persona_questions

      #split this 
      self.LLM = GPT2Model.from_pretrained('gpt2').to("cuda:1")
      self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
      # self.wte = torch.nn.Sequential(*(list(self.LLM.children())[:1]))
      # self.gptmodel = torch.nn.Sequential(*(list(self.LLM.children())[3:]))

      self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.Tanh = nn.Tanh()
     

    def forward(self, feats, personas):

      batch_size = feats.shape[0]

      persona_str = np.tile(self.persona_questions, (personas.shape[0],1))
      np.putmask(persona_str, (personas>3).cpu(), "")
      persona_str = [" ".join(x) for x in persona_str]

      encoded_input = self.tokenizer(persona_str, padding= 'max_length', max_length = 256, return_tensors='pt')#['input_ids'].cuda()
      inputs_embeds = self.LLM.wte(encoded_input['input_ids'].cuda())
      attention_mask = encoded_input['attention_mask'].cuda()
      # out = self.LLM.forward(inputs_embeds = inputs_embeds, attention_mask = attention_mask)
    
      #viz encoder

      feats = self.CNN(feats)
      
      feats, hidden = self.LSTM(feats)
      feats = self.dropout(feats)
      feats = self.fc1(feats.flatten(start_dim =1))
      feats = feats.unsqueeze(1)
      
      attention_mask_prefix = torch.ones(feats.shape[0], feats.shape[1]).cuda()
      #concat 
      inputs_embeds = torch.cat((feats, inputs_embeds), dim = 1)
      attention_mask = torch.cat((attention_mask_prefix, attention_mask), dim = 1)

      out = self.LLM.forward(inputs_embeds = inputs_embeds, attention_mask = attention_mask)

      pdb.set_trace()
      out = out.last_hidden_state.flatten(start_dim = 1)
      out = self.classifier(out)

      return out



      
# from transformers import GPT2Tokenizer, GPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)


class CNNLSTM(nn.Module):
    def __init__(self, input_feats, out_feats, label_levels = 9, hidden_dim = 256, time_step=3):
      super().__init__()
      self.group_num = out_feats
      self.input_feats = input_feats
      self.time_step = time_step
      kernel_size = 3
      out_channels = 32
      self.CNN = torch.nn.Conv1d(64, 64, kernel_size =8, stride = 4)
      self.LSTM = nn.LSTM(32,  16, batch_first = True)
      self.dropout = nn.Dropout(0.5)

      self.fc1 = nn.Linear(1024, 32)
      self.fc2 = nn.Linear(32, label_levels)

    def forward(self, x):

      #(batch_size, sequence length, dim_model)
      #[batcn, node_num, 2048]

      
      x = self.CNN(x)
      output, hidden = self.LSTM(x)
      output = self.dropout(output)
      output = self.fc1(output.flatten(start_dim =1))
      output =  self.fc2(output)

      return output
