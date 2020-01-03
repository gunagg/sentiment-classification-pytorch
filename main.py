import re
import unicodedata
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import numpy as np

class wordIndex(object) :
	def __init__(self) :
		self.count = 0
		self.word_to_idx = {}
		self.word_count = {}

	def add_word(self,word) :
		if not word in self.word_to_idx :
			self.word_to_idx[word] = self.count
			self.word_count[word] = 1
			self.count +=1
		else :
			self.word_count[word]+=1

	def add_text(self,text) :
		for word in text.split(' ') :
			self.add_word(word)

def normalizeString(s):
	s = s.lower().strip()
	s = re.sub(r"<br />",r" ",s)
	# s = re.sub(' +',' ',s)
	s = re.sub(r'(\W)(?=\1)', '', s)
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	
	return s


class Model(torch.nn.Module) :
	def __init__(self,embedding_dim,hidden_dim) :
		super(Model,self).__init__()
		self.hidden_dim = hidden_dim
		self.embeddings = nn.Embedding(vocab_size+1, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim,hidden_dim)
		self.linearOut = nn.Linear(hidden_dim,label_count)
	def forward(self,inputs,hidden) :
		x = self.embeddings(inputs).view(len(inputs),1,-1)
		lstm_out,lstm_h = self.lstm(x,hidden)
		x = lstm_out[-1]
		x = self.linearOut(x)
		x = F.log_softmax(x)
		return x,lstm_h
	def init_hidden(self) :
		if use_cuda:
			return (Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
		else:
			return (Variable(torch.zeros(1, 1, self.hidden_dim)),Variable(torch.zeros(1, 1, self.hidden_dim)))

train_mapping = wordIndex()
label_dict = {}
inverse_label_dict = {}

train_labels = []
train_data = []
label_count = 0
test_data = []
max_sequence_len = 500

if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
else:
    device = torch.device("cpu")
    use_cuda = False

with open("train_data.csv", "r") as f :
	data = f.readlines()[1:]
	for line in data :
		line = line.strip()
		label = line.split(",")[0]
		if label in label_dict :
			train_labels.append(label_dict[label])
		else :
			label_dict[label] = label_count
			inverse_label_dict[label_count] = label
			label_count += 1
			train_labels.append(label_dict[label])
		content = ",".join(line.split(",")[1:])
		content = normalizeString(content)
		train_mapping.add_text(content)
		train_data.append(content)

print(train_mapping.count)

with open("test_data.csv", "r") as f :
	data = f.readlines()[1:]
	for line in data :
		line = line.strip()
		content = ",".join(line.split(",")[1:])
		content = normalizeString(content)
		train_mapping.add_text(content)
		test_data.append(content)

print(train_mapping.count)

vocab_size = train_mapping.count
if use_cuda:
	model = Model(50,100).cuda()
else:
	model = Model(50,100)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

with open('dict.pkl','wb') as f :
	pickle.dump(train_mapping.word_to_idx,f)

def train_loop(model, epoch, optimizer) :
	#training loop - do not be too bothered with everything
	model.train()
	print_freq = 10
	avg_loss = 0
	indices = np.arange(len(train_data))
	np.random.shuffle(indices)
	for i in range(len(train_data)) :
		data = train_data[indices[i]]
		input_data = [train_mapping.word_to_idx[word] for word in data.split(' ')]
		if len(input_data) > max_sequence_len :
			input_data = input_data[0:max_sequence_len]

		if use_cuda:
			input_data = Variable(torch.cuda.LongTensor(input_data))
		else:
			input_data = Variable(torch.LongTensor(input_data))

		target = train_labels[indices[i]]
		if use_cuda:
			target_data = Variable(torch.cuda.LongTensor([target]))
		else:
			target_data = Variable(torch.LongTensor([target]))

		hidden = model.init_hidden()
		y_pred,_ = model(input_data,hidden)
		optimizer.zero_grad()
		loss = loss_function(y_pred,target_data)
		loss.backward()
		optimizer.step()
		avg_loss += loss.item()
		if i % print_freq==0:
			print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_data), avg_loss/float(i+1)  ))

def write_csv(model) :
	with open("output.csv", "w") as f :
		f.write("id,sentiment"+"\n")
		model.eval()
		for i in range(len(test_data)) :
			data = test_data[i]
			input_data = [train_mapping.word_to_idx[word] for word in data.split(' ')]
			if len(input_data) > max_sequence_len :
				input_data = input_data[0:max_sequence_len]
			if use_cuda:
				input_data = Variable(torch.cuda.LongTensor(input_data))
			else:
				input_data = Variable(torch.LongTensor(input_data))

			hidden = model.init_hidden()

			y_pred,_ = model(input_data,hidden)
			pred = y_pred.data.max(1)[1].cpu().numpy()[0]
			print(pred)
			f.write(str(i)+","+str(inverse_label_dict[pred])+"\n")

epochs = 5
for epoch in range(epochs) :
	train_loop(model, epoch, optimizer)
torch.save(model.state_dict(), 'model.pth')		

write_csv(model)





