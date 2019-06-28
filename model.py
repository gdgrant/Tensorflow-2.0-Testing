from parameters import par
from stimulus import Stimulus
from plotting import behavior

import os, sys, time
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Match GPU IDs to nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def synaptic_plasticity(h, syn_x, syn_u):

	syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
	syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
	syn_x  = tf.clip_by_value(syn_x, 0., 1.)
	syn_u  = tf.clip_by_value(syn_u, 0., 1.)
	h_post = syn_u*syn_x*h

	return h_post, syn_x, syn_u


def recurrent_cell(x, h, syn_x, syn_u, var_dict):

	if par['use_stp']:
		h, syn_x, syn_u = synaptic_plasticity(h, syn_x, syn_u)

	h = tf.nn.relu(
		  (1-par['alpha_neuron']) * h \
		+ par['alpha_neuron'] * (x @ var_dict['W_in'] + h @ var_dict['W_rnn'] + var_dict['b_rnn']) \
		+ tf.random.normal(h.shape, 0, par['noise_rnn']))

	y = h @ var_dict['W_out'] + var_dict['b_out']

	return y, h, syn_x, syn_u


class Model:

	def __init__(self):

		var_names = ['W_in', 'W_rnn', 'W_out', 'b_rnn', 'b_out']
		self.var_dict = {n:tf.Variable(par[n+'_init'], name=n) for n in var_names}


	def make_eff_vars(self):

		eff_vars = {}
		for name, val in self.var_dict.items():
			if name == 'W_rnn':
				eff_vars[name] = par['EI_matrix'] @ tf.nn.relu(val)
			else:
				eff_vars[name] = val

		return eff_vars


	@tf.function
	def run_model(self, input_data, output_data, train_mask):

		# print(self.var_dict)

		eff_vars = self.make_eff_vars()
		rec = {
			'h'		: [],
			'y'		: [],
			'syn_x' : [],
			'syn_u'	: [],
		}

		h     = tf.zeros([par['batch_size'],par['n_hidden']])
		syn_x = tf.constant(par['syn_x_init'])
		syn_u = tf.constant(par['syn_u_init'])

		for x in tf.unstack(input_data):
			
			y, h, syn_x, syn_u = recurrent_cell(x, h, syn_x, syn_u, eff_vars)

			rec['y'].append(y)
			rec['h'].append(h)
			rec['syn_x'].append(syn_x)
			rec['syn_u'].append(syn_u)

		for key, val in rec.items():
			rec[key] = tf.stack(val, axis=0)

		return rec


def main(gpu_id=None):

	# Isolate requested GPU
	if gpu_id is not None:
		os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


	model = Model()
	stim  = Stimulus()
	opt   = tf.keras.optimizers.Adam(par['learning_rate'])

	for i in range(par['iterations']):

		trial_info = stim.make_batch()
		with tf.GradientTape() as tape:
			outputs = model.run_model(trial_info['neural_input'], trial_info['desired_output'], trial_info['train_mask'])

			# total_loss = tf.reduce_mean(tf.square(outputs['y'] - trial_info['desired_output']))
			total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
				logits=trial_info['train_mask'][:,:,np.newaxis]*outputs['y'],
				labels=trial_info['train_mask'][:,:,np.newaxis]*trial_info['desired_output'],
				axis=-1))

			mask = trial_info['train_mask'] * (np.argmax(trial_info['desired_output'], axis=-1)!=0)
			accuracy = np.sum(mask * (np.argmax(outputs['y'].numpy(), axis=-1) == np.argmax(trial_info['desired_output'], axis=-1)))/np.sum(mask)

			var_list = [v for k, v in model.var_dict.items()]
			gradients = tape.gradient(total_loss, var_list)
		
		opt.apply_gradients(zip(gradients, var_list))
		if i%10 == 0:

			behavior(trial_info, outputs, i)
			print('{:>5} | Loss: {:6.3f} | Task Acc: {:5.3f} |'.format(i, total_loss.numpy(), accuracy))

if __name__ == '__main__':
	
	if len(sys.argv) > 1:
		main(sys.argv[1])
	else:
		main()