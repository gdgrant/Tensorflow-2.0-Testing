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


##########################
### MODEL SUBFUNCTIONS ###
##########################

def synaptic_plasticity(h, syn_x, syn_u):
	""" Calculate impact of synaptic efficacy on hidden state projection,
		and update the synaptic efficacy states """

	syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
	syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
	syn_x  = tf.clip_by_value(syn_x, 0., 1.)
	syn_u  = tf.clip_by_value(syn_u, 0., 1.)
	h_post = syn_u*syn_x*h

	return h_post, syn_x, syn_u


def recurrent_cell(x, h, syn_x, syn_u, var_dict):
	""" Process one step of the recurrent cell, updating the hidden state
		by way of the inputs, synaptic efficacy, and previous hidden state.
		Return the states and network output. """

	if par['use_stp']:
		h, syn_x, syn_u = synaptic_plasticity(h, syn_x, syn_u)

	h = tf.nn.relu(
		  (1-par['alpha_neuron']) * h \
		+ par['alpha_neuron'] * (x @ var_dict['W_in'] + h @ var_dict['W_rnn'] + var_dict['b_rnn']) \
		+ tf.random.normal(h.shape, 0, par['noise_rnn']))

	y = h @ var_dict['W_out'] + var_dict['b_out']

	return y, h, syn_x, syn_u


###################
### MODEL CLASS ###
###################

class Model:

	def __init__(self):

		# Make and store variables
		var_names = ['W_in', 'W_rnn', 'W_out', 'b_rnn', 'b_out']
		self.var_dict = {n:tf.Variable(par[n+'_init'], name=n) for n in var_names}

		# Initialize optimizer
		self.opt = tf.keras.optimizers.Adam(par['learning_rate'])


	def make_eff_vars(self):
		""" Process variables at the start of each trial to apply
			variable rules and updates that should not be stored in the
			variables themselves """

		eff_vars = {}
		for name, val in self.var_dict.items():
			if name == 'W_rnn':
				eff_vars[name] = par['EI_matrix'] @ tf.nn.relu(val)
			else:
				eff_vars[name] = val

		return eff_vars


	@tf.function
	def run_model(self, input_data, output_data, train_mask):
		""" Initialize a fresh model, then loop through time, updating
			the model along the way.  Return the model's outputs. """

		# Process variable rules
		eff_vars = self.make_eff_vars()

		# Set up recording
		rec = {'h' : [], 'y' : [], 'syn_x' : [], 'syn_u' : []}

		# Initialize model at zero state
		h     = tf.zeros([par['batch_size'],par['n_hidden']])
		syn_x = tf.constant(par['syn_x_init'])
		syn_u = tf.constant(par['syn_u_init'])

		# Unstack the input data across time and iterate through it 
		for x in tf.unstack(input_data):
			
			# Step the recurrent cell state and obtain the requisite outputs
			y, h, syn_x, syn_u = recurrent_cell(x, h, syn_x, syn_u, eff_vars)

			# Record output and cell state
			rec['y'].append(y)
			rec['h'].append(h)
			rec['syn_x'].append(syn_x)
			rec['syn_u'].append(syn_u)

		# Stack records into full arrays (instead of lists of arrays)
		for key, val in rec.items():
			rec[key] = tf.stack(val, axis=0)

		return rec


	@tf.function
	def train_model(self, trial_info):
		""" Run the model and apply a training step from the results """

		# Start recording gradients
		with tf.GradientTape() as tape:

			# Run the model and obtain its outputs
			outputs = self.run_model(trial_info['neural_input'], trial_info['desired_output'], trial_info['train_mask'])

			# Calculate the losses of the network's trial
			task_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
				logits=trial_info['train_mask'][:,:,tf.newaxis]*outputs['y'],
				labels=trial_info['train_mask'][:,:,tf.newaxis]*trial_info['desired_output'],
				axis=-1))
			activity_loss = par['activity_cost']*tf.reduce_mean(tf.square(outputs['h']))

			# Aggregate total loss and collect variables
			total_loss = task_loss + activity_loss
			var_list = [v for k, v in self.var_dict.items()]

		# Calculate gradients
		gradients = tape.gradient(task_loss, var_list)
		
		# Apply gradients (through the optimizer)
		self.opt.apply_gradients(zip(gradients, var_list))

		return task_loss, activity_loss, outputs


#####################################
### MODEL SETUP AND TRAINING LOOP ###
#####################################

def main(gpu_id=None):

	# Isolate requested GPU
	if gpu_id is not None: os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

	# Initialize the model and stimulus
	model = Model()
	stim  = Stimulus()

	for i in range(par['iterations']):

		# Generate a batch of trials
		trial_info = stim.make_batch()

		# Run the model on the provided batch of trials
		task_loss, act_loss, outputs = model.train_model(trial_info)

		# Calculate the network's accuracy
		acc_mask = trial_info['train_mask'] * (np.argmax(trial_info['desired_output'], axis=-1)!=0)
		accuracy = np.sum(acc_mask * (np.argmax(outputs['y'], axis=-1) == np.argmax(trial_info['desired_output'], axis=-1)))/np.sum(acc_mask)

		# Intermittently report feedback on the network
		if i%10 == 0:

			# Plot the network's behavior
			behavior(trial_info, outputs, i)

			# Output the network's performance on the task
			print('{:>5} | Task Loss: {:6.3f} | Task Acc: {:5.3f} | Act. Loss: {:6.3f} |'.format(\
				i, task_loss.numpy(), accuracy, act_loss.numpy()))


# Run the model immediately if this file is run from command line
if __name__ == '__main__':

	# Quit cleanly if using Ctrl-C to end model early
	try:
		# If a gpu ID is specified, send its value to the main
		# Otherwise, simply enter main
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')