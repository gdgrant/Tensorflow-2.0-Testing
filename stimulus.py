import numpy as np
from parameters import par


class Stimulus:

	def __init__(self):

		self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_tuning_functions()


	def make_batch(self):

		if par['task'] == 'dmc':
			trial_info = self.dmc()
		else:
			raise Exception('Task "{}" not implemented.'.format(par['task']))

		return trial_info


	def dmc(self):

		trial_info = {
			'neural_input'      : np.random.normal(0., par['noise_in'], size=[par['num_time_steps'], par['batch_size'], par['n_input']]).astype(np.float32),
			'desired_output'    : np.zeros([par['num_time_steps'], par['batch_size'], par['n_output']], dtype=np.float32),
			'train_mask'        : np.ones([par['num_time_steps'], par['batch_size']], dtype=np.float32),
		}

		sample_direction = np.random.choice(par['num_motion_dirs'], size=par['batch_size'])
		test_direction   = np.random.choice(par['num_motion_dirs'], size=par['batch_size'])

		sample_category  = sample_direction//int(par['num_motion_dirs']/2)
		test_category    = test_direction//int(par['num_motion_dirs']/2)

		match = sample_category == test_category
		output_neuron = np.where(match, 1, 2)

		end_dead_time       = par['dead_time']//par['dt']
		end_fix_time        = end_dead_time + par['fix_time']//par['dt']
		end_sample_time     = end_fix_time + par['sample_time']//par['dt']
		end_delay_time		= end_sample_time + par['delay_time']//par['dt']
		end_mask_time		= end_delay_time + par['mask_time']//par['dt']
		end_test_time		= par['num_time_steps']

		trial_info['train_mask'][:end_dead_time,:] = 0.
		trial_info['train_mask'][end_delay_time:end_mask_time,:] = 0.
		trial_info['train_mask'][end_mask_time:,:] = par['response_mult']

		trial_info['neural_input'][:end_delay_time,:,par['num_motion_tuned']:par['num_motion_tuned']+par['num_fix_tuned']] \
			+= self.fix_tuning[np.newaxis,:,0]
		trial_info['desired_output'][:end_delay_time,:,0] = 1.

		for t in range(par['batch_size']):

			trial_info['neural_input'][end_fix_time:end_sample_time,t,:par['num_motion_tuned']] \
				+= self.motion_tuning[:,0,sample_direction[t]]
			trial_info['neural_input'][end_delay_time:end_test_time,t,:par['num_motion_tuned']] \
				+= self.motion_tuning[:,0,test_direction[t]]

			trial_info['desired_output'][end_delay_time:,t,output_neuron[t]] = 1.

		return trial_info 

 
	def create_tuning_functions(self):

		motion_tuning = np.zeros((par['num_motion_tuned'], par['num_receptive_fields'], par['num_motion_dirs']))
		fix_tuning    = np.zeros((par['num_fix_tuned'], par['num_receptive_fields']))
		rule_tuning   = np.zeros((par['num_rule_tuned'], par['num_rules']))

		pref_dirs = np.arange(0,360,360/(par['num_motion_tuned']//par['num_receptive_fields']))

		# generate list of possible stimulus directions
		stim_dirs = np.arange(0,360,360/par['num_motion_dirs'])

		for n in range(par['num_motion_tuned']//par['num_receptive_fields']):
			for i in range(len(stim_dirs)):
				for r in range(par['num_receptive_fields']):
					d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
					n_ind = n+r*par['num_motion_tuned']//par['num_receptive_fields']
					motion_tuning[n_ind,r,i] = par['tuning_height']*np.exp(par['kappa']*d)/np.exp(par['kappa'])

		for n in range(par['num_fix_tuned']):
			for i in range(par['num_receptive_fields']):
				if n%par['num_receptive_fields'] == i:
					fix_tuning[n,i] = par['tuning_height']

		neurons_per_rule = par['num_rule_tuned']//par['num_rules']
		for n in range(par['num_rule_tuned']):
			for i in range(par['num_rules']):
				if n in range(i*neurons_per_rule, (i+1)*neurons_per_rule):
					rule_tuning[n,i] = par['tuning_height']

		return motion_tuning.astype(np.float32), fix_tuning.astype(np.float32), rule_tuning.astype(np.float32)


if __name__ == '__main__':

	import matplotlib.pyplot as plt
	s = Stimulus()
	trial_info = s.make_batch()

	fig, ax = plt.subplots(3, sharex=True)
	ax[0].imshow(trial_info['neural_input'][:,0,:].T, aspect='auto')
	ax[1].imshow(trial_info['desired_output'][:,0,:].T, aspect='auto')
	ax[2].imshow(trial_info['train_mask'][:,0,np.newaxis].T, aspect='auto')

	fig.suptitle('Demo Trial')
	ax[0].set_title('Input')
	ax[1].set_title('Output')
	ax[2].set_title('Mask')

	ax[0].set_ylabel('Input Neurons')
	ax[1].set_ylabel('Output Neurons')
	ax[2].set_ylabel('Output Mask')
	ax[2].set_yticks([0])

	ax[2].set_xlabel('Time')

	plt.show()