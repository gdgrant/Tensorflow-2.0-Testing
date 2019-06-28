import numpy as np

par = { 

	'savedir'				: './savedir/',
	'savefn'				: 'testing',

	'batch_size'			: 512,
	'iterations'			: 1000,
	'learning_rate'			: 2.5e-3,

	'use_stp'				: True,
	'EI_prop'				: 0.8,

	'n_hidden'				: 100,
	'num_motion_tuned'		: 24,
	'num_fix_tuned'			: 4,
	'num_rule_tuned'		: 4,
	'num_receptive_fields'	: 1,
	'num_rules'				: 1,

	'noise_in'				: 0.,
	'noise_rnn'				: 0.,
	'tau_hidden'			: 100,
	'tau_fast'				: 200,
	'tau_slow'				: 1500,

	'task'					: 'dmc',
	'num_motion_dirs'		: 8,
	'kappa'					: 2.,
	'tuning_height'			: 4.,
	'response_mult'			: 1.,
	
	'dt'					: 20,
	'dead_time'				: 40,
	'fix_time'				: 200,
	'sample_time'			: 300,
	'delay_time'			: 300,
	'test_time'				: 400,
	'mask_time'				: 40,


}


def variable_initialization():

	par['W_in_init']  = np.random.gamma(1., 0.1, size=[par['n_input'], par['n_hidden']]).astype(np.float32)
	par['W_rnn_init'] = np.random.gamma(1., 0.1, size=[par['n_hidden'], par['n_hidden']]).astype(np.float32)
	par['W_out_init'] = np.random.gamma(1., 0.1, size=[par['n_hidden'], par['n_output']]).astype(np.float32)

	par['b_rnn_init'] = np.zeros([1,par['n_hidden']], dtype=np.float32)
	par['b_out_init'] = np.zeros([1,par['n_output']], dtype=np.float32)


def update_dependencies():

	if par['task'] == 'dmc':
		par['n_output'] = 3

	par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']

	variable_initialization()

	par['trial_length'] = par['dead_time'] + par['fix_time'] \
		+ par['sample_time'] + par['delay_time'] + par['test_time']
	par['num_time_steps'] = par['trial_length'] // par['dt']

	par['n_exc'] = int(par['n_hidden']*par['EI_prop'])
	par['EI_vector'] = np.ones(par['n_hidden'], dtype=np.float32)
	par['EI_vector'][par['n_exc']:] *= -1
	par['EI_matrix'] = np.diag(par['EI_vector']).astype(np.float32)

	par['alpha_neuron'] = par['dt']/par['tau_hidden']
	par['dt_sec'] = par['dt']/1000

	par['alpha_stf'] = np.ones([1,par['n_hidden']], dtype=np.float32)
	par['alpha_std'] = np.ones([1,par['n_hidden']], dtype=np.float32)
	par['U']         = np.ones([1,par['n_hidden']], dtype=np.float32)

	par['syn_x_init'] = np.zeros([par['batch_size'],par['n_hidden']], dtype=np.float32)
	par['syn_u_init'] = np.zeros([par['batch_size'],par['n_hidden']], dtype=np.float32)


	for i in range(0,par['n_hidden'],2):
		par['alpha_stf'][:,i] = par['dt']/par['tau_slow']
		par['alpha_std'][:,i] = par['dt']/par['tau_fast']
		par['U'][:,i] = 0.15
		par['syn_x_init'][:,i] = 1
		par['syn_u_init'][:,i] = par['U'][:,i]

		par['alpha_stf'][:,i+1] = par['dt']/par['tau_fast']
		par['alpha_std'][:,i+1] = par['dt']/par['tau_slow']
		par['U'][:,i+1] = 0.45
		par['syn_x_init'][:,i+1] = 1
		par['syn_u_init'][:,i+1] = par['U'][:,i+1]


update_dependencies()