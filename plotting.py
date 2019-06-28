from parameters import par
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def behavior(trial_info, outputs, i):

	match    = np.argmax(trial_info['desired_output'][-1,:,:], axis=-1) == 1
	nonmatch = np.argmax(trial_info['desired_output'][-1,:,:], axis=-1) == 2
	y = tf.nn.softmax(outputs['y'], axis=-1).numpy()

	match_response = y[:,match,:]
	nonmatch_response = y[:,nonmatch,:]

	y_match    = np.mean(match_response, axis=1)
	y_nonmatch = np.mean(nonmatch_response, axis=1)

	y_match_std    = np.std(match_response, axis=1)
	y_nonmatch_std = np.std(nonmatch_response, axis=1)

	c_res = [[60/255, 21/255, 59/255, 1.0], [164/255, 14/255, 76/255, 1.0], [77/255, 126/255, 168/255, 1.0]]
	c_err = [[60/255, 21/255, 59/255, 0.5], [164/255, 14/255, 76/255, 0.5], [77/255, 126/255, 168/255, 0.5]]

	time = np.arange(par['num_time_steps'])

	fig, ax = plt.subplots(2, 1, figsize=[12,9], sharex=True)
	for p, (r,e) in enumerate(zip([y_match, y_nonmatch], [y_match_std, y_nonmatch_std])):

		err_low = r-e
		err_high = r+e

		ax[p].fill_between(time, err_low[:,0], err_high[:,0], color=c_err[0])
		ax[p].fill_between(time, err_low[:,1], err_high[:,1], color=c_err[1])
		ax[p].fill_between(time, err_low[:,2], err_high[:,2], color=c_err[2])

		ax[p].plot(time, r[:,0], c=c_res[0], label='Fixation')
		ax[p].plot(time, r[:,1], c=c_res[1], label='Cat. 1 / Match')
		ax[p].plot(time, r[:,2], c=c_res[2], label='Cat. 2 / Non-Match')

		ax[p].legend(loc="upper left")
		ax[p].set_ylabel('Mean Response')
		ax[p].set_xlim(time.min(), time.max())

	fig.suptitle('Output Neuron Behavior')
	ax[0].set_title('Task: {} | Cat. 1 / Match Trials'.format(par['task'].upper()))
	ax[1].set_title('Task: {} | Cat. 2 / Non-Match Trials'.format(par['task'].upper()))
	ax[1].set_xlabel('Time')

	plt.savefig('./savedir/{}_outputs_iter{:0>6}.png'.format(par['savefn'], i), bbox_inches='tight')
	plt.clf()
	plt.close()