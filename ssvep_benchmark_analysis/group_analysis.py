import function_pools as fp
from scipy.io import loadmat
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

pd.options.mode.chained_assignment = None   # default='warn'

# Parser
parser = argparse.ArgumentParser(description = 'Add some integers.')

parser.add_argument('--length', action = 'store', type = float, default = 5,
                    help = 'Length of data to take into account (0,5].')

parser.add_argument('--subjects', action = 'store', type = int, default = 35,
                    help = 'Number of subjects to use [1,35].')

parser.add_argument('--tag', action = 'store', default = '',
                    help = 'Tag to add to the files.')

args = parser.parse_args()
N_sec = args.length
Ns = args.subjects
sTag = args.tag

print("Summary: Tag: " + sTag + ", Subjects: " + str(Ns) + ", Data length: " + str(N_sec))

# Set Working Directory
abspath = os.path.abspath('__file__')
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

# Load and prepare data
dict_freq_phase = loadmat(os.path.join(dir_data, 'Freq_Phase.mat'))
vec_freq = dict_freq_phase['freqs'][0]
vec_phase = dict_freq_phase['phases'][0]

# list_subject_data = loadData(dirname, '.mat')  # load all subject data

sTag = '_' + str(sTag)
sSec = '_' + str(N_sec)
if sTag != "":
    sNs = '_' + str(Ns)

cca_mat_time = np.load(os.path.join(dir_results, 'cca_mat_time' + sSec + sNs + sTag + '.npy'), allow_pickle=True)
cca_mat_rho = np.load(os.path.join(dir_results, 'cca_mat_rho' + sSec + sNs + sTag + '.npy'))
cca_mat_result = np.load(os.path.join(dir_results, 'cca_mat_result' + sSec + sNs + sTag + '.npy'))
cca_mat_b = np.load(os.path.join(dir_results, 'cca_mat_b' + sSec + sNs + sTag + '.npy'))
cca_mat_b_thresh = np.load(os.path.join(dir_results, 'cca_mat_b_thresh' + sSec + sNs + sTag + '.npy'))
cca_mat_max = np.load(os.path.join(dir_results, 'cca_mat_max' + sSec + sNs + sTag + '.npy'))

ext_cca_mat_time = np.load(os.path.join(dir_results, 'ext_cca_mat_time' + sSec + sNs + sTag + '.npy'), allow_pickle=True)
ext_cca_mat_rho = np.load(os.path.join(dir_results, 'ext_cca_mat_rho' + sSec + sNs + sTag + '.npy'))
ext_cca_mat_result = np.load(os.path.join(dir_results, 'ext_cca_mat_result' + sSec + sNs + sTag + '.npy'))
ext_cca_mat_b = np.load(os.path.join(dir_results, 'ext_cca_mat_b' + sSec + sNs + sTag + '.npy'))
ext_cca_mat_b_thresh = np.load(os.path.join(dir_results, 'ext_cca_mat_b_thresh' + sSec + sNs + sTag + '.npy'))
ext_cca_mat_max = np.load(os.path.join(dir_results, 'ext_cca_mat_max' + sSec + sNs + sTag + '.npy'))

fbcca_mat_time = np.load(os.path.join(dir_results, 'fbcca_mat_time' + sSec + sNs + sTag + '.npy'), allow_pickle=True)
fbcca_mat_rho = np.load(os.path.join(dir_results, 'fbcca_mat_rho' + sSec + sNs + sTag + '.npy'))
fbcca_mat_result = np.load(os.path.join(dir_results, 'fbcca_mat_result' + sSec + sNs + sTag + '.npy'))
fbcca_mat_b = np.load(os.path.join(dir_results, 'fbcca_mat_b' + sSec + sNs + sTag + '.npy'))
fbcca_mat_b_thresh = np.load(os.path.join(dir_results, 'fbcca_mat_b_thresh' + sSec + sNs + sTag + '.npy'))
fbcca_mat_max = np.load(os.path.join(dir_results, 'fbcca_mat_max' + sSec + sNs + sTag + '.npy'))

ext_fbcca_mat_time = np.load(os.path.join(dir_results, 'ext_fbcca_mat_time' + sSec + sNs + sTag + '.npy'), allow_pickle=True)
ext_fbcca_mat_rho = np.load(os.path.join(dir_results, 'ext_fbcca_mat_rho' + sSec + sNs + sTag + '.npy'))
ext_fbcca_mat_result = np.load(os.path.join(dir_results, 'ext_fbcca_mat_result' + sSec + sNs + sTag + '.npy'))
ext_fbcca_mat_b = np.load(os.path.join(dir_results, 'ext_fbcca_mat_b' + sSec + sNs + sTag + '.npy'))
ext_fbcca_mat_b_thresh = np.load(os.path.join(dir_results, 'ext_fbcca_mat_b_thresh' + sSec + sNs + sTag + '.npy'))
ext_fbcca_mat_max = np.load(os.path.join(dir_results, 'ext_fbcca_mat_max' + sSec + sNs + sTag + '.npy'))

# Convert to pandas dataframe
Ns = 35
Nb = 6
Nf = 40
fs = 250     # sampling frequency in hz

df_cca = fp.mk_df(cca_mat_time, cca_mat_rho, cca_mat_result, cca_mat_b_thresh, cca_mat_max, vec_freq, Nf, Ns, Nb)
df_fbcca = fp.mk_df(fbcca_mat_time, fbcca_mat_rho, fbcca_mat_result, fbcca_mat_b_thresh, fbcca_mat_max, vec_freq, Nf, Ns, Nb)
df_ext_cca = fp.mk_df(ext_cca_mat_time, ext_cca_mat_rho, ext_cca_mat_result, ext_cca_mat_b_thresh, ext_cca_mat_max, vec_freq, Nf, Ns, Nb)
df_ext_fbcca = fp.mk_df(ext_fbcca_mat_time, ext_fbcca_mat_rho, ext_fbcca_mat_result, ext_fbcca_mat_b_thresh, ext_fbcca_mat_max, vec_freq, Nf, Ns, Nb)

# convert to subject wise representation
df_subject = pd.DataFrame()

df_subject['Accuracy CCA'] = df_cca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100
df_subject['Accuracy FBCCA'] = df_fbcca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100
df_subject['Accuracy ext CCA'] = df_ext_cca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100
df_subject['Accuracy ext FBCCA'] = df_ext_fbcca.groupby(['Subject']).sum()['Compare'] / (Nb * Nf) * 100

df_subject['Time CCA'] = df_cca.groupby(['Subject']).mean()['Time'] / 1000
df_subject['Time FBCCA'] = df_fbcca.groupby(['Subject']).mean()['Time'] / 1000
df_subject['Time ext CCA'] = df_ext_cca.groupby(['Subject']).mean()['Time'] / 1000
df_subject['Time ext FBCCA'] = df_ext_fbcca.groupby(['Subject']).mean()['Time'] / 1000

df_subject['ITR CCA'] = df_subject['Accuracy CCA'].apply((lambda x: fp.itr(x, N_sec + 0.5)))  # the average time for a selection (N_sec + 0.5); 0.5 - gaze shifting time
df_subject['ITR FBCCA'] = df_subject['Accuracy FBCCA'].apply((lambda x: fp.itr(x, N_sec + 0.5)))
df_subject['ITR ext CCA'] = df_subject['Accuracy ext CCA'].apply((lambda x: fp.itr(x, N_sec + 0.5)))
df_subject['ITR ext FBCCA'] = df_subject['Accuracy ext FBCCA'].apply((lambda x: fp.itr(x, N_sec + 0.5)))

# Plot
palette = sns.color_palette('Greys')
lLabels = ['CCA', 'FBCCA', 'Extended \n CCA', 'Extended \n FBCCA']

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
sns.barplot(ax = ax1, data = df_subject[['Accuracy CCA', 'Accuracy FBCCA', 'Accuracy ext CCA', 'Accuracy ext FBCCA']],
            ci=95, palette = 'Greys', capsize = .1, orient = 'h')
ax1.set_yticklabels(lLabels)
ax1.set_xlabel('Accuracy in %')
fp.set_style(fig1, ax1)
fp.set_size(fig1, 3, 2.2)
# plt.setp(ax1.get_xticklabels(), rotation = 45, ha = 'right')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
sns.barplot(ax = ax2, data = df_subject[['Time CCA', 'Time FBCCA', 'Time ext CCA', 'Time ext FBCCA']], 
            ci = 95, palette = 'Greys', capsize = .1, orient = 'h')
ax2.set_yticklabels(lLabels)
ax2.set_xlabel('Time elapsed in s')
fp.set_style(fig2, ax2)
fp.set_size(fig2, 3, 2.2)
# plt.setp(ax2.get_xticklabels(), rotation = 45, ha = 'right')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
sns.barplot(ax = ax3, data = df_subject[['ITR CCA', 'ITR FBCCA', 'ITR ext CCA', 'ITR ext FBCCA']], 
            ci = 95, palette = 'Greys', capsize = .1, orient = 'h')
ax3.set_yticklabels(lLabels)
ax3.set_xlabel('ITR')
fp.set_style(fig3, ax3)
fp.set_size(fig3, 3, 2.2)
# plt.setp(ax3.get_xticklabels(), rotation = 45, ha = 'right')

fig1.savefig(os.path.join(dir_figures, 'accuracy' + sSec + sNs + sTag + '.pdf'), dpi=300)
fig1.savefig(os.path.join(dir_figures, 'accuracy' + sSec + sNs + sTag + '.png'), dpi=300)
fig2.savefig(os.path.join(dir_figures, 'time' + sSec + sNs + sTag + '.pdf'), dpi=300)
fig2.savefig(os.path.join(dir_figures, 'time' + sSec + sNs + sTag + '.png'), dpi=300)
fig3.savefig(os.path.join(dir_figures, 'itr' + sSec + sNs + sTag + '.pdf'), dpi=300)
fig3.savefig(os.path.join(dir_figures, 'itr' + sSec + sNs + sTag + '.png'), dpi=300)

print("=====================================")
print("Accuracy CCA Mean: " + str(df_subject['Accuracy CCA'].mean()) + ", Std: " + str(df_subject['Accuracy CCA'].std()))
print("Accuracy FBCCA Mean: " + str(df_subject['Accuracy FBCCA'].mean()) + ", Std: " + str(df_subject['Accuracy FBCCA'].std()))
print("Accuracy Extended CCA Mean: " + str(df_subject['Accuracy ext CCA'].mean()) + ", Std: " + str(df_subject['Accuracy ext CCA'].std()))
print("Accuracy Extended FBCCA Mean: " + str(df_subject['Accuracy ext FBCCA'].mean()) + ", Std: " + str(df_subject['Accuracy ext FBCCA'].std()))
print("=====================================")

print("Time CCA Mean: " + str(df_subject['Time CCA'].mean()) + ", Std: " + str(df_subject['Time CCA'].std()))
print("Time FBCCA Mean: " + str(df_subject['Time FBCCA'].mean()) + ", Std: " + str(df_subject['Time FBCCA'].std()))
print("Time Extended CCA Mean: " + str(df_subject['Time ext CCA'].mean()) + ", Std: " + str(df_subject['Time ext CCA'].std()))
print("Time Extended FBCCA Mean: " + str(df_subject['Time ext FBCCA'].mean()) + ", Std: " + str(df_subject['Time ext FBCCA'].std()))
print("=====================================")

print("ITR CCA Mean: " + str(df_subject['ITR CCA'].mean()) + ", Std: " + str(df_subject['ITR CCA'].std()))
print("ITR FBCCA Mean: " + str(df_subject['ITR FBCCA'].mean()) + ", Std: " + str(df_subject['ITR FBCCA'].std()))
print("ITR Extended CCA Mean: " + str(df_subject['ITR ext CCA'].mean()) + ", Std: " + str(df_subject['ITR ext CCA'].std()))
print("ITR Extended FBCCA Mean: " + str(df_subject['ITR ext FBCCA'].mean()) + ", Std: " + str(df_subject['ITR ext FBCCA'].std()))
print("=====================================")

fig4, ax4 = fp.plot_trial(cca_mat_result)
fig4.savefig(os.path.join(dir_figures, 'cca_freq' + sSec + sNs + sTag + '.pdf'), dpi=300)
fig4.savefig(os.path.join(dir_figures, 'cca_freq' + sSec + sNs + sTag + '.png'), dpi=300)

fig5, ax5 = fp.plot_trial(fbcca_mat_result)
fig5.savefig(os.path.join(dir_figures, 'fbcca_freq' + sSec + sNs + sTag + '.pdf'), dpi=300)
fig5.savefig(os.path.join(dir_figures, 'fbcca_freq' + sSec + sNs + sTag + '.png'), dpi=300)

fig6, ax6 = fp.plot_trial(ext_cca_mat_result)
fig6.savefig(os.path.join(dir_figures, 'ext_cca_freq' + sSec + sNs + sTag + '.pdf'), dpi=300)
fig6.savefig(os.path.join(dir_figures, 'ext_cca_freq' + sSec + sNs + sTag + '.png'), dpi=300)

fig7, ax7 = fp.plot_trial(ext_fbcca_mat_result)
fig7.savefig(os.path.join(dir_figures, 'ext_fbcca_freq' + sSec + sNs + sTag + '.pdf'), dpi=300)
fig7.savefig(os.path.join(dir_figures, 'ext_fbcca_freq' + sSec + sNs + sTag + '.png'), dpi=300)
