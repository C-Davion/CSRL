import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


td3res=np.load('results/eval_results.npy',allow_pickle=True).item()
ddpgres=np.load('DDPGresults/results.npy',allow_pickle=True).item()
ppores=np.load('logs/evaluations.npz')

td3_timesteps=td3res['timesteps']
ddpg_timesteps=ddpgres['timesteps']
ppo_timesteps=ppores['timesteps']
print(td3_timesteps.shape,ddpg_timesteps.shape,ppo_timesteps.shape)


ppomeans = np.mean(ppores['results'], axis=1)
ppostds = np.std(ppores['results'], axis=1)

td3mean_smooth=uniform_filter1d(td3res['returns_mean'],size=5)
ddpg_mean_smooth=uniform_filter1d(ddpgres['returns_mean'],size=5)
ppo_mean_smooth=uniform_filter1d(ppomeans,size=5)

plt.figure(figsize=(10, 6))


#td3

plt.plot(td3_timesteps, td3mean_smooth, color='blue', linewidth=2, label=' Td3 Mean Return (Smoothed)')
plt.fill_between(
    td3_timesteps, 
    td3mean_smooth - td3res['returns_std'], 
    td3mean_smooth + td3res['returns_std'], 
    alpha=0.2, 
    color='blue',
)

#DDPG

plt.plot(ddpg_timesteps,ddpg_mean_smooth, color='orange', linewidth=2, label=' DDPG Mean Return (Smoothed)')
plt.fill_between(
    ddpg_timesteps, 
    ddpg_mean_smooth - ddpgres['returns_std'], 
    ddpg_mean_smooth + ddpgres['returns_std'], 
    alpha=0.2, 
    color='orange',
)


#PPO
plt.plot(ppo_timesteps,ppo_mean_smooth,color='green',linewidth=2,label='PPO Mean return smoothed')
plt.fill_between(
    ppo_timesteps, 
    ppo_mean_smooth - ppostds, 
    ppo_mean_smooth + ppostds, 
    alpha=0.2, 
    color='green',
)

plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel('Timesteps', fontsize=12)
plt.ylabel('Return', fontsize=12)
plt.title('Compared performance on HalfCheetah-v4', fontsize=14)
plt.legend(loc='best')
plt.savefig('compare_plot.png', dpi=300, bbox_inches='tight')
