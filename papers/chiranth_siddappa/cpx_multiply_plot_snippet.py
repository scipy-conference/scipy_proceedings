fig, axs = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0})

axs[0].plot(ffp, 10*np.log10(Pxp))
axs[0].set_ylim([-80, 25])
axs[0].set_xlim([-0.2, 0.2])

axs[1].plot(ffn, 10*np.log10(Pxn))
axs[1].set_ylim([-80, 25])
axs[1].set_ylabel('Power Spectral Density (dB/Hz)')

axs[2].plot(ff_out, 10*np.log10(Px_out))
axs[2].set_ylim(bottom=0)
axs[2].set_xlabel('Frequency (Hz)')
fig.tight_layout()
fig.savefig('cpx_multiply_all.png')