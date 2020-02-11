


class Data_Visualizer:
    
    def __init__(self, dataframe, save_path):
        
        self.dataframe = dataframe
        self.results = {}
        self.save_path = save_path
        
    def plot_histogram(self, df, step):
        comp_path = self.save_path + "\\" + "hist.png"
       
        try:
            df = df.iloc[:,0:step]
            df.plot.hist(alpha=0.5)
            plt.savefig(comp_path)
            return True

        except Exception as Ex:
            return False
                
                
    def bar_plot(self, df):
        comp_path = self.save_path + "\\" + "bar_plot.png"
        color = dict(boxes='DarkGreen', whiskers='DarkOrange',
                     medians='DarkBlue', caps='Gray')
        try:
            df.plot.box(color=color, sym='r+')
            plt.savefig(comp_path)
            return True

        except Exception as Ex:
            return False
    
    
    def plot_cum(self, df_cum):
        
        for k, v in list(locals().iteritems()):
            if v is df_cum:
                function = k
        
        for i, var_name in df_cum.columns.values:
           
            df_cum.iloc[:,i].plot()
            comp_path = self.save_path + "\\" + function + "_" + i 
            print("Composed path: " + str(comp_path))
            plt.savefig(comp_path)            
            self.results[function][var_name] = df_cum.iloc[:,i]
            
            
        
    def visualize_distribution(self, index, values):
    
        sns.set(style="darkgrid")
        sns.barplot(index, values, alpha=0.9)
        plt.title('Frequency Distribution of Different Campaigns')
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Campaigns', fontsize=12)
        plt.show()

    def plot_progression_kernels(self, X):
        
        np.random.seed(1)
        
        X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
        bins = np.linspace(-5, 10, 10)
        
        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        
        # histogram 1
        ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
        ax[0, 0].text(-3.5, 0.31, "Histogram")
        
        # histogram 2
        ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
        ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")
        
        # tophat KDE
        kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
        log_dens = kde.score_samples(X_plot)
        ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")
        
        # Gaussian KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
        log_dens = kde.score_samples(X_plot)
        ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")
        
        for axi in ax.ravel():
            axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
            axi.set_xlim(-4, 9)
            axi.set_ylim(-0.02, 0.34)
        
        for axi in ax[:, 0]:
            axi.set_ylabel('Normalized Density')
        
        for axi in ax[1, :]:
            axi.set_xlabel('x')
        
    def plot_cumsum(self, var, *args):

        cumsum = self.dataframe.cumsum()
        print("Cumsum computed")
        if len(args) is not 0:
            cumsum.plot(args[0], args[1])
        
        else:
            cumsum.plot(var)
        
    
    def plot_distributions(self, X, *args):
                
        print("length of optional_variables : " + str(len(args)))
        print("Number of variables found in X: " + str(X.shape[1]))
        
        if args is None or len(args) <= 1:
            variables = X.shape[1]        
            try:
                for var in range(0,variables):
                    
                    sns.distplot(X[var])
                    time.sleep(2)
                return True
            
            except Exception as Ex:
                return None
        else:
            try:
                for var in args:
                    sns.distplot(X[var])
                    time.sleep(2)
                return True
            
            except Exception as Ex:
                return None    
    
        