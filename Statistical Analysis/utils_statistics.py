
####################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
TO-DO : Expand the functionalities of these units, to include values obtained using other python modules/packages : 
    

Classes:
    
    UnivariateAnalysis : Analysis over a dataframe for all columns where the following different statistical measures are computed: 
                         mean, median, std, variance, std_error, skewness, cummax, cummin, cumprod, pct_change, rank, 
                         quantile, abs_values, compound, kurtosis. 
                         
    
    BiVariate Analysis : Analysis over a dataframe where different techniques are computed. These are used to infer possible relationships between d
                        variables. These include measures like : correlation, covariance, ANOVA, regression analyis. 
                        
                        
                        
    MultiVariate Analysis : Analysis over a dataframe. We aim to find dependencies/relationships of variables. 




                        
                        
                        
                        
Perform pandas Summary Statistics over data. Methods to compute statistics over 
a dataframe or a serie :
         
    Inputs:
        
        - dataframe : pd.Series or pd.DataFrame 
        
        - axis : Perform statistics over rows(0) or columns(1) -> default (0)
        
        - level : int or level name, default None
            If the axis is a MultiIndex (hierarchical), count along a particular level, 
            collapsing into a Series.
            
        - skipna : boolean, default True
            Exclude NA/null values when computing the result.
        
        - nummeric_only : boolean, default None
            Include only float, int, boolean columns. If None, will attempt 
            to use everything, then use only numeric data. Not implemented for Series.
        
        - inplace : bool, default False
            If the expression contains an assignment, whether to perform the operation
            inplace and mutate the existing DataFrame. Otherwise, a new DataFrame
            is returned.
            
"""

   
class UnivariateAnalysis(): 
    
    #### This class expects a dataframe as the input data source. 
    
    def __init__(self, dataframe, axis, level=None, skipna=True, nummeric_only=None, inplace=False):
        
        self.dataframe = dataframe
        self.axis = axis
        self.level = level
        self.skipna = skipna
        self.nummeric_only = nummeric_only
        self.inplace = inplace        
        self.results = []
        self.value = []
        
   
    def abs_values(self):
        #Return a Series/DataFrame with absolute numeric value of each element.
        try:
            df = self.dataframe.abs()
            self.results.append(df)
            return df
        
        except Exception as Ex:
            return None
        
        
    def compound(self):
        #Return the compound percentage of the values for the requested axis try:
        try:        
            compound_percentage = self.dataframe.compound(1, self.skipna, self.level)
            self.results.append(compound_percentage)
            return self.compound_percentage
        
        except Exception as Ex:
            return None
        
    
    
    def cummax(self):
    
        #Return cumulative maximum over a DataFrame or Series axis.
        # additional_inputs : skipna
        try:        
            cummax = self.dataframe.cummax(self.axis, self.skipna)
            self.results.append(cummax)    
            return cummax
        
        except Exception as Ex:
            return None
    
    def cummin(self):
        #	Return cumulative minimum over a DataFrame or Series axis.
        # additional_inputs : skipna
    
        try:        
            cum_min = self.dataframe.cummin(self.axis, self.skipna)
            self.results.append(cum_min)    
            return cum_min
        
        except Exception as Ex:
            return None
    
    def cumsum(self):
        #	Return cumulative sum over a DataFrame or Series axis.
        # additional_inputs : skipna
        try:        
            cumsum = self.dataframe.cumsum(self.axis, self.skipna)
            self.results.append(cumsum)
            return cumsum
        
        except Exception as Ex:
            return None
    
    def cumprod(self):
        #Return cumulative product over a DataFrame or Series axis.
        try:        
            cumprod = self.dataframe.cumprod(self.axis, self.skipna)
            self.results.append(cumprod)    
            return cumprod
        
        except Exception as Ex:
            return None
    
    
    
    def mean_abs_dev(self):
        #Return the mean absolute deviation of the values for the requested axis
        try:        
            mean_abs_dev = self.dataframe.mad(self.axis, self.skipna, self.level)
            self.results.append(mean_abs_dev)
            return mean_abs_dev
        
        except Exception as Ex:
            return None
    
    
    def max_value(self):
        #This method returns the maximum of the values in the object.
        try:        
            max_value = self.dataframe.max(self.axis, self.skipna, self.level, self.nummeric_only)
            self.results.append(max_value)    
            return max_value
        
        except Exception as Ex:
            return None
    
    
    def mean(self):
        #Return the mean of the values for the requested axis 
        try:      
            print("The axis is: " + str(self.axis))
            mean = self.dataframe.mean(self.axis, self.skipna, self.level,self.nummeric_only)
            self.results.append(mean)
            return mean
        
        except Exception as Ex:
            return None
        

    def median(self):
        #Return the median of the values for the requested axis
        #, skipna, level, …
        try:        
            median = self.dataframe.median(self.axis, self.skipna, self.level)
            self.results.append(median)
            return median
        
        except Exception as Ex:
            return None
    
    
    def min_value(self):
        #This method returns the minimum of the values in the object
        try:        
            min_value = self.dataframe.min(self.axis, self.skipna, self.level)
            self.results.append(min_value)
            return min_value
        
        except Exception as Ex:
            return None
    
    
    
    def mode(self):
        #Gets the mode(s) of each element along the axis selected.
        try:        
            mode = self.dataframe.mode(self.axis, self.nummeric_only)
            self.results.append(mode)
            return mode
        
        except Exception as Ex:
            return None
    
  
    def pct_change(self):
        
        #Percentage change between the current and a prior element.
        """
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : str, default ‘pad’
            How to handle NAs before computing percent changes.
        limit : int, default None
            The number of consecutive NAs to fill before stopping.
        freq : DateOffset, timedelta, or offset alias string, optional
            Increment to use from time series API (e.g. ‘M’ or BDay()).
        """
        try:        
            pct_change = self.dataframe.pct_change(axis='columns')
            self.results.append(pct_change)
            return pct_change
        
        except Exception as Ex:
            return None
        
    
    def product(self, min_count):
        #Return the product of the values for the requested axis
        try:        
            product = self.dataframe.prod(self.axis, self.skipna, self.level, self.nummeric_only, min_count)
            self.results.append(product)    
            return product
        
        except Exception as Ex:
            return None
    
    def quantile(self, q):
        #Return values at the given quantile over requested axis, a la numpy.percentile.
        try:        
            quantile = self.dataframe.quantile(q, self.axis, self.nummeric_only)
            self.results.append(quantile)
            return quantile    
        
        except Exception as Ex:
            return None
    
        
    def rank(self, method=None, na_option = None, ascending = True, pct = False ):
        #Compute numerical data ranks (1 through n) along axis.
        #, method, numeric_only, …
        try:        
            rank = self.dataframe.rank(axis = self.axis, ascending=1)
            self.results.append(rank)    
            return rank    
        
        except Exception as Ex:
            return None
        
        
        
    def sem(self):
        
        #Return unbiased standard error of the mean over requested axis.
        try:        
            std_ub = self.dataframe.sem(self.axis, self.skipna,
                                        self.level, 1, self.nummeric_only)
            self.results.append(std_ub)
            return std_ub    
        
        except Exception as Ex:
            return None
        
  
    def std(self):
        #Return sample standard deviation over requested axis.
        #inputs: , skipna, level, ddof
        
        try:        
            std = self.dataframe.std(self.axis, self.skipna, self.level)
            self.results.append(std)
            return std   
        
        except Exception as Ex:
            return None
   
    
     
    def sum_values(self):
    
        #	Return the sum of the values for the requested axis
        try:        
            sum_values = self.dataframe.sum(self.axis, self.skipna, self.level)
            self.results.append(sum_values)
            return sum_values   
        
        except Exception as Ex:
            return None
        
    
  
        
    def unbiased_variance(self):
        #Return sample standard deviation over requested axis.
        # Documentation for ddof 
        
        try:        
            ub_variance = self.dataframe.var(self.axis, self.skipna,
                                             self.level, 1, self.nummeric_only)
            self.results.append(ub_variance)
            return ub_variance
        
        except Exception as Ex:
            return None
    
    def unique(self, dropna=True):
        #Return Series with number of distinct observations over requested axis.
        #inputs : dropna
        try:
            nunique = self.dataframe.nunique(self.axis, dropna)
            self.results.append(nunique)
            return nunique
        except Exception as Ex:
            return None
             
        
    
    def kurtosis(self):
        #Return unbiased kurtosis over requested axis using Fisher’s definition of kurtosis (kurtosis of normal == 0.0).
        try:        
            kurtosis = self.dataframe.kurtosis(self.axis, self.skipna,
                                               self.level, self.nummeric_only)
            self.results.append(kurtosis)
            return kurtosis    
        except Exception as Ex:
            return None
    
    
    def beautify_results(self):
        
        computed_methods = ["Absolute Values", "compound", "cummax", "cum", "cummin", "cumprod",
                            "cumsum", "max_value", "mean", "mean_abs_dev", "min", "mode", "pct_change", 
                            "product", "quantile", "rank" "sem", "std", "sum_values", "unbiased_variance", 
                            "unique"]
        
        with open("results.pdf", "w") as f: 
            
            for i, result in enumerate(self.results):
                
                result_dict = result.to_dict()
                header = "######## These are the results for  " + computed_methods[i] + "##############"
                f.write(header)
                f.write(result_dict)
                
            
        
        self.results
        
    
    def run(self): 
        
        self.abs_values()
        self.compound()
        self.cummax()
        self.cummin()
        self.cumprod()
        self.cumsum()
        self.kurtosis()
        self.max_value()
        self.mean()
        self.mean_abs_dev()
        self.median()
        self.min_value()
        self.mode()
        self.pct_change()
        self.product(min_count=self.dataframe.shape[0])
        self.quantile(q=20)
        self.rank()
        self.sem()
        self.std()
        self.sum_values()
        self.unbiased_variance()
        self.unique()
        
        print("Univariate Analysis Computed")
        print("Saving Results into a pdf")
#        self.beautify_results()
        
        
        
#####################################################################################################################
#####################################################################################################################
    
   
class BiVariateAnalysis():
    
    
    def __init__(self, dataframe, level=None, axis=None):
        
        self.dataframe = dataframe
        self.axis = axis
        self.results = []
        
    
    def cov(self, min_periods):
    
        #Compute pairwise covariance of columns, excluding NA/null values.
        #min_periods : int, optional
            #Minimum number of observations required per pair of 
            #columns to have a valid result.
        try:        
            cov = self.dataframe.cov(min_periods)
            self.results.append(cov)
            return cov
        
        except Exception as Ex:
            return None
    

    def corr(self, method, min_periods=None):
        #Compute pairwise correlation of columns, excluding NA/null values
        #additional_inputs : method, min_periods.
    
        """
        method : {‘pearson’, ‘kendall’, ‘spearman’}
    
            pearson : standard correlation coefficient
            kendall : Kendall Tau correlation coefficient
            pearman : Spearman rank correlation
    
        min_periods : int, optional
    
            Minimum number of observations required per pair of columns 
            to have a valid result. Currently only available for pearson
            and spearman correlation
        """
        
        try:        
            corr = self.dataframe.corr(method, min_periods)
            self.results.append(corr)
            return corr
        
        except Exception as Ex:
            return None
     
        
    def corr_with(self, other, drop):
        #Compute pairwise correlation between rows or columns of two DataFrame objects.
        #inputs : other[, axis, drop]
    
        try:        
            corr_with = self.dataframe.corrwith(other, self.axis, drop)
            self.results.append(corr_with)
            return corr_with
        
        except Exception as Ex:
            return None
    
    def heatmap(self, corr):
        
        sns.heatmap(corr,
                    xticklabels=corr.columns.values, 
                    yticklabels=corr.columns.values, 
                    annot=True)
        
            
    #fig = plt.figure(figsize=(10, 8))
    #
    #genre_sales_percentages_by_year = (vg_df.groupby(['Year_of_Release', 'Genre']).Global_Sales.sum())*(100)/vg_df.groupby(['Year_of_Release']).Global_Sales.sum()
    #genre_sales_percentages_by_year.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', grid=False, figsize=(13, 4))
    #
    #yearlySales = vg_df.groupby(['Year_of_Release','Genre']).Global_Sales.sum()
    #yearlySales.unstack().plot(kind='area',stacked=True, colormap= 'Spectral', figsize=(13, 4) ) ;


        
    
    def most_highly_correlated(self, corr_mat, numtoreport):
        
        try:
            # find the correlations
            cormatrix = corr_mat
            # set the correlations on the diagonal or lower triangle to zero,
            # so they will not be reported as the highest ones:
            cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
            # find the top n correlations
            cormatrix = cormatrix.stack()
            cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
            # assign human-friendly names
            cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
            
            report = cormatrix.head(numtoreport)
            self.results.append(report)
            return report, corrmatrix
        
        except Exception as Ex:
            return None
    
    

class MultiVariateAnalysis(object):
    
    def __init__(self, dataframe, axis, level=None, skipna=True, nummeric_only=None, inplace=False):
    
        self.dataframe = dataframe
        self.axis = axis
        self.level = level
        self.skipna = skipna
        self.nummeric_only = nummeric_only
        self.inplace = inplace        
        self.results = []
        
        self.value = []

    


    
#####################################################################################################################
#####################################################################################################################
        
    
class Summary_Statistics(object):
    
    
    def __init__(self, dataframe, axis): 
        
        self.dataframe = dataframe
        self.axis = axis
        
    
    def describe(self, percentiles, include=None, exclude=None):
        #Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        try:        
            desc_statistics = self.dataframe.describe(percentiles, include, exclude)
            self.results.append(desc_statistics)
            return desc_statistics
        
        except Exception as Ex:
            return None



    
#####################################################################################################################
#####################################################################################################################
    

    
class UtilsAnalysis(object):
    
    def __init__(self, dataframe, axis, level=None, skipna=True, nummeric_only=None, inplace=False):
        
        self.dataframe = dataframe
        self.axis = axis
        self.level = level
        self.skipna = skipna
        self.nummeric_only = nummeric_only
        self.inplace = inplace        
        self.results = []
        
        self.value = []
                
        
    def all_true(self, bool_only):
        #Return whether all elements are True, potentially over an axis.
        try:
            
            boolean = self.dataframe.all(self.axis, bool_only,self.skipna, self.level)
            self.results.append(boolean)
            return boolean
        
        except Exception as Ex:
            return None
    
    def any_true(self, bool_only):
        #	Return whether any element is True over requested axis.
        
        try:
            boolean = self.dataframe.any(self.axis, bool_only, self.skipna, self.level)
            self.results.append(boolean)
            return boolean
        
        except Exception as Ex:
            return None
    
    def clip_values(self, lower, upper):
        #	Trim values at input threshold(s)
        try:
            df = self.dataframe.clip(lower, upper, self.axis, self.inplace)
            self.results.append(df)
            return df
        
        except Exception as Ex:
            return None
    
    
    def clip_lower(self, threshold):
        #Return copy of the input with values below a threshold truncated.
        #inputs : threshold[, axis, inplace]
        try:
            df = self.dataframe.clip_lower(threshold, self.axis, self.inplace)
            self.results.append(df)
            return df
        
        except Exception as Ex:
            return None
    
    
    def clip_upper(self, threshold):
       #Return copy of input with values above given value(s) truncated.
       #inputs : threshold[, axis, inplace]
       try:
           df = self.dataframe.clip_upper(threshold, self.axis, self.inplace)
           self.results.append(df)
           return df
        
       except Exception as Ex:
           return None
    
   
    
    def count_non_na(self):
    
        #	Count non-NA cells for each column or row.
        # Additional_inputs : level, numeric_only
        try:        
            non_na = self.dataframe.count(self.axis, self.level, self.nummeric_only)
            self.results.append(non_na)
            return non_na
        
        except Exception as Ex:
            return None
            
    def evaluate(self, expression):
        #Evaluate a string describing operations on DataFrame columns.
        #inplace : bool, default False
            #If the expression contains an assignment,
            #whether to perform the operation inplace and
            #mutate the existing DataFrame. Otherwise, a new DataFrame is returned.

        try:        
            result = self.dataframe.eval(expression, self.inplace)
            self.results.append(result)
            return result
        
        except Exception as Ex:
            return None
    
    
    
    
    
    def diff(self, periods):
        #First discrete difference of element.
        #periods : int, default 1
            #Periods to shift for calculating difference, accepts negative values.
        try:        
            discrete_diff = self.dataframe.diff(periods, self.axis)
            self.results.append(discrete_diff)
            return discrete_diff
        
        except Exception as Ex:
            return None
    
    
    
  
    def round_to_number(self, decimals):
        """
        Number of decimal places to round each column to. If an int is given,
        round each column to the same number of places.
        
        Otherwise dict and Series round to variable numbers of places.
        Column names should be in the keys if decimals is a dict-like, 
        or in the index if decimals is a Series. Any columns not included 
        in decimals will be left as is. Elements of decimals which are not columns of the input will be ignored.
        """
        #Round a DataFrame to a variable number of decimal places.
        
        try:        
            df = self.dataframe.round(decimals)	
            self.results.append(df)
            return df
        
        except Exception as Ex:
            return None
    
  
    
def statistical_unit(df):
    
    
    print("Please Select which Analysis Test you would like to run: ")
    
    print("[1] Univariate Analysis")
    print("[2] BiVariate Analysis")
    print("[3] Multivariate Analysis")
    print("[4] Select Method")
    
    opt = int(input("Choose an option : "))
    
    
    if opt == 1:
        stats = UnivariateAnalysis(df, axis=1, level=None, skipna=True, nummeric_only=True, inplace=False) 
        stats.run()
        
    elif opt == 2:
        stats = BiVariateAnalysis(df, level=None, axis=1)
       
        
    elif opt == 3: 
        stats = MultiVariateAnalysis()
        
    elif opt == 4:
        
        opt = int(input('[1]  Absolute Values :',
                        '[2]  Compound : ',
                        '[3]  Cummax : ',
                        '[4]  Cummin : ',
                        '[5]  Cumpro : ',
                        '[6]  Cumsum :',
                        '[7]  Kurtosis : ',
                        '[8]  Max_value : ',
                        '[9]  Mean: ',
                        '[10] Mean_abs_dev: ',
                        '[11] Median: ',
                        '[12] Min_value: ',
                        '[13] Mode: ',
                        '[14] Pct_change :',
                        '[15] Product :',
                        '[16] Quantile : ',
                        '[17] Rank : ',
                        '[18] Results :',
                        '[19] Sem :sem',
                        '[20] Std :std',
                        '[21] Sum_values: ',
                        '[22] Unbiased Variance :',
                        '[23] Unique :',
                        '[24] Value :'))
        
        
        
        
                        
        
    results = stats.results
    return results


##########################################
##########################################
##########################################
##########################################

#
## The custom dictionary
#class member_table(dict):
#  def __init__(self):
#     self.member_names = []
#
#  def __setitem__(self, key, value):
#     # if the key is not already defined, add to the
#     # list of keys.
#     if key not in self:
#        self.member_names.append(key)
#
#     # Call superclass
#     dict.__setitem__(self, key, value)
#
#
## The metaclass
#class OrderedClass(type):
#
#   # The prepare function
#   @classmethod
#   def __prepare__(metacls, name, bases): # No keywords in this case
#      return member_table()
#
#   # The metaclass invocation
#   def __new__(cls, name, bases, classdict):
#      # Note that we replace the classdict with a regular
#      # dict before passing it to the superclass, so that we
#      # don't continue to record member names after the class
#      # has been created.
#      result = type.__new__(cls, name, bases, dict(classdict))
#      result.member_names = classdict.member_names
#      return result
#
#class MyClass(metaclass=OrderedClass):
#  # method1 goes in array element 0
#  def method1(self):
#     pass
#
#  # method2 goes in array element 1
#  def method2(self):
#     pass
#
#x = MyClass()
#print([name for name in x.member_names if hasattr(getattr(x, name), '__call__')])
#


##########################################
##########################################
##########################################
##########################################

    
def dependency_stats(df):
    
    
    print("Here we are....")
    
    
    summary_stats = BiVariateAnalysis(df, axis=1, level=None) 
    
    
#    summary_stats = Statistics_Inference(df, axis=1, level=None, skipna=True, nummeric_only=True,
#                                           inplace=False) 
    
    
    corr_pearson = summary_stats.corr('pearson', None)
    number_report = df.shape[1]
    summary_stats.most_highly_correlated(corr_pearson, number_report)
    
    # Compute covariance matrix.
    
    summary_stats.cov(min_periods = None)

    results = summary_stats.results
    print(results)
    return results









###############################################################


def extract_correlation_values(column, corr_matrix):

    
    lpt_correlation_dict = {}
    
    dict1 = corr_matrix[column].to_dict()
    
    index = [i for i, row in enumerate(corr_matrix.iterrows()) if row[0] == column]

    dict2 = corr_matrix.iloc[index[0]][:].to_dict()
    
    
    for variable in dict1.keys():
        
        lpt_correlation_dict[variable] = abs(dict1[variable] + dict2[variable])
    
    
    for variable in lpt_correlation_dict.keys():
        
#        print("lpt_correlation_dict[", variable, "]: ", lpt_correlation_dict[variable])
        
        if lpt_correlation_dict[variable] is np.NAN:
            
            lpt_correlation_dict.pop(variable)
            
            
    return lpt_correlation_dict
    


#### Compute correlation at some level...
    

    
def corrLevel(level="bookie_id"):
    """
    Implement a function that takes level=column nane and measures correlation at this level. 
    level : "bookie_id"
    """
    
    lpt_r_values_bookie = {}
    
    
    for bookie_id in final_table_with_probabilities[level].unique():
        
        df_bookie = final_table_with_probabilities[final_table_with_probabilities["bookie_id"] == bookie_id]
        
        computed_statistics = Statistics.dependency_stats(df_bookie)
        r_team_end_score = extract_correlation_values("team_end_score_sum", computed_statistics[0])
        r_team_end_score = sort_values(r_team_end_score, by="descending")
        
        lpt_r_values_bookie[bookie_id] = r_team_end_score
    
# Computing Pearson Correlation values at a Team/bookie/market level. 
        
        
        
        
        

#############################
######### CORRELATION #######
#############################

### Pearson´s correlation. 
### Spearman´s rank correlation. 
### Kendall´s rank correlation.

### Spearman and Kendall are referred to as distribution-free correlation or non-parametric correlation. 
### Rank correlation measures are often used as the basis for other statistical hypothesis tests, such as determining whether two 
### samples were drawn from the same (or different) population distributions. 


## Spearman´s rank correlation : Quantifies the degree to which the ranked variables are associated by a monotonic function, meaning an increasing or
## decreasing relationship. 

## Pearson´s correlation is the calculation of the covariance(or expected difference of observations from the mean) between the two variables normalized
## by the variance or spread of both variables. 


corr_pearson = df_final.corr("pearson", min_periods=None)

##########################################################

from numpy.random import rand
from numpy.random import seed
from scipy.stats import spearmanr
# seed random number generator
seed(1)

df_final.dropna()

variable = "team_end_score_sum"
other_variables = [column for column in columns if column != "team_end_score_sum"]

rho = {}
p = {}

for other_variable in other_variables:
    
    spearman = spearmanr(df_final[variable], df_final[other_variable])
    
    rho[other_variable] = spearman[0]
    p[other_variable] = spearman[1]



#############################################################

#corr_spearman = df_final.corr("spearman", min_periods=None)
corr_kendall = df_final.corr("kendall", min_periods=None)


sns_plot_pearson = sns.heatmap(corr_pearson, xticklabels=corr_pearson.columns.values,  yticklabels=corr_pearson.columns.values, annot=True)
sns_plot_spearman =  sns.heatmap(corr_spearman, xticklabels=corr_spearman.columns.values,  yticklabels=corr_spearman.columns.values, annot=True)
sns_plot_kendall = sns.heatmap(corr_kendall, xticklabels=corr_kendall.columns.values,  yticklabels=corr_kendall.columns.values, annot=True)

fig_pearson = sns_plot_pearson.get_figure()
fig_spearman = sns_plot_spearman.get_figure()
fig_kendall = sns_plot_kendall.get_figure()

sns.set(style="darkgrid")

path = os.getcwd()
fig_pearson.savefig(path + "/figures" + "/pearson_heatmap.png")
fig_kendall.savefig(path + "/figures" + "/kendall_heatmap.png")


#############################
######### COVARIANCE ########
#############################

def covariance():

    cov = df_final.cov(min_periods=None)
    cov.to_csv("covariance.csv", sep=";")
    covariance_read = pd.read_csv("covariance.csv", sep=";")

        
        
        




################################################################################################################################################################
import sys
import pandas as pd
path_sources = ".//Sources//sr_match_14894425//"

sys.path.append(path_sources)

from Statistics import Statistics_Inference 
df = pd.read_csv(path_sources + "df_player_stats_cumulative_match.csv", sep=";")

################################################################################################################################################################

def generate_df_filtered(dataframe, column_name):
    
    for value in dataframe[column_name].unique():
        
        df = dataframe.loc[dataframe[column_name] == value]
        yield df
    
    
def generate_statistics(dataframe, method):
    
    stats_df = Statistics_Inference(dataframe, 1)  
    
    value = stats_df(5)
    print("value is :", value)


    correlation = getattr(stats_df, method)
    correlation("pearson", None)
    
    yield stats_df
    


dataframes_teams = [dataframe for dataframe in generate_df_filtered(cumulative_data_team, "team_name")]
stats_dataframes_teams = [stats_df for dataframe in dataframes_teams for stats_df in generate_statistics(dataframe, "corr")]
results_corr_teams = [stats_df.results for stats_df in stats_dataframes_teams]

dataframes_players = [dataframe for dataframe in generate_df_filtered(cumulative_data_player, "player_name")]
stats_dataframes_players = [stats_df for dataframe in dataframes_players for stats_df in generate_statistics(dataframe, "corr")]
results_corr_players = [stats_df.results for stats_df in stats_dataframes_players]


##############



########################################################################################
###################---------DATA ANALYSIS-------------##################################
########################################################################################

# 14 - Perform Statistics on the Data.Compute Statistics over variables, estimate how do they relate to eache other,
# how dependent on each other they are, etc.

summary_stats = Statistics_Computation(df_data, axis=1, level=None, skipna=True, nummeric_only=True,
                                       inplace=False) 



# Compute abs_values.
abs_values = summary_stats.abs_values()
# Compute pearson correlation.
corr_Pearson = summary_stats.corr('pearson', None)
#For the Pearson r correlation, both variables should be normally distributed
# (normally distributed variables have a bell-shaped curve).  Other assumptions
# include linearity and homoscedasticity.  Linearity assumes a straight line 
# relationship between each of the two variables and homoscedasticity assumes 
# that data is equally distributed about the regression line.

# Compute kendall correlation.
corr_Kendall = summary_stats.corr('kendall', None)
# Compute spearman correlation.
from scipy.stats import spearmanr
corr_Spearman, pvalue = spearmanr(X)
summary_stats.results.append(corr_Spearman)

###Outputting a score of the most highly correlated variables for each mode : 'Pearson', 'Kendall', 'Spearman'
number_report = X.shape[1]
most_correlated_Pearson = summary_stats.most_highly_correlated(corr_Pearson, number_report)
most_correlated_Kendall = summary_stats.most_highly_correlated(corr_Kendall, number_report)
most_correlated_Spearman = summary_stats.most_highly_correlated(pd.DataFrame(corr_Spearman), number_report)

# Compute covariance matrix.
cov_matrix = summary_stats.cov(min_periods = None)

# Compute cummax, cummin and cumprod
summary_stats.axis = 0
cummax = summary_stats.cummax()
cummin = summary_stats.cummin()
cumprod = summary_stats.cumprod()

# Compute mean, std, median, mean_abs_dev over columns/"variables"
mean = summary_stats.mean()
std = summary_stats.std()
median = summary_stats.median()
mean_abs_dev = summary_stats.mean_abs_dev()

# Inferring unbiased statistical results:
# standard error of the mean and unbiased variance. 
sem = summary_stats.sem()
unbiased_variance = summary_stats.unbiased_variance()


# Compute min_value and max_value and sum_values
min_value = summary_stats.min_value()
max_value = summary_stats.max_value()
sum_values = summary_stats.sum_values()


# Compute kurtosis.

""" Used to describe the shape of a distribution's 
    tails in relation to its overall shape. A distribution can be infinitely 
    peaked with low kurtosis, and a distribution can be perfectly flat-topped
    with infinite kurtosis its “tailedness,” not “peakedness.”
"""   
kurtosis = summary_stats.kurtosis()

# Compute mode to see what value appears more often and pct_change to see how percentually
# their value changes along the samples for each variable.
"""Note: 
    In computed pct_change -> Element with 'nan' value means no change with respect of the 
    previous element in the axis = 1, columns.
"""
mode = summary_stats.mode()
pct_change = summary_stats.pct_change(periods = 1)

# Compute rank.
summary_stats.axis = 1
rank = summary_stats.rank()
#results = summary_stats.results

##### Multi-Collinearity.


from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# First drop na values()
# Second apply ._get_nummeric_data to get rid of the non-nummerical Series of the dataframe. 


#ind_predictor = list(df_final.columns.values).index("team_end_score_sum")

features = "+".join(columns)
y, X = dmatrices('team_end_score_sum ~' + features, df_final, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif.to_csv("variation_inflation.csv", sep=";", index=False)


#3 - ######### Inferential Statistics / Regression Analysis. ##########


##############################################
##### Chi2 ##################################
##############################################

#### Chi-Square Test is used whether two categorical variables are related or independent. 

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#### Split data target value = y and non_target values = X

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)


### To retrieve chi2 statistics of each feature: 

#chi2, pvalues = chi2(X,y)

##############################################
##### Tree model Selection ###################
##############################################

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)



##############################################
##### Independence T-test ####################
##############################################

#### Independent t-test(Welch’s t-test) : 
#### To perform a t-test and F-test(Anova), look at t-test values at different levels : bookie, bookie + market, bookie + market + other_markets.

from scipy.stats import ttest_ind
import numpy as np

ttest = {}
p_value = {}
ttest, p_value = ttest_ind(array1, array2, equal_var=False)


##############################################
##### Anova F-test ###########################
##############################################
            
#### Analysis of Variance Group Test. 
            
# F = Between Group Variability/Within Group Variability. 

import scipy.stats as stats

F, p = stats.f_oneway(*values)


##############################################
##### Wilcoxon t-test ########################
##############################################

from scipy.stats import wilcoxon
W, p = wilcoxon(array1, array2, zero_method="wilcox")


#############################################
##### Kruskal-Wallis H Test #################
#############################################
            
from numpy.random import seed
from numpy.random import randn
from scipy.stats import kruskal

seed(1)
KW, p_wallis = kruskal(market1, market2)


#############################################
##### Friedman Test #################
#############################################
       
## Tests whether the distributions of two or more paired samples are equal or not.

from scipy.stats import friedmanchisquare
stat, p = friedmanchisquare(Serie1, Serie2, Serie3)


#############################################
##### Multi-Variate Analysis of Variance ########
#############################################

### It won´t work since we don´t count with categorical variables. 

from statsmodels.multivariate.manova import MANOVA
multivariate_analysis = MANOVA(dependent_variables, independent_variables)




"""
Perform Statistic over Data, Dataframes. Provides functionalities to statistically infer
explanation from data: 
    correlation, covriance, mean, std_dev, mean_abs_dev, abs_value, pct_change, 
    kurtosis, cummulative scores, etc can be found. 
    
    
Notes : 
    
    The bulk of the Statistical Logic can be found in the class Statistics_Computation.
    


To-Do's:
    
    Improve: 
        Better Structure the Logic at Statistics_Computation class:
            Use Decorators, Abstract Classes, etc?´.
    
    Perform :
        Chi-Square Analysis, G-Test, Fisher's Exact, p_value, etc. 
    
        log-linear models or classification trees. 
        
        Contingency Table.

"""


# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:25:55 2018

@author: Francisco
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

    


    
class Statistics_Computation:

    """
    Perform pandas Summary Statistics over data. Methods to compute statistics over 
    a dataframe or a serie :
             
        Inputs:
            
            - dataframe : pd.Series or pd.DataFrame 
            
            - axis : Perform statistics over rows(0) or columns(1) -> default (0)
            
            - level : int or level name, default None
                If the axis is a MultiIndex (hierarchical), count along a particular level, 
                collapsing into a Series.
                
            - skipna : boolean, default True
                Exclude NA/null values when computing the result.
            
            - nummeric_only : boolean, default None
                Include only float, int, boolean columns. If None, will attempt 
                to use everything, then use only numeric data. Not implemented for Series.
            
            - inplace : bool, default False
                If the expression contains an assignment, whether to perform the operation
                inplace and mutate the existing DataFrame. Otherwise, a new DataFrame
                is returned.
    """

   
    
    def __init__(self, dataframe, axis, level , skipna, nummeric_only, inplace):
    
        
        self.dataframe = dataframe
        self.axis = axis
        self.level = level
        self.skipna = skipna
        self.nummeric_only = nummeric_only
        self.inplace = inplace        
        self.results = []
        
        #self.abs_values = None
        #self.compound_percentage = None
        #self.corr = None
        #self.cov = None
        
    
    def most_highly_correlated(self, corr_mat, numtoreport):
        
        try:
            # find the correlations
            cormatrix = corr_mat
            # set the correlations on the diagonal or lower triangle to zero,
            # so they will not be reported as the highest ones:
            cormatrix *= np.tri(*cormatrix.values.shape, k=-1).T
            # find the top n correlations
            cormatrix = cormatrix.stack()
            cormatrix = cormatrix.reindex(cormatrix.abs().sort_values(ascending=False).index).reset_index()
            # assign human-friendly names
            cormatrix.columns = ["FirstVariable", "SecondVariable", "Correlation"]
            
            report = cormatrix.head(numtoreport)
            self.results.append(report)
            return report
        
        except Exception as Ex:
            return None

        
    def abs_values(self):
        #Return a Series/DataFrame with absolute numeric value of each element.
        try:
            df = self.dataframe.abs()
            self.results.append(df)
            return df
        
        except Exception as Ex:
            return None
        
        
    def all_true(self, bool_only):
        #Return whether all elements are True, potentially over an axis.
        try:
            
            boolean = self.dataframe.all(self.axis, bool_only,self.skipna, self.level)
            self.results.append(boolean)
            return boolean
        
        except Exception as Ex:
            return None
    
    def any_true(self, bool_only):
        #	Return whether any element is True over requested axis.
        
        try:
            boolean = self.dataframe.any(self.axis, bool_only, self.skipna, self.level)
            self.results.append(boolean)
            return boolean
        
        except Exception as Ex:
            return None
    
    def clip_values(self, lower, upper):
        #	Trim values at input threshold(s)
        try:
            df = self.dataframe.clip(lower, upper, self.axis, self.inplace)
            self.results.append(df)
            return df
        
        except Exception as Ex:
            return None
    
    
    def clip_lower(self, threshold):
        #Return copy of the input with values below a threshold truncated.
        #inputs : threshold[, axis, inplace]
        try:
            df = self.dataframe.clip_lower(threshold, self.axis, self.inplace)
            self.results.append(df)
            return df
        
        except Exception as Ex:
            return None
    
    
    def clip_upper(self, threshold):
       #Return copy of input with values above given value(s) truncated.
       #inputs : threshold[, axis, inplace]
       try:
           df = self.dataframe.clip_upper(threshold, self.axis, self.inplace)
           self.results.append(df)
           return df
        
       except Exception as Ex:
           return None
    
    
    def compound(self):
        #Return the compound percentage of the values for the requested axis try:
        try:        
            compound_percentage = self.dataframe.compound(1, self.skipna, self.level)
            self.results.append(compound_percentage)
            return self.compound_percentage
        
        except Exception as Ex:
            return None
    
    def corr(self, method, min_periods):
        #Compute pairwise correlation of columns, excluding NA/null values
        #additional_inputs : method, min_periods.

        """
        method : {‘pearson’, ‘kendall’, ‘spearman’}

            pearson : standard correlation coefficient
            kendall : Kendall Tau correlation coefficient
            pearman : Spearman rank correlation

        min_periods : int, optional

            Minimum number of observations required per pair of columns 
            to have a valid result. Currently only available for pearson
            and spearman correlation
        """
        
        try:        
            corr = self.dataframe.corr(method, min_periods)
            self.results.append(corr)
            return corr
        
        except Exception as Ex:
            return None
     
        
    def corr_with(self, other, drop):
        #Compute pairwise correlation between rows or columns of two DataFrame objects.
        #inputs : other[, axis, drop]

        try:        
            corr_with = self.dataframe.corrwith(other, self.axis, drop)
            self.results.append(corr_with)
            return corr_with
        
        except Exception as Ex:
            return None
    
    def count_non_na(self):
    
        #	Count non-NA cells for each column or row.
        # Additional_inputs : level, numeric_only
        try:        
            non_na = self.dataframe.count(self.axis, self.level, self.nummeric_only)
            self.results.append(non_na)
            return non_na
        
        except Exception as Ex:
            return None
            
    def cov(self, min_periods):
    
        #Compute pairwise covariance of columns, excluding NA/null values.
        #min_periods : int, optional
            #Minimum number of observations required per pair of 
            #columns to have a valid result.
        try:        
            cov = self.dataframe.cov(min_periods)
            self.results.append(cov)
            return cov
        
        except Exception as Ex:
            return None
    
    def cummax(self):
    
        #Return cumulative maximum over a DataFrame or Series axis.
        # additional_inputs : skipna
        try:        
            cummax = self.dataframe.cummax(self.axis, self.skipna)
            self.results.append(cummax)    
            return cummax
        
        except Exception as Ex:
            return None
    
    def cummin(self):
        #	Return cumulative minimum over a DataFrame or Series axis.
        # additional_inputs : skipna
    
        try:        
            cum_min = self.dataframe.cummin(self.axis, self.skipna)
            self.results.append(cum_min)    
            return cum_min
        
        except Exception as Ex:
            return None
    
    def cumsum(self):
        #	Return cumulative sum over a DataFrame or Series axis.
        # additional_inputs : skipna
        try:        
            cumsum = self.dataframe.cumsum(self.axis, self.skipna)
            self.results.append(cumsum)
            return cumsum
        
        except Exception as Ex:
            return None
    
    def cumprod(self):
        #Return cumulative product over a DataFrame or Series axis.
        try:        
            cumprod = self.dataframe.cumprod(self.axis, self.skipna)
            self.results.append(cumprod)    
            return cumprod
        
        except Exception as Ex:
            return None
    
    
    def describe(self, percentiles, include=None, exclude=None):
        #Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        try:        
            desc_statistics = self.dataframe.describe(percentiles, include, exclude)
            self.results.append(desc_statistics)
            return desc_statistics
        
        except Exception as Ex:
            return None
    
    
    def diff(self, periods):
        #First discrete difference of element.
        #periods : int, default 1
            #Periods to shift for calculating difference, accepts negative values.
        try:        
            discrete_diff = self.dataframe.diff(periods, self.axis)
            self.results.append(discrete_diff)
            return discrete_diff
        
        except Exception as Ex:
            return None
    
    def evaluate(self, expression):
        #Evaluate a string describing operations on DataFrame columns.
        #inplace : bool, default False
            #If the expression contains an assignment,
            #whether to perform the operation inplace and
            #mutate the existing DataFrame. Otherwise, a new DataFrame is returned.

        try:        
            result = self.dataframe.eval(expression, self.inplace)
            self.results.append(result)
            return result
        
        except Exception as Ex:
            return None
    
    
    def kurtosis(self):
        #Return unbiased kurtosis over requested axis using Fisher’s definition of kurtosis (kurtosis of normal == 0.0).
        try:        
            kurtosis = self.dataframe.kurtosis(self.axis, self.skipna,
                                               self.level, self.nummeric_only)
            self.results.append(kurtosis)
            return kurtosis    
        except Exception as Ex:
            return None
    
    
    def mean_abs_dev(self):
        #Return the mean absolute deviation of the values for the requested axis
        try:        
            mean_abs_dev = self.dataframe.mad(self.axis, self.skipna, self.level)
            self.results.append(mean_abs_dev)
            return mean_abs_dev
        
        except Exception as Ex:
            return None
        
    def max_value(self):
        #This method returns the maximum of the values in the object.
        try:        
            max_value = self.dataframe.max(self.axis, self.skipna, self.level, self.nummeric_only)
            self.results.append(max_value)    
            return max_value
        
        except Exception as Ex:
            return None
        
    
    def mean(self):
        #Return the mean of the values for the requested axis 
        try:      
            print("The axis is: " + str(self.axis))
            mean = self.dataframe.mean(self.axis, self.skipna, self.level,self.nummeric_only)
            self.results.append(mean)
            return mean
        
        except Exception as Ex:
            return None
        

    def median(self):
        #Return the median of the values for the requested axis
        #, skipna, level, …
        try:        
            median = self.dataframe.median(self.axis, self.skipna, self.level)
            self.results.append(median)
            return median
        
        except Exception as Ex:
            return None
    
    
    
    def mode(self):
        #Gets the mode(s) of each element along the axis selected.
        try:        
            mode = self.dataframe.mode(self.axis, self.nummeric_only)
            self.results.append(mode)
            return mode
        
        except Exception as Ex:
            return None
    
    def min_value(self):
        #This method returns the minimum of the values in the object
        try:        
            min_value = self.dataframe.min(self.axis, self.skipna, self.level)
            self.results.append(min_value)
            return min_value
        
        except Exception as Ex:
            return None
    
    def pct_change(self):
        
        #Percentage change between the current and a prior element.
        """
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : str, default ‘pad’
            How to handle NAs before computing percent changes.
        limit : int, default None
            The number of consecutive NAs to fill before stopping.
        freq : DateOffset, timedelta, or offset alias string, optional
            Increment to use from time series API (e.g. ‘M’ or BDay()).
        """
        try:        
            pct_change = self.dataframe.pct_change(axis='columns')
            self.results.append(pct_change)
            return pct_change
        
        except Exception as Ex:
            return None
    
    
    def product(self, min_count):
        #Return the product of the values for the requested axis
        try:        
            product = self.dataframe.prod(self.axis, self.skipna, self.level, self.nummeric_only, min_count)
            self.results.append(product)    
            return product
        
        except Exception as Ex:
            return None
    
    def quantile(self, q):
        #Return values at the given quantile over requested axis, a la numpy.percentile.
        try:        
            quantile = self.dataframe.quantile(q, self.axis, self.nummeric_only)
            self.results.append(quantile)
            return quantile    
        
        except Exception as Ex:
            return None
    
    
    def rank(self, method=None, na_option = None, ascending = True, pct = False ):
        #Compute numerical data ranks (1 through n) along axis.
        #, method, numeric_only, …
        try:        
            rank = self.dataframe.rank(axis = self.axis, ascending=1)
            self.results.append(rank)    
            return rank    
        
        except Exception as Ex:
            return None
    
    def round_to_number(self, decimals):
        """
        Number of decimal places to round each column to. If an int is given,
        round each column to the same number of places.
        
        Otherwise dict and Series round to variable numbers of places.
        Column names should be in the keys if decimals is a dict-like, 
        or in the index if decimals is a Series. Any columns not included 
        in decimals will be left as is. Elements of decimals which are not columns of the input will be ignored.
        """
        #Round a DataFrame to a variable number of decimal places.
        
        try:        
            df = self.dataframe.round(decimals)	
            self.results.append(df)
            return df
        
        except Exception as Ex:
            return None
    
    def sem(self):
        
        #Return unbiased standard error of the mean over requested axis.
        try:        
            std_ub = self.dataframe.sem(self.axis, self.skipna,
                                        self.level, 1, self.nummeric_only)
            self.results.append(std_ub)
            return std_ub    
        
        except Exception as Ex:
            return None
        
    def skew(self):
        
        #Return unbiased skew over requested axis Normalized by N-1
        # skipna, level, …
        
        try:            
            skew = self.dataframe.skew(self.axis, self.skipna, self.level)
            self.results.append(skew)
            return skew   
        
        except Exception as Ex:
            return None
    
    def sum_values(self):
        
        #	Return the sum of the values for the requested axis
        try:        
            sum_values = self.dataframe.sum(self.axis, self.skipna, self.level)
            self.results.append(sum_values)
            return sum_values   
        
        except Exception as Ex:
            return None
        
    
    def std(self):
        #Return sample standard deviation over requested axis.
        #inputs: , skipna, level, ddof
        
        try:        
            std = self.dataframe.std(self.axis, self.skipna, self.level)
            self.results.append(std)
            return std   
        
        except Exception as Ex:
            return None
        
    def unbiased_variance(self):
        #Return sample standard deviation over requested axis.
        # Documentation for ddof 
        
        try:        
            ub_variance = self.dataframe.var(self.axis, self.skipna,
                                             self.level, 1, self.nummeric_only)
            self.results.append(ub_variance)
            return ub_variance
        
        except Exception as Ex:
            return None
    
    def unique(self, dropna):
        #Return Series with number of distinct observations over requested axis.
        #inputs : dropna
        try:
            nunique = self.dataframe.nunique(self.axis, dropna)
            self.results.append(nunique)
            return nunique
        except Exception as Ex:
            return None
        





def dependency_stats(df):

    summary_stats = Statistics_Computation(df, axis=1, level=None, skipna=True, nummeric_only=True,
                                           inplace=False) 
    
    corr_pearson = summary_stats.corr('pearson', None)
    number_report = df.shape[1]
    summary_stats.most_highly_correlated(corr_pearson, number_report)
    
    # Compute covariance matrix.
    summary_stats.cov(min_periods = None)

    results = summary_stats.results
    return results






##Other Statistical Inference Methods :
    # F-Test. 
    # Chi-Square Test. 
    # Anova Test. 
    # Multivariate analysis. 
    # Bivariate and univariate analysis of data. 
    # Create Statistical Models / Hypothesis Testing. 
    # Measure how skewed distributions are. 
    # Covariance, Variance, etc. 
    # Compute abs values, most_correlated_values. 
    # Compute cummax, cummin, cumprod. 
    # Compute mean, std, median, std. 
    # min_value, max_value. 
    # kurtosis, mode, pct_change, rank.
    
#TO-DO : 
    # import plotlet. 
    # import XGBoost/LightGBM, CatBoost. 
    # import Eli5 -> visualization and debugging machine learning models and tracking the work of an algorithm step by step. 
