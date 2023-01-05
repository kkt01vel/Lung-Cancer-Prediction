
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns

sns.set()
px.defaults.width = 800
px.defaults.height = 500

# Functions Exploratory Analysis
class EDA:

    def __init__(self, dataframe):
        self.df = dataframe
        self.columns = self.df.columns
        self.num_vars = self.df.select_dtypes(include=[np.number]).columns
        self.cat_vars = self.df.select_dtypes(include=[np.object]).columns

        
    def histogram_num(self, main_var, hue=None, bins = None, ranger=None):
        return  px.histogram(self.df[self.df[main_var].between(left = ranger[0], right = ranger[1])], \
            x=main_var, nbins =bins , color=hue, marginal='violin')

    def scatter_plot(self, col_x,col_y,hue=None, size=None):
        return px.scatter(self.df, x=col_x, y=col_y, color=hue,size=size)

    def bar_plot(self, col_y, col_x, hue=None):
        return px.bar(self.df, x=col_x, y=col_y,color=hue)
        
      
    def heatmap_vars(self,cols, func = np.mean):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        chart = sns.heatmap(self.df.pivot_table(index =cols[0], columns =cols[1],  values =cols[2], aggfunc=func, fill_value=0).dropna(axis=1), annot=True, annot_kws={"size": 7}, linewidths=.5)
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def Corr(self, cols=None, method = 'pearson'):
        sns.set(style="whitegrid")
        sns.set(font_scale=0.6)
        if len(cols) != 0:
            corr = self.df[cols].corr(method = method)
        else:
            corr = self.df.corr(method = method)
        chart = sns.heatmap(corr, annot=True, annot_kws={"size": 7}, linewidths=.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30)
        chart.set_yticklabels(chart.get_yticklabels(), rotation=30)
        return chart
   
    
   

@st.cache
def get_stats(df):
    stats_num = df.describe()
    if df.select_dtypes(np.object).empty :
        return stats_num.transpose(), None
    if df.select_dtypes(np.number).empty :
        return None, df.describe(include=np.object).transpose()
    else:
        return stats_num.transpose(), df.describe(include=np.object).transpose()

@st.cache
def get_info(df):
    return pd.DataFrame({'types': df.dtypes, 'nan': df.isna().sum(), 'nan%': round((df.isna().sum()/len(df))*100,2), 'unique':df.nunique()})




def plot_multivariate(obj_plot, radio_plot):

     

    def pretty(method):
        return method.capitalize()

    if radio_plot == ('Correlation'):
        st.subheader('Heatmap Correlation Plot')
        correlation = st.sidebar.selectbox("Choose the correlation method", ('pearson', 'kendall','spearman'), format_func=pretty)
        cols_list = st.sidebar.multiselect("Select columns",obj_plot.columns)
        st.sidebar.markdown("If None selected, it will plot the correlation of all numeric variables.")
        if st.sidebar.button('Plot correlation chart'):
            fig = obj_plot.Corr(cols_list, correlation)
            st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

    def map_func(function):
        dic = {np.mean:'Mean', np.sum:'Sum', np.median:'Median'}
        return dic[function]
    
    if radio_plot == ('Heatmap'):
        st.subheader('Heatmap between vars')
        st.markdown(" In order to plot this chart remember that the order of the selection matters, \
            chooose in order the variables that will build the pivot table: row, column and value.")
        cols_list = st.sidebar.multiselect("Select 3 variables (2 categorical and 1 numeric)",obj_plot.columns, key= 'heatmapvars')
        agg_func = st.sidebar.selectbox("Choose one function to aggregate the data", (np.mean, np.sum, np.median), format_func=map_func)
        if st.sidebar.button('Plot heatmap between vars'):
            fig = obj_plot.heatmap_vars(cols_list, agg_func)
            st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if radio_plot == ('Histogram'):
        st.subheader('Histogram')
        col_hist = st.sidebar.selectbox("Choose main variable", obj_plot.num_vars)
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'hist')
        bins_, range_ = None, None
        bins_ = st.sidebar.slider('Number of bins optional', value = 30)
        range_ = st.sidebar.slider('Choose range optional', int(obj_plot.df[col_hist].min()), int(obj_plot.df[col_hist].max()),\
                (int(obj_plot.df[col_hist].min()),int(obj_plot.df[col_hist].max())))    
        if st.sidebar.button('Plot histogram chart'):
                st.plotly_chart(obj_plot.histogram_num(col_hist, hue_opt, bins_, range_))
        st.set_option('deprecation.showPyplotGlobalUse', False)

    if radio_plot == ('Scatterplot'): 
        st.subheader('Scatter plot')
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.num_vars, key = 'scatter1')
        col_y = st.sidebar.selectbox("Choose y variable (numerical)", obj_plot.num_vars, key = 'scatter2')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key = 'scatter3')
        size_opt = st.sidebar.selectbox("Size (numerical) optional",obj_plot.columns.insert(0,None), key = 'scatter')
        if st.sidebar.button('Plot scatter chart'):
            st.plotly_chart(obj_plot.scatter_plot(col_x,col_y, hue_opt, size_opt))
            st.set_option('deprecation.showPyplotGlobalUse', False)
        st.set_option('deprecation.showPyplotGlobalUse', False)
   
    
    if radio_plot == ('Barplot'):
        st.subheader('Barplot') 
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='barplot1')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns,key='barplot2')
        hue_opt = st.sidebar.selectbox("Hue (categorical/numerical) optional", obj_plot.columns.insert(0,None),key='barplot')
        if st.sidebar.button('Plot barplot chart'):
            st.plotly_chart(obj_plot.bar_plot(col_y,col_x, hue_opt))
        st.set_option('deprecation.showPyplotGlobalUse', False)
   
    
def main():

    st.title('Lung Cancer Prediction ')
    
   
        
    df = pd.read_csv("C:/Users/DELL/Downloads/cancer patient data sets_1.csv")

   
    def basic_info(df):
            st.header("Data")
            st.write('Number of observations', df.shape[0]) 
            st.write('Number of variables', df.shape[1])
            st.write('Number of missing (%)',((df.isna().sum().sum()/df.size)*100).round(2))

        #Visualize data
    basic_info(df)
        
        #Sidebar Menu
    options = ["Summary statistics", "Analysis and Visualization"]
    menu = st.sidebar.selectbox("Menu options", options)

        #Data statistics
    df_info = get_info(df)   
    if (menu == "Summary statistics"):
        df_stat_num, df_stat_obj = get_stats(df)
        st.markdown('**Numerical summary**')
        st.table(df_stat_num)
        st.markdown('**Categorical summary**')
        st.table(df_stat_obj)
        st.markdown('**Missing Values**')
        st.table(df_info)

    eda_plot = EDA(df) 

        # Visualize data

       

    if (menu =="Analysis and Visualization" ):
        st.header("Analysis and Visualization")

        st.markdown('Here you can visualize your data by choosing one of the chart options available on the sidebar!')
               
        st.sidebar.subheader('Data visualization options')
        radio_plot = st.sidebar.radio('Choose plot style', ('Correlation', 'Heatmap', 'Histogram', \
                'Scatterplot', 'Barplot'))

        plot_multivariate(eda_plot, radio_plot)


        st.sidebar.title('Lung Cancer Prediction')
        


if __name__ == '__main__':
    main()