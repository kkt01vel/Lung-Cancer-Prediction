
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import pybase64
sns.set()
px.defaults.width = 900
px.defaults.height = 500

df1 = pd.read_csv("E:/raw_data.csv")

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
def get_na_info(df_preproc, df, col):
    raw_info = pd_of_stats(df, col)
    prep_info = pd_of_stats(df_preproc,col)
    return raw_info.join(prep_info, lsuffix= '_raw', rsuffix='_prep').T

@st.cache     
def pd_of_stats(df,col):
    #Descriptive Statistics
    stats = dict()
    stats['Mean']  = df[col].mean()
    stats['Std']   = df[col].std()
    stats['Var'] = df[col].var()
    stats['Kurtosis'] = df[col].kurtosis()
    stats['Skewness'] = df[col].skew()
    stats['Coefficient Variance'] = stats['Std'] / stats['Mean']
    return pd.DataFrame(stats, index = col).T.round(2)

@st.cache   
def pf_of_info(df,col):
    info = dict()
    info['Type'] =  df[col].dtypes
    info['Unique'] = df[col].nunique()
    info['n_zeros'] = (len(df) - np.count_nonzero(df[col]))
    info['p_zeros'] = round(info['n_zeros'] * 100 / len(df),2)
    info['nan'] = df[col].isna().sum()
    info['p_nan'] =  (df[col].isna().sum() / df.shape[0]) * 100
    return pd.DataFrame(info, index = col).T.round(2)
    
   

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
    return pd.DataFrame({'types': df.dtypes, 'nan': df.isna().sum(), 'nan%': round((df.isna().sum()/len(df))*100,2)})


def plot_multivariate(obj_plot, radio_plot):

        
    def map_func(function):
        dic = {np.mean:'Mean', np.sum:'Sum', np.median:'Median'}
        return dic[function]
    
        
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
   
    

def main():

    st.title('Lung Cancer Prediction :bar_chart:')
    
   
        
    df = pd.read_csv('E:/Clean_data.csv')
    

   
    def basic_info(df):
            left_column, middle_column, right_column = st.columns(3)
            with left_column:
                st.subheader('Total Number of observations')
                st.subheader(df.shape[0])
            with middle_column:
                st.subheader("Total Number of variables")
                st.subheader(df.shape[1])
            with right_column:
                st.subheader("Number of missing variables in percentage ")
                st.subheader(((df.isna().sum().sum()/df.size)*100).round(2))
            
        #Visualize data
    basic_info(df)
        
        #Sidebar Menu
    options = ["Raw Dataset","Clean Dataset","Descriptive Statistics","Insights about the data", "Analysis","Visualization"]
    menu = st.sidebar.selectbox("Menu options", options)

        #Data statistics
    df1_info = get_info(df1)    
    df_info = get_info(df)   
    if (menu == "Insights about the data"):
        df_stat_num, df_stat_obj = get_stats(df)
        st.header("Information about the dataset")
        st.markdown('**Numerical summary**')
        st.table(df_stat_num)
        st.markdown('**Categorical summary**')
        st.table(df_stat_obj)
        st.sidebar.title('**Statistical Sumary**')

    eda_plot = EDA(df) 

        # Visualize data
   
    if (menu =="Visualization" ):
        st.sidebar.title('**Visualization of Lung Cancer Prediction dataset**')
        st.header("Visualization")

        st.markdown('User can visualize different variables using the options given in sidebar')
               
        st.sidebar.subheader('**Data visualization options**')
        radio_plot = st.sidebar.radio('Choose plot style', ('Histogram','Scatterplot'))

        plot_multivariate(eda_plot, radio_plot)

    
    if (menu == "Raw Dataset"):
       st.sidebar.title('Raw Dataset') 
       st.header("Raw Dataset given for analysis")
       st.dataframe(data=df1, width=None, height=None)
       st.table(df1_info)
       
       
    if (menu == "Clean Dataset"):
       st.sidebar.title('After Cleaning')
       st.header("Dataset after cleaning")
       st.dataframe(df.style.highlight_max(axis=0),width=None, height=None)
       st.table(df_info)
       
    if (menu =="Analysis" ):
        st.sidebar.title('Correlation Analysis')
        st.header("Correlation between variables")
        st.markdown('User can correlate different variables using the options given in sidebar')
        st.sidebar.subheader('**Correlation between variables**')
        st.subheader('Heatmap Correlation Plot')
        correlation = st.sidebar.selectbox("Choose the correlation method", ('pearson', 'kendall','spearman'))
        cols_list = st.sidebar.multiselect("Select columns",eda_plot.columns)
        st.sidebar.markdown("If None selected, it will plot the correlation of all numeric variables.")
        if st.sidebar.button('Plot correlation chart'):
            fig = eda_plot.Corr(cols_list, correlation)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            
    if (menu =="Descriptive Statistics" ):
        st.header("Descriptive Statisstics")
        st.markdown("Provides summary statistics of only one variable in the dataset.")
        main_var = st.selectbox("Choose one variable to analyze:", df.columns.insert(0,None))

        if main_var in df.columns: 
            if main_var != None:
                st.subheader("Variable info")
                st.table(pf_of_info(df, [main_var]).T)
                st.subheader("Descriptive Statistics")
                st.table((pd_of_stats(df, [main_var])).T)
                
                
            
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = pybase64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};pybase64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local("C:/Users/DELL/Downloads/bg3.jpg")                    
        
if __name__ == '__main__':
    main()

'''hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
'''
