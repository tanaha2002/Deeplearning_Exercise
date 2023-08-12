import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle

#set application always wide mode
st.set_page_config(layout="wide")

st.title('Iris Classification App')
#list model load
models = ['Iris-Classification-LR.pkl']

#cache data
@st.cache_data
def load_data():
    df = pd.read_csv('iris.data', header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    return df

iris = load_data()

col1, col2, col3,col4 = st.columns([0.5,1.25,2.25,1.25])
def header():
    with col2:
        st.markdown('###### 1. Select Input Type')
        global Input_type 
        Input_type= st.selectbox('Input Type?', ('User Input', 'Loading File'))
    with col3:
        global col3_1, col3_2 
        col3_1,col3_2 = st.columns(2)
        with col3_1:
            st.markdown('###### 2. Select Model')
            global Model_type 
            Model_type = st.selectbox(
                'Model Select?',
                (models))
            
        with col3_2:
            #selected feature
            feature_selection(iris)
            #expander to show model information
            
            st.write('Model Information')
            expander = st.expander("See explanation")
            expander.write(
                "Accuracy: 0.97 \
                \n F1 Score: 0.9778718400940623 \
                \n Confusion Matrix: "
            )
            #show confusion matrix
            expander.image('confusion_matrix_LOGRE.png')
    with col4:
        st.markdown('###### 3. Predict')



def user_input():
    global col1_1, col1_2
    col1_1, col1_2 = st.columns(2)
    with col1_2:
        mode = st.radio('Input Type',('Number Input','Slider Input'))
    with col1_1:
        if mode == 'Number Input':
            Sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            Sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            Petal_length = st.number_input('Petal Length', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            Petal_width = st.number_input('Petal Width', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        else:
            Sepal_length = st.slider('Sepal Length', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            Sepal_width = st.slider('Sepal Width', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            Petal_length = st.slider('Petal Length', min_value=0.0, max_value=20.0, value=5.0, step=0.1)
            Petal_width = st.slider('Petal Width', min_value=0.0, max_value=20.0, value=5.0, step=0.1)

    return pd.DataFrame([[Sepal_length, Sepal_width, Petal_length, Petal_width]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])


def new_data_point(df,fig,new_data):

    new_data = new_data.values.tolist()[0]
    fig.add_trace(go.Scatter(x=[new_data[df.columns.get_loc(feature_1)]], y=[new_data[df.columns.get_loc(feature_2)]], mode='markers', marker=dict(size=10, color='red'), name='New Data'))
#using to visual graph model
def graph_visual(df,new_data):

    
    

    # Vẽ biểu đồ 2D
    fig = px.scatter(df, x=feature_1, y=feature_2, color='class')
    #add new data to one point on graph
    new_data_point(df,fig,new_data)
    fig.update_layout(
        width=300,  # Adjust the width as needed
        height=150,
        margin=dict(t=0, l=0, r=0, b=0),
        )
    st.plotly_chart(fig)


def user_input_graph(new_data):
    with col3_1:
        graph_visual(iris,new_data)

def loading_file():
    global upload_file
    upload_file = st.file_uploader('Upload File', type=['csv'])
    if upload_file is not None:
        df = pd.read_csv(upload_file,header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        return df
    else:
        return None
    


#loading selected feature to visual graph model
def feature_selection(df):
    st.write('#')
    global feature_1
    feature_1 =  st.selectbox('Feature 1', df.columns[:-1])
    global feature_2 
    feature_2 = st.selectbox('Feature 2', df.columns[:-1])




def plot_predict(model,new_data):

    #rename column to fit with feature names
    new_data.columns = ['sepal_length (cm)', 'sepal_width (cm)', 'petal_length (cm)', 'petal_width (cm)']
    #predict
    predict = model.predict_proba(new_data)
    #create a columns name for class and columns value for value of predict
    data = {'Class': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 'Value': predict[0]}
    

    

    
    #plot predict


    # Create a smaller bar chart using Plotly
    fig = px.bar(data, x='Class', y='Value',color = 'Class',  range_y=[0, 1])
    fig.update_layout(
        width=300,  # Adjust the width as needed
        height=150,
        margin=dict(t=15, l=0, r=0, b=15),
        xaxis=dict(
            showticklabels=False
        ),
        showlegend=False,  # Hide the legend
        #hide x label
        xaxis_title=None,
    #add number on top of bar
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(yi),
                xanchor='center',
                yanchor='bottom',
                showarrow=False,
                font=dict(
                    family='Courier New, monospace',
                    size=12,
                    color='#ffffff'
                ),
            ) for xi, yi in zip(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [round(predict[0][0],2),round(predict[0][1],2), round(predict[0][2],2)])
        ],
        )





    #resize plot using css
    # st.write('<style>div[class="user-select-none svg-container"] { height: 100px !important; }</style>', unsafe_allow_html=True)
    # Display the Plotly plot using st.plotly_chart
    st.plotly_chart(fig)
def load_model():
    with open(Model_type, 'rb') as file:
        model = pickle.load(file)
    return model

def spacer():
    with col3_1:
        st.write('#')
        st.write('#')
        st.write('#')
        st.write('#')


    with col4:
        st.write('#')
        st.write('#')
        st.write('#')
        st.write('#')
        st.write('#')
        st.write('#')







def show_data_each_row(df,model,new_data):
    
    spacer()
    #for each row in new data, create a new dataframe to st.write
    for i in range(len(new_data)):
        with col2:
            st.write(new_data.iloc[i:i+1])
            st.write('#')
            st.write('#')
        with col3_1:
            graph_visual(df,new_data.iloc[i:i+1])
        with col4:
            plot_predict(model,new_data.iloc[i:i+1])




        
def week1():
    header()
    if Input_type == 'User Input':
        with col2:
            new_data = user_input()
        with col3:
            user_input_graph(new_data)
        with col4:
            model = load_model()
            st.write('#')
            plot_predict(model,new_data)
    elif Input_type == 'Loading File':
        with col2:
            data = loading_file()
            if data is not None:
                show_data_each_row(iris,load_model(),data)



        
        


def sidebar():
    #hide the default menu
    hide_streamlit_style = ""
    st.sidebar.title('Weekly Exercise')
    st.sidebar.write('##')
    selected_week = st.sidebar.selectbox('Select Week', ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6'])
    return selected_week



#main
if __name__ == '__main__':
    selected_week = sidebar()
    if selected_week == 'Week 1':
        week1()







    

